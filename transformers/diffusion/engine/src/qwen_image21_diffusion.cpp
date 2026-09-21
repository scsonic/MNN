//
//  qwen_image21_diffusion.cpp
//  Qwen-Image-2.1 text-to-image on MNN.
//
//  Pipeline (mirrors diffusers QwenImage21Pipeline with use_kv_cache=True, no CFG):
//    1. Qwen3-VL-8B encodes the chat-templated prompt; the last decoder layer's pre-norm hidden states are kept
//       and the system-prompt tokens dropped.
//    2. txt_in + dit (prefix pass, timestep 0, causal) produce every block's text K/V once.
//    3. Each Euler step runs only the image tokens: img_in + dit (step pass) against the cached text K/V.
//    4. The VAE decodes the 64-channel latents to RGBA.
//
#include <cmath>
#include <cstring>
#include <fstream>
#include "diffusion/qwen_image21_diffusion.hpp"
#include "llm/llm.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <cv/cv.hpp>

#if defined(_MSC_VER)
#include <Windows.h>
#else
#include <sys/time.h>
#endif

using namespace CV;

namespace MNN {
namespace DIFFUSION {

namespace {
constexpr int kLayers = 32;
constexpr int kHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kDim = 4096;
constexpr int kLatentC = 64;
constexpr float kMaskNeg = -30000.0f;
const char* kSystemPrompt = "Comprehend and analyze the provided prompt.";

int64_t nowUs() {
#if defined(_MSC_VER)
    return GetTickCount64() * 1000;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
#endif
}

VARP hostCopy(VARP v) {
    auto conv = _Convert(v, NCHW);
    auto info = conv->getInfo();
    auto dst = _Input(info->dim, NCHW, halide_type_of<float>());
    ::memcpy(dst->writeMap<float>(), conv->readMap<float>(), info->size * sizeof(float));
    dst.fix(VARP::CONSTANT);
    return dst;
}

VARP hostTensor(const std::vector<int>& dims, const float* data) {
    auto v = _Input(dims, NCHW, halide_type_of<float>());
    size_t n = 1;
    for (auto d : dims) n *= d;
    ::memcpy(v->writeMap<float>(), data, n * sizeof(float));
    return v;
}
} // namespace

QwenImage21Diffusion::QwenImage21Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType,
                                           int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU,
                                           bool vaeOnCPU, DiffusionGpuMemoryMode gpuMemoryMode,
                                           DiffusionPrecisionMode precisionMode, DiffusionCFGMode cfgMode, int numThreads)
    : Diffusion(modelPath, modelType, backendType, memoryMode, imageWidth, imageHeight, textEncoderOnCPU, vaeOnCPU,
                gpuMemoryMode, precisionMode, cfgMode, numThreads) {
    // Attention masks use a finite negative value, so fp16 is safe and is the default for the DiT.
    if (mPrecisionMode == PRECISION_AUTO) {
        mPrecisionMode = PRECISION_LOW;
    }
    int w = mImageWidth > 0 ? mImageWidth : 512;
    int h = mImageHeight > 0 ? mImageHeight : 512;
    mImageWidth = std::max(256, (w / 32) * 32);
    mImageHeight = std::max(256, (h / 32) * 32);
    mLatentW = mImageWidth / 16;
    mLatentH = mImageHeight / 16;
    MNN_PRINT("[QwenImage21] image=%dx%d latent tokens=%dx%d\n", mImageWidth, mImageHeight, mLatentH, mLatentW);
}

QwenImage21Diffusion::~QwenImage21Diffusion() {
    mTxtIn.reset();
    mImgIn.reset();
    mDitStep.reset();
    mVae.reset();
    mTextEncoder.reset();
}

std::shared_ptr<Module> QwenImage21Diffusion::loadModule(const std::string& file, const std::vector<std::string>& inputs,
                                                         const std::vector<std::string>& outputs,
                                                         std::shared_ptr<Executor::RuntimeManager> rt) {
    AUTOTIME;
    Module::Config config;
    config.shapeMutable = true;
    config.rearrange = true;
    std::string path = mModelPath + "/" + file;
    rt->setExternalFile(path + ".weight");
    std::shared_ptr<Module> m(Module::load(inputs, outputs, path.c_str(), rt, &config));
    rt->setExternalFile("");
    if (!m) {
        MNN_ERROR("[QwenImage21] failed to load %s\n", path.c_str());
    } else {
        MNN_PRINT("[QwenImage21] loaded %s\n", file.c_str());
    }
    return m;
}

bool QwenImage21Diffusion::load() {
    AUTOTIME;
    if (!initRuntimeManagers(/*gpuBufferMode=*/true)) {
        return false;
    }
    if (mMemoryMode == 1) {
        // keep everything resident
        mTxtIn = loadModule("txt_in.mnn", {"txt"}, {"txt_h"}, runtime_manager_);
        mImgIn = loadModule("img_in.mnn", {"lat"}, {"img_h"}, runtime_manager_);
        mVae = loadModule("vae_decoder.mnn", {"latent"}, {"image"},
                          runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_);
        return mTxtIn && mImgIn && mVae;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- text encoder

VARP QwenImage21Diffusion::encodePrompt(const std::string& prompt) {
    AUTOTIME;
    if (!mTextEncoder) {
        std::string cfg = mModelPath + "/text_encoder/te_config.json";
        mTextEncoder.reset(Transformer::Llm::createLLM(cfg), Transformer::Llm::destroy);
        if (!mTextEncoder) {
            MNN_ERROR("[QwenImage21] cannot create text encoder from %s\n", cfg.c_str());
            return nullptr;
        }
        std::string backend = (mTextEncoderOnCPU || mBackendType == MNN_FORWARD_CPU) ? "cpu" : "opencl";
        std::string over = "{\"backend_type\":\"" + backend + "\",\"thread_num\":" +
                           std::to_string(backend == "cpu" ? mNumThreads : 68) + "}";
        mTextEncoder->set_config(over);
        if (!mTextEncoder->load()) {
            MNN_ERROR("[QwenImage21] text encoder load failed\n");
            mTextEncoder.reset();
            return nullptr;
        }
    }
    mTextEncoder->reset();
    std::string text = std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n<|im_start|>user\n" +
                       (prompt.empty() ? std::string(" ") : prompt) + "<|im_end|>\n<|im_start|>assistant\n";
    auto ids = mTextEncoder->tokenizer_encode(text);
    MNN_PRINT("[QwenImage21] prompt tokens=%d (drop %d)\n", (int)ids.size(), mDropIdx);
    if ((int)ids.size() <= mDropIdx) {
        MNN_ERROR("[QwenImage21] prompt too short after tokenization\n");
        return nullptr;
    }
    if (mTextEncoder->forward(ids) == nullptr) {
        MNN_ERROR("[QwenImage21] text encoder forward failed\n");
        return nullptr;
    }
    auto outputs = mTextEncoder->getOutputs();
    int index = mTextEncoder->getOutputIndex(mTeOutputName);
    if (index < 0 || index >= (int)outputs.size()) {
        MNN_ERROR("[QwenImage21] text encoder has no output %s; check text_encoder/te_llm_config.json\n",
                  mTeOutputName.c_str());
        return nullptr;
    }
    auto hidden = _Convert(outputs[index], NCHW);  // [1, T, 4096]
    auto info = hidden->getInfo();
    int T = info->dim[1];
    int L = T - mDropIdx;
    auto result = _Input({1, L, kDim}, NCHW, halide_type_of<float>());
    ::memcpy(result->writeMap<float>(), hidden->readMap<float>() + (size_t)mDropIdx * kDim, (size_t)L * kDim * sizeof(float));
    result.fix(VARP::CONSTANT);
    outputs.clear();
    if (mMemoryMode != 1) {
        mTextEncoder.reset();
        MNN_PRINT("[QwenImage21] text encoder unloaded\n");
    }
    return result;
}

// ---------------------------------------------------------------------------------------------- DiT

void QwenImage21Diffusion::ropeTables(int textLen, int hTokens, int wTokens, std::vector<float>& cosTab,
                                      std::vector<float>& sinTab) {
    // QwenImage21Rope: axes (frame 16, height 56, width 56), theta 10000, interleaved complex pairs.
    const int axes[3] = {16, 56, 56};
    int n = textLen + hTokens * wTokens;
    cosTab.resize((size_t)n * 64);
    sinTab.resize((size_t)n * 64);
    auto fill = [&](int row, int f, int hh, int ww) {
        int pos[3] = {f, hh, ww};
        int col = 0;
        for (int a = 0; a < 3; ++a) {
            for (int i = 0; i < axes[a] / 2; ++i) {
                double inv = 1.0 / std::pow(10000.0, (2.0 * i) / axes[a]);
                double ang = pos[a] * inv;
                cosTab[(size_t)row * 64 + col] = (float)std::cos(ang);
                sinTab[(size_t)row * 64 + col] = (float)std::sin(ang);
                ++col;
            }
        }
    };
    for (int i = 0; i < textLen; ++i) {
        fill(i, i, i, i);
    }
    int row = textLen;
    for (int h = -(hTokens - hTokens / 2); h < hTokens / 2; ++h) {
        for (int w = -(wTokens - wTokens / 2); w < wTokens / 2; ++w) {
            fill(row++, textLen, h, w);
        }
    }
}

std::vector<float> QwenImage21Diffusion::sigmas(int steps, int imageSeqLen) {
    // FlowMatchEulerDiscreteScheduler: exponential time shift (mu from image_seq_len) + shift_terminal 0.02
    const double baseSeq = 256, maxSeq = 8192, baseShift = 0.5, maxShift = 0.9, terminal = 0.02;
    double m = (maxShift - baseShift) / (maxSeq - baseSeq);
    double mu = imageSeqLen * m + baseShift - m * baseSeq;
    std::vector<double> s(steps);
    for (int i = 0; i < steps; ++i) {
        double lin = steps == 1 ? 1.0 : 1.0 + (1.0 / steps - 1.0) * i / (steps - 1);
        s[i] = std::exp(mu) / (std::exp(mu) + (1.0 / lin - 1.0));
    }
    double scale = (1.0 - s[steps - 1]) / (1.0 - terminal);
    std::vector<float> out(steps + 1);
    for (int i = 0; i < steps; ++i) {
        out[i] = (float)(1.0 - (1.0 - s[i]) / scale);
    }
    out[steps] = 0.0f;
    return out;
}

VARP QwenImage21Diffusion::buildPrefixCache(VARP textHidden) {
    AUTOTIME;
    int L = textHidden->getInfo()->dim[1];
    if (!mTxtIn) {
        mTxtIn = loadModule("txt_in.mnn", {"txt"}, {"txt_h"}, runtime_manager_);
    }
    if (!mTxtIn) return nullptr;
    auto txtH = mTxtIn->onForward({textHidden});
    if (txtH.empty()) return nullptr;
    auto hidden = hostCopy(txtH[0]);
    if (mMemoryMode != 1) mTxtIn.reset();

    std::vector<float> cosTab, sinTab;
    ropeTables(L, mLatentH, mLatentW, cosTab, sinTab);
    std::vector<float> mask((size_t)L * (L + 1), kMaskNeg);
    for (int q = 0; q < L; ++q) {
        for (int k = 0; k <= q; ++k) mask[(size_t)q * (L + 1) + 1 + k] = 0.0f;
    }
    std::vector<float> pastZero((size_t)kLayers * 2 * kHeads * kHeadDim, 0.0f);
    float t0 = 0.0f;

    auto prefix = loadModule("dit.mnn", {"hidden", "timestep", "rope_cos", "rope_sin", "past_kv", "attn_mask"},
                             {"present_kv"}, runtime_manager_);
    if (!prefix) return nullptr;
    int64_t st = nowUs();
    auto out = prefix->onForward({hidden, hostTensor({1}, &t0), hostTensor({L, 64}, cosTab.data()),
                                  hostTensor({L, 64}, sinTab.data()),
                                  hostTensor({kLayers, 2, 1, kHeads, kHeadDim}, pastZero.data()),
                                  hostTensor({1, 1, L, L + 1}, mask.data())});
    if (out.empty()) return nullptr;
    auto kv = hostCopy(out[0]);
    MNN_PRINT("[QwenImage21] prefix pass L=%d: %.2f s\n", L, (nowUs() - st) / 1e6);
    out.clear();
    prefix.reset();
    return kv;
}

VARP QwenImage21Diffusion::denoise(VARP prefixKV, int textLen, int steps, int seed,
                                   std::function<void(int)> progressCallback) {
    AUTOTIME;
    const int N = mLatentH * mLatentW;
    if (!mImgIn) mImgIn = loadModule("img_in.mnn", {"lat"}, {"img_h"}, runtime_manager_);
    if (!mDitStep) {
        mDitStep = loadModule("dit.mnn", {"hidden", "timestep", "rope_cos", "rope_sin", "past_kv", "attn_mask"},
                              {"out"}, runtime_manager_);
    }
    if (!mImgIn || !mDitStep) return nullptr;

    std::vector<float> cosTab, sinTab;
    ropeTables(textLen, mLatentH, mLatentW, cosTab, sinTab);
    auto ropeCos = hostTensor({N, 64}, cosTab.data() + (size_t)textLen * 64);
    auto ropeSin = hostTensor({N, 64}, sinTab.data() + (size_t)textLen * 64);
    ropeCos.fix(VARP::CONSTANT);
    ropeSin.fix(VARP::CONSTANT);
    std::vector<float> zeroMask((size_t)N * (textLen + N), 0.0f);
    auto mask = hostTensor({1, 1, N, textLen + N}, zeroMask.data());
    mask.fix(VARP::CONSTANT);

    std::vector<float> latents((size_t)N * kLatentC);
    generateLatentNoise(latents.data(), (int)latents.size(), seed);
    auto sig = sigmas(steps, N);
    auto latVar = _Input({1, N, kLatentC}, NCHW, halide_type_of<float>());
    auto tVar = _Input({1}, NCHW, halide_type_of<float>());

    for (int i = 0; i < steps; ++i) {
        int64_t st = nowUs();
        ::memcpy(latVar->writeMap<float>(), latents.data(), latents.size() * sizeof(float));
        tVar->writeMap<float>()[0] = sig[i];
        auto imgH = mImgIn->onForward({latVar});
        if (imgH.empty()) return nullptr;
        auto out = mDitStep->onForward({imgH[0], tVar, ropeCos, ropeSin, prefixKV, mask});
        if (out.empty()) return nullptr;
        auto v = _Convert(out[0], NCHW);
        const float* vp = v->readMap<float>();
        if (!vp) return nullptr;
        float dt = sig[i + 1] - sig[i];
        bool bad = false;
        for (size_t k = 0; k < latents.size(); ++k) {
            latents[k] += dt * vp[k];
            if (!std::isfinite(latents[k])) bad = true;
        }
        MNN_PRINT("[QwenImage21] step %d/%d sigma=%.4f %.2f s%s\n", i + 1, steps, sig[i], (nowUs() - st) / 1e6,
                  bad ? " (non-finite!)" : "");
        if (mBackendType == MNN_FORWARD_OPENCL) {
            ExecutorScope::Current()->gc(Executor::PART);
        }
        if (progressCallback) progressCallback(10 + 80 * (i + 1) / steps);
    }
    if (mMemoryMode == 0) {
        mDitStep.reset();
        mImgIn.reset();
    }
    return hostTensor({1, N, kLatentC}, latents.data());
}

// ---------------------------------------------------------------------------------------------- VAE

VARP QwenImage21Diffusion::decode(VARP packedLatents) {
    AUTOTIME;
    const int N = mLatentH * mLatentW;
    auto vaeRt = runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_;
    if (!mVae) mVae = loadModule("vae_decoder.mnn", {"latent"}, {"image"}, vaeRt);
    if (!mVae) return nullptr;
    // [1, N, 64] -> [1, 64, H, W]
    std::vector<float> nchw((size_t)kLatentC * N);
    const float* src = packedLatents->readMap<float>();
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < kLatentC; ++c) nchw[(size_t)c * N + n] = src[(size_t)n * kLatentC + c];
    }
    int64_t st = nowUs();
    auto out = mVae->onForward({hostTensor({1, kLatentC, mLatentH, mLatentW}, nchw.data())});
    if (out.empty()) return nullptr;
    auto img = hostCopy(out[0]);
    MNN_PRINT("[QwenImage21] vae decode %.2f s\n", (nowUs() - st) / 1e6);
    if (mMemoryMode != 1) mVae.reset();
    return img;
}

bool QwenImage21Diffusion::saveRGBA(VARP image, const std::string& path) {
    auto info = image->getInfo();
    int C = info->dim[1], H = info->dim[2], W = info->dim[3];
    const float* p = image->readMap<float>();
    auto hwc = _Input({H, W, C}, NHWC, halide_type_of<uint8_t>());
    auto dst = hwc->writeMap<uint8_t>();
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < H * W; ++i) {
            float v = (p[(size_t)c * H * W + i] * 0.5f + 0.5f) * 255.0f;
            v = std::min(255.0f, std::max(0.0f, std::round(v)));
            dst[(size_t)i * C + c] = (uint8_t)v;
        }
    }
    // imwrite swaps BGR->RGB for 3 channels only; 4-channel data is written as RGBA as is.
    return imwrite(path, hwc);
}

// ---------------------------------------------------------------------------------------------- run

bool QwenImage21Diffusion::run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed,
                               float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath) {
    AUTOTIME;
    try {
        if (iterNum < 1) iterNum = 20;
        if (iterNum > 100) iterNum = 100;
        int seed = randomSeed < 0 ? (int)(nowUs() & 0x7fffffff) : randomSeed;
        int64_t st = nowUs();
        auto text = encodePrompt(prompt);
        if (text.get() == nullptr) return false;
        int L = text->getInfo()->dim[1];
        MNN_PRINT("[QwenImage21] text encoder done: %.2f s\n", (nowUs() - st) / 1e6);
        if (progressCallback) progressCallback(5);
        auto kv = buildPrefixCache(text);
        if (kv.get() == nullptr) return false;
        if (progressCallback) progressCallback(10);
        auto latents = denoise(kv, L, iterNum, seed, progressCallback);
        kv = nullptr;
        if (latents.get() == nullptr) return false;
        auto image = decode(latents);
        if (image.get() == nullptr) return false;
        bool ok = saveRGBA(image, outputPath);
        MNN_PRINT("[QwenImage21] %s %s (seed %d, total %.1f s)\n", ok ? "saved" : "FAILED to save",
                  outputPath.c_str(), seed, (nowUs() - st) / 1e6);
        if (progressCallback) progressCallback(100);
        return ok;
    } catch (const std::exception& e) {
        MNN_ERROR("[QwenImage21] exception: %s\n", e.what());
        return false;
    }
}

bool QwenImage21Diffusion::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed,
                               std::function<void(int)> progressCallback) {
    return run(prompt, imagePath, iterNum, randomSeed, 1.0f, progressCallback, "");
}

bool QwenImage21Diffusion::run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                               const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                               bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) {
    MNN_ERROR("[QwenImage21] input_embeds interface is not supported\n");
    return false;
}

} // namespace DIFFUSION
} // namespace MNN
