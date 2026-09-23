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
constexpr int kImagePadId = 151655;  // <|image_pad|>
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

// QWEN_IMAGE21_DUMP=<dir> writes intermediate tensors as raw float32 for offline comparison.
void dump(const char* name, const float* data, size_t n) {
    const char* dir = getenv("QWEN_IMAGE21_DUMP");
    if (!dir) return;
    std::ofstream f(std::string(dir) + "/" + name + ".f32", std::ios::binary);
    f.write((const char*)data, n * sizeof(float));
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
    if (mBackendType == MNN_FORWARD_CPU) {
        // CPU Memory_Low routes quantized convs through dynamic int8 GEMM, which returned garbage for long
        // prefixes (edit mode, P ~ 1000) on SME2 hosts; use Memory_Normal for the CPU DiT.
        ScheduleConfig config;
        BackendConfig bc;
        config.type = MNN_FORWARD_CPU;
        config.numThread = mNumThreads;
        bc.memory = BackendConfig::Memory_Normal;
        bc.precision = mPrecisionMode == PRECISION_LOW ? BackendConfig::Precision_Low : BackendConfig::Precision_High;
        config.backendConfig = &bc;
        runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    }
    if (runtime_manager_vae_cpu_) {
        // The exported VAE is fp16-safe (rescaled residual stream); fp16 halves its CPU memory (~2.5 GB at 512^2).
        ScheduleConfig config;
        BackendConfig bc;
        config.type = MNN_FORWARD_CPU;
        config.numThread = mNumThreads;
        bc.memory = BackendConfig::Memory_Low;
        bc.precision = BackendConfig::Precision_Low;
        config.backendConfig = &bc;
        runtime_manager_vae_cpu_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    }
    // Winograd pre-transforms the VAE's 3x3 weights (up to 1152 channels) into several GB; keep it off.
    for (auto rt : {runtime_manager_, runtime_manager_cpu_, runtime_manager_vae_cpu_}) {
        if (rt) rt->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
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
    VARP result;
    {
        // Every VARP that references text-encoder memory must die before the encoder is unloaded.
        auto outputs = mTextEncoder->getOutputs();
        int index = mTextEncoder->getOutputIndex(mTeOutputName);
        if (index < 0 || index >= (int)outputs.size()) {
            MNN_ERROR("[QwenImage21] text encoder has no output %s; check text_encoder/te_llm_config.json\n",
                      mTeOutputName.c_str());
            return nullptr;
        }
        auto hidden = _Convert(outputs[index], NCHW);  // [1, T, 4096]
        int T = hidden->getInfo()->dim[1];
        int L = T - mDropIdx;
        std::vector<float> host((size_t)L * kDim);
        ::memcpy(host.data(), hidden->readMap<float>() + (size_t)mDropIdx * kDim, host.size() * sizeof(float));
        dump("text_hidden", host.data(), host.size());
        dump("text_hidden_full", hidden->readMap<float>(), (size_t)T * kDim);
        hidden = nullptr;
        outputs.clear();
        mTextEncoder->reset();
        ExecutorScope scope(Executor::getGlobalExecutor());
        result = _Input({1, L, kDim}, NCHW, halide_type_of<float>());
        ::memcpy(result->writeMap<float>(), host.data(), host.size() * sizeof(float));
        result.fix(VARP::CONSTANT);
    }
    if (mMemoryMode != 1) {
        mTextEncoder.reset();
        MNN_PRINT("[QwenImage21] text encoder unloaded\n");
    }
    return result;
}

// ---------------------------------------------------------------------------------------------- DiT

void QwenImage21Diffusion::ropeFromPositions(const std::vector<int>& frame, const std::vector<int>& hpos,
                                             const std::vector<int>& wpos, std::vector<float>& cosTab,
                                             std::vector<float>& sinTab) {
    // QwenImage21Rope: axes (frame 16, height 56, width 56), theta 10000, interleaved complex pairs.
    const int axes[3] = {16, 56, 56};
    size_t n = frame.size();
    cosTab.resize(n * 64);
    sinTab.resize(n * 64);
    for (size_t row = 0; row < n; ++row) {
        int pos[3] = {frame[row], hpos[row], wpos[row]};
        int col = 0;
        for (int a = 0; a < 3; ++a) {
            for (int i = 0; i < axes[a] / 2; ++i) {
                double ang = pos[a] / std::pow(10000.0, (2.0 * i) / axes[a]);
                cosTab[row * 64 + col] = (float)std::cos(ang);
                sinTab[row * 64 + col] = (float)std::sin(ang);
                ++col;
            }
        }
    }
}

namespace {
// Centered latent grid positions of one image block (QwenImage21Rope).
void appendGrid(int frameValue, int h, int w, std::vector<int>& f, std::vector<int>& hp, std::vector<int>& wp) {
    for (int y = -(h - h / 2); y < h / 2; ++y) {
        for (int x = -(w - w / 2); x < w / 2; ++x) {
            f.push_back(frameValue);
            hp.push_back(y);
            wp.push_back(x);
        }
    }
}
void appendText(int start, int count, std::vector<int>& f, std::vector<int>& hp, std::vector<int>& wp) {
    for (int i = 0; i < count; ++i) {
        f.push_back(start + i);
        hp.push_back(start + i);
        wp.push_back(start + i);
    }
}
} // namespace

void QwenImage21Diffusion::ropeTables(int textLen, int hTokens, int wTokens, std::vector<float>& cosTab,
                                      std::vector<float>& sinTab) {
    std::vector<int> f, hp, wp;
    appendText(0, textLen, f, hp, wp);
    appendGrid(textLen, hTokens, wTokens, f, hp, wp);
    ropeFromPositions(f, hp, wp, cosTab, sinTab);
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

// The K/V cache is one tensor per layer ("past_kv_0".."past_kv_31"), not a single [32,2,P,...] blob: that blob is
// exactly 1 MiB per prefix token, so an image-edit prefix (P > 1000) exceeds OpenCL's CL_DEVICE_MAX_MEM_ALLOC_SIZE
// (1 GiB on Adreno 740) and the host-staging buffer fails to allocate. Per layer it is P/32 MiB.
std::vector<std::string> QwenImage21Diffusion::kvNames(const char* prefix) {
    std::vector<std::string> names(kLayers);
    for (int i = 0; i < kLayers; ++i) names[i] = std::string(prefix) + std::to_string(i);
    return names;
}

std::vector<VARP> QwenImage21Diffusion::runPrefix(VARP hidden, const std::vector<float>& cosTab,
                                                  const std::vector<float>& sinTab, const std::vector<float>& mask) {
    AUTOTIME;
    int P = hidden->getInfo()->dim[1];
    std::vector<float> pastZero((size_t)2 * kHeads * kHeadDim, 0.0f);
    float t0 = 0.0f;
    auto inputNames = kvNames("past_kv_");
    inputNames.insert(inputNames.begin(), {"hidden", "timestep", "rope_cos", "rope_sin", "attn_mask"});
    auto prefix = loadModule("dit.mnn", inputNames, kvNames("present_kv_"), runtime_manager_);
    if (!prefix) return {};
    std::vector<VARP> ins = {hidden, hostTensor({1}, &t0), hostTensor({P, 64}, cosTab.data()),
                             hostTensor({P, 64}, sinTab.data()), hostTensor({1, 1, P, P + 1}, mask.data())};
    for (int i = 0; i < kLayers; ++i) ins.push_back(hostTensor({2, 1, kHeads, kHeadDim}, pastZero.data()));
    int64_t st = nowUs();
    auto out = prefix->onForward(ins);
    if (out.size() != (size_t)kLayers) return {};
    std::vector<VARP> kv(kLayers);
    for (int i = 0; i < kLayers; ++i) {
        kv[i] = hostCopy(out[i]);
        if (kv[i].get() == nullptr) return {};
        dump(("prefix_kv_" + std::to_string(i)).c_str(), kv[i]->readMap<float>(), kv[i]->getInfo()->size);
    }
    MNN_PRINT("[QwenImage21] prefix pass P=%d: %.2f s (K/V cache %d MB)\n", P, (nowUs() - st) / 1e6,
              (int)((size_t)kLayers * 2 * P * kHeads * kHeadDim * sizeof(float) / (1 << 20)));
    out.clear();
    prefix.reset();
    return kv;
}

VARP QwenImage21Diffusion::embedText(VARP textHidden) {
    if (!mTxtIn) mTxtIn = loadModule("txt_in.mnn", {"txt"}, {"txt_h"}, runtime_manager_);
    if (!mTxtIn) return nullptr;
    auto txtH = mTxtIn->onForward({textHidden});
    if (txtH.empty()) return nullptr;
    auto hidden = hostCopy(txtH[0]);
    if (mMemoryMode != 1) mTxtIn.reset();
    return hidden;
}

std::vector<VARP> QwenImage21Diffusion::buildPrefixCache(VARP textHidden) {
    int L = textHidden->getInfo()->dim[1];
    auto hidden = embedText(textHidden);
    if (hidden.get() == nullptr) return {};
    dump("txt_h", hidden->readMap<float>(), hidden->getInfo()->size);
    std::vector<float> cosTab, sinTab;
    ropeTables(L, mLatentH, mLatentW, cosTab, sinTab);
    cosTab.resize((size_t)L * 64);
    sinTab.resize((size_t)L * 64);
    std::vector<float> mask((size_t)L * (L + 1), kMaskNeg);
    for (int q = 0; q < L; ++q) {
        for (int k = 0; k <= q; ++k) mask[(size_t)q * (L + 1) + 1 + k] = 0.0f;
    }
    return runPrefix(hidden, cosTab, sinTab, mask);
}

VARP QwenImage21Diffusion::denoise(const std::vector<VARP>& prefixKV, int prefixLen, const float* cosTarget,
                                   const float* sinTarget, int steps, int seed,
                                   std::function<void(int)> progressCallback) {
    AUTOTIME;
    const int N = mLatentH * mLatentW;
    if (prefixKV.size() != (size_t)kLayers) return nullptr;
    if (!mImgIn) mImgIn = loadModule("img_in.mnn", {"lat"}, {"img_h"}, runtime_manager_);
    if (!mDitStep) {
        auto inputNames = kvNames("past_kv_");
        inputNames.insert(inputNames.begin(), {"hidden", "timestep", "rope_cos", "rope_sin", "attn_mask"});
        mDitStep = loadModule("dit.mnn", inputNames, {"out"}, runtime_manager_);
    }
    if (!mImgIn || !mDitStep) return nullptr;

    auto ropeCos = hostTensor({N, 64}, cosTarget);
    auto ropeSin = hostTensor({N, 64}, sinTarget);
    ropeCos.fix(VARP::CONSTANT);
    ropeSin.fix(VARP::CONSTANT);
    std::vector<float> zeroMask((size_t)N * (prefixLen + N), 0.0f);
    auto mask = hostTensor({1, 1, N, prefixLen + N}, zeroMask.data());
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
        auto imgOut = mImgIn->onForward({latVar});
        if (imgOut.empty()) return nullptr;
        // Own copy: feeding another module's output directly lets the DiT reuse that buffer (seen with long prefixes).
        auto imgH = hostCopy(imgOut[0]);
        imgOut.clear();
        std::vector<VARP> ins = {imgH, tVar, ropeCos, ropeSin, mask};
        ins.insert(ins.end(), prefixKV.begin(), prefixKV.end());
        if (i == 0 && getenv("QWEN_IMAGE21_DUMP")) {
            auto names = kvNames("past_kv_");
            names.insert(names.begin(), {"hidden", "timestep", "rope_cos", "rope_sin", "attn_mask"});
            std::vector<VARP> saved;
            for (size_t k = 0; k < ins.size(); ++k) {
                auto c = hostCopy(ins[k]);
                c->setName(names[k]);
                saved.push_back(c);
            }
            Variable::save(saved, (std::string(getenv("QWEN_IMAGE21_DUMP")) + "/step_input.mnn").c_str());
        }
        auto out = mDitStep->onForward(ins);
        if (out.empty()) return nullptr;
        auto v = _Convert(out[0], NCHW);
        const float* vp = v->readMap<float>();
        if (!vp) return nullptr;
        if (i == 0) {
            dump("noise", latents.data(), latents.size());
            dump("v0", vp, latents.size());
        }
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
        ExecutorScope::Current()->gc(Executor::FULL);
    }
    dump("latents", latents.data(), latents.size());
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
    float memMB = 0.0f;
    vaeRt->getInfo(Interpreter::MEMORY, &memMB);
    MNN_PRINT("[QwenImage21] vae decode %.2f s, runtime memory %.0f MB\n", (nowUs() - st) / 1e6, memMB);
    if (mMemoryMode != 1) mVae.reset();
    return img;
}

namespace {
uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xffffffffu) {
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    }
    return c;
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}
void chunk(std::ofstream& f, const char* type, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> buf(type, type + 4);
    buf.insert(buf.end(), data.begin(), data.end());
    std::vector<uint8_t> head;
    put32(head, (uint32_t)data.size());
    std::vector<uint8_t> tail;
    put32(tail, crc32(buf.data(), buf.size()) ^ 0xffffffffu);
    f.write((const char*)head.data(), 4);
    f.write((const char*)buf.data(), buf.size());
    f.write((const char*)tail.data(), 4);
}
// Minimal RGBA PNG writer (stored deflate blocks); MNN cv's imwrite only handles 1/3 channels.
bool writePngRGBA(const std::string& path, const uint8_t* rgba, int w, int h) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    f.write((const char*)sig, 8);
    std::vector<uint8_t> ihdr;
    put32(ihdr, w); put32(ihdr, h);
    ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});
    chunk(f, "IHDR", ihdr);
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (w * 4 + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba + (size_t)y * w * 4, rgba + (size_t)(y + 1) * w * 4);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (auto c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (size_t pos = 0; pos < raw.size(); pos += 65535) {
        size_t n = std::min<size_t>(65535, raw.size() - pos);
        z.push_back(pos + n == raw.size() ? 1 : 0);
        z.push_back(n & 0xff); z.push_back(n >> 8);
        z.push_back(~n & 0xff); z.push_back((~n >> 8) & 0xff);
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
    }
    put32(z, (b << 16) | a);
    chunk(f, "IDAT", z);
    chunk(f, "IEND", {});
    return (bool)f;
}
} // namespace

bool QwenImage21Diffusion::saveRGBA(VARP image, const std::string& path) {
    auto info = image->getInfo();
    int C = info->dim[1], H = info->dim[2], W = info->dim[3];
    const float* p = image->readMap<float>();
    std::vector<uint8_t> rgba((size_t)H * W * 4, 255);
    for (int c = 0; c < std::min(C, 4); ++c) {
        for (int i = 0; i < H * W; ++i) {
            float v = (p[(size_t)c * H * W + i] * 0.5f + 0.5f) * 255.0f;
            rgba[(size_t)i * 4 + c] = (uint8_t)std::min(255.0f, std::max(0.0f, std::round(v)));
        }
    }
    return writePngRGBA(path, rgba.data(), W, H);
}

// ---------------------------------------------------------------------------------------------- errors / memory

int QwenImage21Diffusion::availableMemoryMB() {
#if defined(__ANDROID__) || defined(__linux__)
    std::ifstream f("/proc/meminfo");
    std::string key;
    long value = 0;
    std::string unit;
    while (f >> key >> value >> unit) {
        if (key == "MemAvailable:") return (int)(value / 1024);
    }
#endif
    return -1;
}

void QwenImage21Diffusion::setImageSize(int width, int height) {
    mImageWidth = std::max(256, (width / 32) * 32);
    mImageHeight = std::max(256, (height / 32) * 32);
}

bool QwenImage21Diffusion::failStage(const char* stage) {
    int avail = availableMemoryMB();
    bool oom = avail >= 0 && avail < 1500;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s failed%s (available memory %d MB)", stage, oom ? ": out of memory" : "", avail);
    return fail(oom ? kOutOfMemory : kRuntimeError, buf);
}

bool QwenImage21Diffusion::fail(int code, const std::string& message) {
    mErrorCode = code;
    mError = message;
    MNN_ERROR("[QwenImage21] %s\n", message.c_str());
    releaseAll();
    return false;
}

void QwenImage21Diffusion::releaseAll() {
    mTextEncoder.reset();
    mTxtIn.reset();
    mImgIn.reset();
    mDitStep.reset();
    mVae.reset();
    ExecutorScope::Current()->gc(Executor::FULL);
}

namespace {
// Rough peak RAM per stage (MB), measured on Snapdragon 8 Gen 2 / Apple M-series.
int teNeedMB(bool vision) { return vision ? 6100 : 5600; }
int ditNeedMB(int prefixLen, int tokens) { return 5000 + (prefixLen + tokens) / 2; }
// Runtime memory reported on an 8 Gen 2: 320x320 1985 MB, 512x288 2646 MB, 448x576 4266 MB -> ~500 MB fixed
// (fp16 weights) plus ~3900 MB per 512x512 of pixels.
int vaeDecodeNeedMB(int w, int h) { return 500 + (int)(3900.0 * w * h / 262144.0); }
int vaeEncodeNeedMB(int w, int h) { return (int)(1200.0 * w * h / 262144.0); }
} // namespace

bool QwenImage21Diffusion::ensureMemory(const char* stage, int needMB) {
    const int marginMB = 400;
    int avail = availableMemoryMB();
    MNN_PRINT("[QwenImage21] %s: needs ~%d MB, available %d MB\n", stage, needMB, avail);
    if (avail >= 0 && avail < needMB + marginMB) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Out of memory before %s: needs about %d MB, only %d MB available", stage, needMB,
                 avail);
        return fail(kOutOfMemory, buf);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- edit

void QwenImage21Diffusion::editSize(int srcW, int srcH, int& w, int& h) const {
    // diffusers calculate_dimensions(output_resolution^2, ratio): keep the area, round each side to 32.
    double area = (double)mImageWidth * mImageHeight;
    double ratio = (double)srcW / srcH;
    double fw = std::sqrt(area * ratio);
    w = std::max(256, (int)std::lround(fw / 32.0) * 32);
    h = std::max(256, (int)std::lround(fw / ratio / 32.0) * 32);
}

VARP QwenImage21Diffusion::encodeEditPrompt(const std::string& prompt, VARP bgr, int w, int h,
                                            std::vector<char>& isPad) {
    AUTOTIME;
    // Qwen3-VL with the vision tower: the condition image is read as vision context.
    std::string cfg = mModelPath + "/text_encoder/te_vl_config.json";
    std::shared_ptr<Transformer::Llm> te(Transformer::Llm::createLLM(cfg), Transformer::Llm::destroy);
    if (!te) {
        MNN_ERROR("[QwenImage21] cannot create text encoder from %s\n", cfg.c_str());
        return nullptr;
    }
    std::string backend = (mTextEncoderOnCPU || mBackendType == MNN_FORWARD_CPU) ? "cpu" : "opencl";
    te->set_config("{\"backend_type\":\"" + backend + "\",\"thread_num\":" +
                   std::to_string(backend == "cpu" ? mNumThreads : 68) + "}");
    if (!te->load()) {
        MNN_ERROR("[QwenImage21] text encoder (vision) load failed; is text_encoder/visual.mnn present?\n");
        return nullptr;
    }
    Transformer::MultimodalPrompt mp;
    // QwenImage21Pipeline.prompt_template_ti2i; <img>..</img> expands to <|vision_start|><|image_pad|>*n<|vision_end|>
    mp.prompt_template = std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n<|im_start|>user\n" +
                         "<image1><img>cond</img>" + (prompt.empty() ? std::string(" ") : prompt) +
                         "<|im_end|>\n<|im_start|>assistant\n";
    mp.images["cond"] = Transformer::PromptImagePart{bgr, w, h};
    auto ids = te->tokenizer_encode(mp);
    if ((int)ids.size() <= mDropIdx || te->forward(ids) == nullptr) {
        MNN_ERROR("[QwenImage21] text encoder (vision) forward failed\n");
        return nullptr;
    }
    VARP result;
    {
        auto outputs = te->getOutputs();
        int index = te->getOutputIndex(mTeOutputName);
        if (index < 0 || index >= (int)outputs.size()) {
            MNN_ERROR("[QwenImage21] text encoder has no output %s\n", mTeOutputName.c_str());
            return nullptr;
        }
        auto hidden = _Convert(outputs[index], NCHW);
        int T = hidden->getInfo()->dim[1];
        int L = T - mDropIdx;
        std::vector<float> host((size_t)L * kDim);
        ::memcpy(host.data(), hidden->readMap<float>() + (size_t)mDropIdx * kDim, host.size() * sizeof(float));
        isPad.resize(L);
        for (int i = 0; i < L; ++i) isPad[i] = ids[mDropIdx + i] == kImagePadId;
        hidden = nullptr;
        outputs.clear();
        te->reset();
        ExecutorScope scope(Executor::getGlobalExecutor());
        result = _Input({1, L, kDim}, NCHW, halide_type_of<float>());
        ::memcpy(result->writeMap<float>(), host.data(), host.size() * sizeof(float));
        result.fix(VARP::CONSTANT);
    }
    te.reset();
    return result;
}

VARP QwenImage21Diffusion::encodeImage(VARP rgb, int w, int h) {
    AUTOTIME;
    auto rt = runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_;
    auto enc = loadModule("vae_encoder.mnn", {"image"}, {"latent"}, rt);
    if (!enc) return nullptr;
    // RGB uint8 HWC -> RGBA float NCHW in [-1, 1] (opaque alpha)
    const uint8_t* src = rgb->readMap<uint8_t>();
    std::vector<float> rgba((size_t)4 * h * w);
    for (int i = 0; i < h * w; ++i) {
        for (int c = 0; c < 3; ++c) rgba[(size_t)c * h * w + i] = src[i * 3 + c] / 127.5f - 1.0f;
        rgba[(size_t)3 * h * w + i] = 1.0f;
    }
    dump("edit_rgba", rgba.data(), rgba.size());
    int64_t st = nowUs();
    auto out = enc->onForward({hostTensor({1, 4, h, w}, rgba.data())});
    if (out.empty()) return nullptr;
    auto lat = _Convert(out[0], NCHW);  // [1, 64, h/16, w/16]
    int hc = h / 16, wc = w / 16, N = hc * wc;
    const float* lp = lat->readMap<float>();
    std::vector<float> packed((size_t)N * kLatentC);
    for (int c = 0; c < kLatentC; ++c) {
        for (int n = 0; n < N; ++n) packed[(size_t)n * kLatentC + c] = lp[(size_t)c * N + n];
    }
    MNN_PRINT("[QwenImage21] vae encode %dx%d: %.2f s\n", w, h, (nowUs() - st) / 1e6);
    out.clear();
    lat = nullptr;
    enc.reset();
    return hostTensor({1, N, kLatentC}, packed.data());
}

bool QwenImage21Diffusion::runEdit(const std::string& prompt, const std::string& inputImagePath,
                                   const std::string& outputPath, int steps, int seed,
                                   std::function<void(int)> progressCallback) {
    AUTOTIME;
    int64_t st = nowUs();
    auto src = imread(inputImagePath);
    if (src.get() == nullptr || src->getInfo() == nullptr) {
        return fail(kModelError, "cannot read input image " + inputImagePath);
    }
    int srcH = src->getInfo()->dim[0], srcW = src->getInfo()->dim[1];
    int w, h;
    editSize(srcW, srcH, w, h);
    mLatentW = w / 16;
    mLatentH = h / 16;
    // imread returns BGR (a pure red PNG reads back as 0,0,255). Omni's vision path takes BGR and converts to RGB
    // itself; the VAE encoder wants RGB. Getting this backwards tints skin teal in the output, because the model
    // copies the colours it sees in the condition image.
    auto bgr = resize(src, {w, h}, 0, 0, INTER_CUBIC);
    bgr.fix(VARP::CONSTANT);
    auto rgb = cvtColor(bgr, COLOR_BGR2RGB);
    rgb.fix(VARP::CONSTANT);
    MNN_PRINT("[QwenImage21] edit: input %dx%d -> %dx%d\n", srcW, srcH, w, h);

    std::vector<char> isPad;
    if (!ensureMemory("text encoder", teNeedMB(true))) return false;
    auto text = encodeEditPrompt(prompt, bgr, w, h, isPad);
    if (text.get() == nullptr) return failStage("text encoder (vision)");
    if (progressCallback) progressCallback(3);
    if (!ensureMemory("VAE encoder", vaeEncodeNeedMB(w, h))) return false;
    auto cond = encodeImage(rgb, w, h);
    if (cond.get() == nullptr) return failStage("VAE encoder");
    if (progressCallback) progressCallback(5);

    // Layout [t1 text][condition latents][t2 text] + target, see export/qwen_image21_mnn.py:edit_layout
    const int hc = mLatentH, wc = mLatentW, nc = hc * wc;
    int L = (int)isPad.size();
    int first = -1, nslots = 0;
    for (int i = 0; i < L; ++i) {
        if (isPad[i]) {
            if (first < 0) first = i;
            ++nslots;
        }
    }
    if (first < 0 || nslots * 4 != nc) {
        char buf[160];
        snprintf(buf, sizeof(buf), "vision slots %d do not match %dx%d latent tokens", nslots, hc, wc);
        return fail(kRuntimeError, buf);
    }
    const int t1 = first, t2 = L - first - nslots, P = t1 + nc + t2;
    {
        dump("edit_text_hidden", text->readMap<float>(), (size_t)L * kDim);
        std::vector<float> padf(isPad.begin(), isPad.end());
        dump("edit_is_pad", padf.data(), padf.size());
        dump("edit_cond", cond->readMap<float>(), (size_t)nc * kLatentC);
    }

    // text rows without the image slots -> txt_in; condition latents -> img_in
    std::vector<float> textRows((size_t)(t1 + t2) * kDim);
    const float* tp = text->readMap<float>();
    ::memcpy(textRows.data(), tp, (size_t)t1 * kDim * sizeof(float));
    ::memcpy(textRows.data() + (size_t)t1 * kDim, tp + (size_t)(first + nslots) * kDim, (size_t)t2 * kDim * sizeof(float));
    if (!ensureMemory("DiT", ditNeedMB(P, mLatentH * mLatentW))) return false;
    auto txtH = embedText(hostTensor({1, t1 + t2, kDim}, textRows.data()));
    if (txtH.get() == nullptr) return failStage("text projection");
    if (!mImgIn) mImgIn = loadModule("img_in.mnn", {"lat"}, {"img_h"}, runtime_manager_);
    std::vector<float> condHost(cond->readMap<float>(), cond->readMap<float>() + (size_t)nc * kLatentC);
    auto condIn = _Input({1, nc, kLatentC}, NCHW, halide_type_of<float>());
    ::memcpy(condIn->writeMap<float>(), condHost.data(), condHost.size() * sizeof(float));
    auto condOut = mImgIn->onForward({condIn});
    if (condOut.empty()) return failStage("image projection");
    auto condH = hostCopy(condOut[0]);
    condOut.clear();
    std::vector<float> prefix((size_t)P * kDim);
    const float* th = txtH->readMap<float>();
    ::memcpy(prefix.data(), th, (size_t)t1 * kDim * sizeof(float));
    ::memcpy(prefix.data() + (size_t)t1 * kDim, condH->readMap<float>(), (size_t)nc * kDim * sizeof(float));
    ::memcpy(prefix.data() + (size_t)(t1 + nc) * kDim, th + (size_t)t1 * kDim, (size_t)t2 * kDim * sizeof(float));

    std::vector<int> f, hp, wp;
    appendText(0, t1, f, hp, wp);
    appendGrid(t1, hc, wc, f, hp, wp);
    int start2 = t1 + std::max(hc, wc);
    appendText(start2, t2, f, hp, wp);
    appendGrid(start2 + t2, mLatentH, mLatentW, f, hp, wp);
    std::vector<float> cosTab, sinTab;
    ropeFromPositions(f, hp, wp, cosTab, sinTab);
    // block-causal: causal everywhere, bidirectional inside the condition image; key 0 is the masked dummy
    std::vector<float> mask((size_t)P * (P + 1), kMaskNeg);
    for (int q = 0; q < P; ++q) {
        bool qImg = q >= t1 && q < t1 + nc;
        for (int k = 0; k < P; ++k) {
            bool kImg = k >= t1 && k < t1 + nc;
            if (k <= q || (qImg && kImg)) mask[(size_t)q * (P + 1) + 1 + k] = 0.0f;
        }
    }
    std::vector<float> cosP(cosTab.begin(), cosTab.begin() + (size_t)P * 64);
    std::vector<float> sinP(sinTab.begin(), sinTab.begin() + (size_t)P * 64);
    dump("edit_cond_h", condH->readMap<float>(), (size_t)nc * kDim);
    dump("edit_prefix", prefix.data(), prefix.size());
    auto kv = runPrefix(hostTensor({1, P, kDim}, prefix.data()), cosP, sinP, mask);
    if (kv.empty()) return failStage("DiT prefix");
    if (progressCallback) progressCallback(10);
    auto latents = denoise(kv, P, cosTab.data() + (size_t)P * 64, sinTab.data() + (size_t)P * 64, steps, seed,
                           progressCallback);
    kv.clear();
    if (latents.get() == nullptr) return failStage("DiT denoising");
    if (!ensureMemory("VAE decoder", vaeDecodeNeedMB(w, h))) return false;
    auto image = decode(latents);
    if (image.get() == nullptr) return failStage("VAE decoder");
    bool ok = saveRGBA(image, outputPath);
    if (!ok) fail(kRuntimeError, "cannot write " + outputPath);
    MNN_PRINT("[QwenImage21] edit %s %s (seed %d, total %.1f s)\n", ok ? "saved" : "FAILED to save",
              outputPath.c_str(), seed, (nowUs() - st) / 1e6);
    if (progressCallback) progressCallback(100);
    return ok;
}

// ---------------------------------------------------------------------------------------------- run

bool QwenImage21Diffusion::run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed,
                               float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath) {
    AUTOTIME;
    mErrorCode = kOk;
    mError.clear();
    try {
        if (iterNum < 1) iterNum = 20;
        if (iterNum > 100) iterNum = 100;
        int seed = randomSeed < 0 ? (int)(nowUs() & 0x7fffffff) : randomSeed;
        if (!inputImagePath.empty()) {
            return runEdit(prompt, inputImagePath, outputPath, iterNum, seed, progressCallback);
        }
        mLatentW = mImageWidth / 16;
        mLatentH = mImageHeight / 16;
        const int N = mLatentH * mLatentW;
        int64_t st = nowUs();
        if (!ensureMemory("text encoder", teNeedMB(false))) return false;
        auto text = encodePrompt(prompt);
        if (text.get() == nullptr) return failStage("text encoder");
        int L = text->getInfo()->dim[1];
        MNN_PRINT("[QwenImage21] text encoder done: %.2f s\n", (nowUs() - st) / 1e6);
        if (progressCallback) progressCallback(5);
        if (!ensureMemory("DiT", ditNeedMB(L, N))) return false;
        auto kv = buildPrefixCache(text);
        if (kv.empty()) return failStage("DiT prefix");
        if (progressCallback) progressCallback(10);
        std::vector<float> cosTab, sinTab;
        ropeTables(L, mLatentH, mLatentW, cosTab, sinTab);
        auto latents = denoise(kv, L, cosTab.data() + (size_t)L * 64, sinTab.data() + (size_t)L * 64, iterNum, seed,
                               progressCallback);
        kv.clear();
        if (latents.get() == nullptr) return failStage("DiT denoising");
        if (!ensureMemory("VAE decoder", vaeDecodeNeedMB(mImageWidth, mImageHeight))) return false;
        auto image = decode(latents);
        if (image.get() == nullptr) return failStage("VAE decoder");
        bool ok = saveRGBA(image, outputPath);
        if (!ok) fail(kRuntimeError, "cannot write " + outputPath);
        MNN_PRINT("[QwenImage21] %s %s (seed %d, total %.1f s)\n", ok ? "saved" : "FAILED to save",
                  outputPath.c_str(), seed, (nowUs() - st) / 1e6);
        if (progressCallback) progressCallback(100);
        return ok;
    } catch (const std::bad_alloc&) {
        return fail(kOutOfMemory, "Out of memory (allocation failed)");
    } catch (const std::exception& e) {
        return fail(kRuntimeError, std::string("exception: ") + e.what());
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
