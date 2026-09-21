//
//  qwen_image21_demo.cpp
//  Command-line Qwen-Image-2.1 text-to-image.
//
#include <cstdlib>
#include <iostream>
#include <string>
#include "diffusion/qwen_image21_diffusion.hpp"

using namespace MNN::DIFFUSION;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: %s <model_dir> <output.png> <prompt> [steps=20] [seed=42] [backend=opencl|cpu] "
                  "[memory_mode=0] [te_on_cpu=1] [size=512] [precision=low|normal|high] [threads=4] [vae_on_cpu=0]\n", argv[0]);
        return 1;
    }
    std::string modelDir = argv[1];
    std::string output = argv[2];
    std::string prompt = argv[3];
    int steps = argc > 4 ? atoi(argv[4]) : 20;
    int seed = argc > 5 ? atoi(argv[5]) : 42;
    std::string backend = argc > 6 ? argv[6] : "opencl";
    int memoryMode = argc > 7 ? atoi(argv[7]) : 0;
    bool teOnCpu = argc > 8 ? atoi(argv[8]) != 0 : true;
    int size = argc > 9 ? atoi(argv[9]) : 512;
    std::string precision = argc > 10 ? argv[10] : "low";
    int threads = argc > 11 ? atoi(argv[11]) : 4;
    bool vaeOnCpu = argc > 12 ? atoi(argv[12]) != 0 : false;

    auto type = backend == "cpu" ? MNN_FORWARD_CPU : MNN_FORWARD_OPENCL;
    auto prec = precision == "high" ? PRECISION_HIGH : (precision == "normal" ? PRECISION_NORMAL : PRECISION_LOW);
    std::unique_ptr<Diffusion> d(Diffusion::createDiffusion(modelDir, QWEN_IMAGE_21, type, memoryMode, size, size,
                                                            teOnCpu, vaeOnCpu, GPU_MEMORY_BUFFER, prec, CFG_MODE_AUTO,
                                                            threads));
    if (!d->load()) {
        MNN_ERROR("load failed\n");
        return 1;
    }
    bool ok = d->run(prompt, output, steps, seed, [](int p) { MNN_PRINT("progress %d%%\n", p); });
    return ok ? 0 : 1;
}
