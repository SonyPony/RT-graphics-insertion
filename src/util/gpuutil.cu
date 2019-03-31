#include "gpuutil.cuh"
#include <stdlib.h>

void Gpu::Utils::generateRandStates(Gpu::Utils::RandState ** d_rs, size_t count)
{
    using namespace Gpu::Utils;

    auto h_randStates = new RandState[count];
    for (size_t i = 0; i < count; i++)
        h_randStates[i] = static_cast<uint32_t>(rand());

    // copy states to device
    cudaMalloc(reinterpret_cast<void**>(d_rs), count * sizeof(RandState));
    cudaMemcpy(*d_rs, h_randStates, count * sizeof(RandState), cudaMemcpyHostToDevice);
}