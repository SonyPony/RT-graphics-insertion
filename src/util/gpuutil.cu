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

__global__ void k_sobel_sep_v(uint8_t* input, short2* output) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x == 0 || x == FRAME_WIDTH - 1 || y == 0 || y == FRAME_HEIGHT - 1)
        return;

    const int kernelX[3] = {1, 2, 1};
    const int kernelY[3] = { -1, 0, 1 };
    short2 outputValue = {0, 0};

    for (int i = -1; i <= 1; i++) {
        const int inputValue = input[x + (y + i) * FRAME_WIDTH];

        outputValue.x += kernelX[i + 1] * inputValue;
        outputValue.y += kernelY[i + 1] * inputValue;
    }

    output[x + y * FRAME_WIDTH] = outputValue;
}

__global__ void k_sobel_sep_h(short2* input, short2* output) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x == 0 || x == FRAME_WIDTH - 1 || y == 0 || y == FRAME_HEIGHT - 1)
        return;

    const int kernelX[3] = { 1, 0, -1 };
    const int kernelY[3] = { 1, 2, 1 };
    short2 outputValue = { 0, 0 };

    for (int i = -1; i <= 1; i++) {
        const short2 inputValue = input[x + i + y * FRAME_WIDTH];

        outputValue.x += kernelX[i + 1] * inputValue.x;
        outputValue.y += kernelY[i + 1] * inputValue.y;
    }

    output[x + y * FRAME_WIDTH] = outputValue;
}

__global__ void k_testLab(uchar4* frame, uint8_t* out) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    uchar4 pixel = frame[id];
    uint8_t lab = Gpu::Utils::cvtRGB2GRAY(pixel);

    out[id] = lab;
}

__global__ void k_dualcvtRGBA2RGB(uchar4* in1, uchar4* in2,
    uint8_t* out1, uint8_t* out2) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const uchar4 in1Pixel = in1[id];
    const uchar4 in2Pixel = in2[id];

    out1[id * 3] = in1Pixel.x;
    out1[id * 3 + 1] = in1Pixel.y;
    out1[id * 3 + 2] = in1Pixel.z;

    out2[id * 3] = in2Pixel.x;
    out2[id * 3 + 1] = in2Pixel.y;
    out2[id * 3 + 2] = in2Pixel.z;
}

__global__ void k_cvtRGBA2RGB_a(uchar4* in, uint8_t* out, uint8_t* outAlpha) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const uchar4 inPixel = in[id];

    out[id * 3] = inPixel.x;
    out[id * 3 + 1] = inPixel.y;
    out[id * 3 + 2] = inPixel.z;

    outAlpha[id] = inPixel.w;
}

__global__ void k_mirrorV(uchar4* in, uchar4* out) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    out[x + (FRAME_HEIGHT - y - 1) * FRAME_WIDTH] = in[id];
}

void Gpu::Utils::gradients(dim3 dimGrid, dim3 dimBlock, uint8_t * input, short2 * temp, short2 * dest)
{
    k_sobel_sep_v << <dimGrid, dimBlock >> > (input, temp);
    k_sobel_sep_h << <dimGrid, dimBlock >> > (temp, dest);
}

void Gpu::Utils::dualCvtRGBA2RGB(dim3 dimGrid, dim3 dimBlock, uchar4* d_in1, uchar4* d_in2,
    uint8_t* d_out1, uint8_t* d_out2) {
    k_dualcvtRGBA2RGB << <dimGrid, dimBlock >> > (d_in1, d_in2, d_out1, d_out2);
}

void Gpu::Utils::cvtRGBA2RGB_A(dim3 dimGrid, dim3 dimBlock, uchar4 * d_in, uint8_t * d_out, uint8_t * d_outAlpha)
{
    k_cvtRGBA2RGB_a << <dimGrid, dimBlock >> > (d_in, d_out, d_outAlpha);
}

void Gpu::Utils::mirrorV(dim3 dimGrid, dim3 dimBlock, uchar4* d_in, uchar4* d_out) 
{
    k_mirrorV << <dimGrid, dimBlock >> > (d_in, d_out);
}
