#include "vibe.cuh"
#include <iostream>

__global__ void addKernel(int *c, const int *a, const int *b)
{
	
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void gpu::add(int* a, int* b, int* dest, int arrSize) {
	int* d_a = nullptr;
	int* d_b = nullptr;
	int* d_result = nullptr;
	const size_t bytesArrSize = arrSize * sizeof(int);

	cudaError_t cudaStatus = cudaSetDevice(0);

	// alloc device mem
	cudaMalloc(reinterpret_cast<void**>(&d_a), bytesArrSize);
	cudaMalloc(reinterpret_cast<void**>(&d_b), bytesArrSize);
	cudaMalloc(reinterpret_cast<void**>(&d_result), bytesArrSize);

	// copy values
	cudaMemcpy(d_a, a, bytesArrSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, bytesArrSize, cudaMemcpyHostToDevice);

	addKernel <<<1, arrSize>>> (d_result, d_a, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(dest, d_result, bytesArrSize, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
}

// ARGB
__global__ void segment(uint8_t* input, /*uint8_t* model, */uint8_t* dest) {
	constexpr int stride = 4;	// 4 bytes strides
	const int i = (blockDim.x * blockIdx.x + threadIdx.x) * stride;

	dest[i] = input[i];		// A
	dest[i + 1] = 0;//input[i + 1];		// R
	dest[i + 2] = input[i + 2];		// G
	dest[i + 3] = input[i + 3];		// B
	//printf("%d\n", i);
}

void GPU::InsertionGraphicsPipeline::process(uint8_t * input, uint8_t * graphics, uint8_t * dest) {
	using T = decltype(input);
	T d_input = nullptr;
	T d_dest = nullptr;
	const size_t dataSize = 4 * m_height * m_width;

	cudaSetDevice(0);
	cudaMalloc(reinterpret_cast<void**>(&d_input), dataSize);
	cudaMalloc(reinterpret_cast<void**>(&d_dest), dataSize);

	cudaMemcpy(d_input, input, dataSize, cudaMemcpyHostToDevice);
	dim3 blocks(m_width / 16, m_height / 16);
	dim3 threads(16, 16);

	segment <<<1280, 720 >>> (d_input, d_dest);

	std::cout << "ahoj" << std::endl;

	cudaMemcpy(dest, d_dest, dataSize, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_dest);
}