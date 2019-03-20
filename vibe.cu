#include "vibe.cuh"
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

using RandState = uint32_t;

__device__ __forceinline__ float devRand(RandState & state) {
	constexpr RandState RNG_COEFF = 4164903690U;
	state = state * RNG_COEFF + (state >> 16);
	return static_cast<float>(state) / UINT32_MAX;
}

__global__ void segment(uint8_t* input, uint8_t* model, uint8_t* dest, int size, int currentSample, RandState* randState) {
	constexpr int stride = 4;	// 4 bytes strides
	const int id = (blockDim.x * blockIdx.x + threadIdx.x);
	const int pixel_i = id * stride;

	const float localInput[3] = { (float)input[pixel_i] , (float)input[pixel_i + 1] , (float)input[pixel_i + 2] };

	int count = 0;
	for (int j = 0; j < 20; j++) {
		const int sample_i = size * j * 3;
		float distance = norm3df(
			localInput[0] - model[id + sample_i],
			localInput[1] - model[id + sample_i + size],
			localInput[2] - model[id + sample_i + size * 2]
		);

		if (distance < 20)
			count++;
	}

	const int isForeground = (count < 2);
	dest[id] = isForeground * 255;

	// update
	RandState localRandState = randState[id];
	uint8_t rand = devRand(localRandState) * 16;
	if (rand == 0) {
		for (int i = 0; i < 3; i++)
			model[id + currentSample + i * size] = localInput[i];
	}

	rand = devRand(localRandState) * 16;
	// todo change to something like stencil?
	if (rand == 0) {
		for (int i = 0; i < 3; i++)
			model[id + currentSample + i * size] = localInput[i];
	}

	randState[id] = localRandState;
}

__global__ void initBackgroundModelSamples(uint8_t* input, uint8_t* dest, uint8_t samplesCount, int size) {
	constexpr int stride = 4;	// 4 bytes strides
	const int rgba_i = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
	const int i = (blockDim.x * blockIdx.x + threadIdx.x);
	const int pixelffset = (i % size) * 3;

	for (int j = 0; j < samplesCount; j++) {
		for (int channel = 0; channel < 3; channel++)
			dest[j * size * 3 + i + channel * size] = input[rgba_i + channel];
	}
}

uint8_t* GPU::InsertionGraphicsPipeline::initialize(uint8_t * backgroundModel) {
	cudaSetDevice(0);	// TODO try to move to constructor

	const int sampleSize = m_width * m_height * InsertionGraphicsPipeline::PIXEL_DATA_SIZE;
	// does not have alpha channel
	const int modelSampleSize = m_width * m_height * (InsertionGraphicsPipeline::PIXEL_DATA_SIZE - 1);
	constexpr uint8_t sampleCount = InsertionGraphicsPipeline::SAMPLE_COUNT;
	const int modelSize = modelSampleSize * sampleCount;

	uint8_t* d_input = nullptr;

	cudaMalloc(reinterpret_cast<void**>(&d_input), sampleSize);
	cudaMalloc(reinterpret_cast<void**>(&m_d_bgModel), modelSize);
	cudaMemcpy(d_input, backgroundModel, sampleSize, cudaMemcpyHostToDevice);

	initBackgroundModelSamples<<<900, 1024 >>> (d_input, m_d_bgModel, sampleCount, m_width * m_height);

	uint8_t* testOut = new uint8_t[m_width * m_height * 20 * 3];
	cudaMemcpy(testOut, m_d_bgModel, m_width * m_height * 20 * 3, cudaMemcpyDeviceToHost);
	uint8_t* sample = new uint8_t[m_width * m_height];
	int sample_id = 0;
	for (int i = 0; i < m_width * m_height; i++) {
		sample[i] = testOut[i];
	}

	return sample;
}

__global__ void randomy(RandState* states, float* out) {
	const int id = threadIdx.x;

	out[id] = devRand(states[id]);
}

void GPU::InsertionGraphicsPipeline::process(uint8_t * input, uint8_t * graphics, uint8_t * dest) {
	using T = decltype(input);
	T d_input = nullptr;
	T d_dest = nullptr;
	const uint32_t size = m_height * m_width;
	const size_t dataSize = size * 4;
	const size_t outputDataSize = size;
	RandState* d_randState = nullptr;
	RandState* randState = new uint32_t[size];
	
	srand(time(nullptr));
	for (int i = 0; i < size; i++)
		randState[i] = static_cast<uint32_t>(rand());

	cudaSetDevice(0);
	cudaMalloc(reinterpret_cast<void**>(&d_input), dataSize);
	cudaMalloc(reinterpret_cast<void**>(&d_dest), outputDataSize);
	cudaMalloc(reinterpret_cast<void**>(&d_randState), size * sizeof(RandState));

	cudaMemcpy(d_input, input, dataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_randState, randState, size * sizeof(uint32_t), cudaMemcpyHostToDevice);

	segment<<<1800, 512>>> (d_input, m_d_bgModel, d_dest, size, 0, d_randState);
	auto err = cudaGetLastError();
	std::cout << cudaGetErrorName(err);

	cudaMemcpy(dest, d_dest, outputDataSize, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_dest);
}