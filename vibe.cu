#include "vibe.cuh"
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

using RandState = uint32_t;

__global__ void segment_new(uint8_t* input, uint8_t* model, uint8_t* dest) {
	constexpr int stride = 4;	// 4 bytes strides
	const int i = (blockDim.x * blockIdx.x + threadIdx.x);
	const int pixel_i = i * stride;

	int count = 0;
	for (int j = 0; j < 20; j++) {
		const int sample_i = i * (20 * 3) + j * 3;
		float distance =
			norm3df(
				(float)input[pixel_i] - model[sample_i],
				(float)input[pixel_i + 1] - model[sample_i + 1],
				(float)input[pixel_i + 2] - model[sample_i + 2]);
		if (distance < 20)
			count++;
	}

	dest[i] = 0;
	if (count < 2)
		dest[i] = 255;

}

__device__ __forceinline__ float devRand(RandState & state) {
	constexpr RandState RNG_COEFF = 4164903690U;
	state = state * RNG_COEFF + (state >> 16);
	return static_cast<float>(state) / UINT32_MAX;
}

__global__ void segment_fast(uint8_t* input, uint8_t* model, uint8_t* dest, int size, int currentSample, RandState* randState) {
	constexpr int stride = 4;	// 4 bytes strides
	const int id = (blockDim.x * blockIdx.x + threadIdx.x);
	const int pixel_i = id * stride;

	int count = 0;
	for (int j = 0; j < 20; j++) {
		const int sample_i = size * j * 3;
		float distance = norm3df(
			(float)input[pixel_i] - model[id + sample_i],
			(float)input[pixel_i + 1] - model[id + sample_i + size],
			(float)input[pixel_i + 2] - model[id + sample_i + size * 2]
		);

		if (distance < 20)
			count++;
	}

	uint8_t result = 0;
	const int isBackground = (count >= 2);
	if (count < 2)
		result = 255;
	dest[id] = result;

	// update
	RandState localRandState = randState[id];
	float rand = devRand(localRandState) * 16;
	if (rand == 0) {
		for (int i = 0; i < 3; i++)
			model[id + currentSample + i * size] = input[pixel_i + i];
	}

	rand = devRand(localRandState) * 16;
	// todo change to something like stencil?
	if (rand == 0) {
		for (int i = 0; i < 3; i++)
			model[id + currentSample + i * size] = input[pixel_i + i];
	}

	randState[id] = localRandState;
}

__global__ void initBackgroundModelSamples(uint8_t* input, uint8_t* dest, uint8_t samplesCount) {
	constexpr int stride = 4;	// 4 bytes strides
	const int rgba_i = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
	const int rgb_i = (blockDim.x * blockIdx.x + threadIdx.x) * (stride - 1) * samplesCount;

	for (int j = 0; j < samplesCount; j++) {
		dest[j * 3 + rgb_i] = input[rgba_i + 0];
		dest[j * 3 + rgb_i + 1] = input[rgba_i + 1];
		dest[j * 3 + rgb_i + 2] = input[rgba_i + 2];
	}
}

__global__ void initBackgroundModelSamples_new(uint8_t* input, uint8_t* dest, uint8_t samplesCount, int size) {
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

	cudaMalloc(reinterpret_cast<void**>(&m_d_model), modelSize);
	cudaMalloc(reinterpret_cast<void**>(&d_input), sampleSize);
	cudaMalloc(reinterpret_cast<void**>(&m_d_bgModel), modelSize);
	cudaMemcpy(d_input, backgroundModel, sampleSize, cudaMemcpyHostToDevice);

	initBackgroundModelSamples <<<900, 1024 >>> (d_input, m_d_model, sampleCount);
	initBackgroundModelSamples_new<<<900, 1024 >>> (d_input, m_d_bgModel, sampleCount, m_width * m_height);

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

	segment_new<<<900, 1024>>> (d_input, m_d_model, d_dest);
	segment_fast<<<900, 1024 >>> (d_input, m_d_bgModel, d_dest, size, 0, d_randState);
	auto err = cudaGetLastError();
	std::cout << cudaGetErrorName(err);

	cudaMemcpy(dest, d_dest, outputDataSize, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_dest);
}