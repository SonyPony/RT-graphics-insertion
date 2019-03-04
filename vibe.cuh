#ifndef VIBE
#define VIBE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>

namespace gpu {
	void add(int* a, int* b, int* dest, int arrSize);
	
}

namespace GPU {
	class InsertionGraphicsPipeline
	{
		private:
			uint8_t* m_d_segmentation;
			uint8_t* m_d_model;
			uint8_t* m_d_input;

			const int m_width;
			const int m_height;

		public:
			InsertionGraphicsPipeline(int width, int height):
				m_width{width}, m_height(height) {}

			void process(uint8_t* input, uint8_t* graphics, uint8_t* dest);
	};
	
}

#endif // !VIBE
