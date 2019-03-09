#ifndef VIBE
#define VIBE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>


namespace GPU {
	class InsertionGraphicsPipeline
	{
		private:
			static constexpr uint8_t SAMPLE_COUNT = 20;
			static constexpr uint8_t PIXEL_DATA_SIZE = 4;	 // in bytes

			uint8_t* m_d_segmentation;
			uint8_t* m_d_model;
			uint8_t* m_d_bgModel;
			uint8_t* m_d_input;

			const int m_width;
			const int m_height;

		public:
			InsertionGraphicsPipeline(int width, int height):
				m_width{width}, m_height(height) {}

			uint8_t* initialize(uint8_t* backgroundModel);
			void process(uint8_t* input, uint8_t* graphics, uint8_t* dest);
	};
	
}

#endif // !VIBE
