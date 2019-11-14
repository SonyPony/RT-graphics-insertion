#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

/*#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720*/
#define TEXTURE_ENHANCE_LEVEL 1.f
/*#define GRAPHICS_WIDTH 1280
#define GRAPHICS_HEIGHT 720

#define DIM_GRID dim3{ 80, 45 }
#define DIM_BLOCK dim3{ 16, 16 }*/

#define FRAME_WIDTH 1920
#define FRAME_HEIGHT 1080
#define FRAME_SIZE (FRAME_WIDTH * FRAME_HEIGHT)

#define GRAPHICS_WIDTH 1280
#define GRAPHICS_HEIGHT 720

#define DIM_GRID dim3{ 120, 135 }
#define DIM_BLOCK dim3{ 16, 8 }

#define DIM_GRID_GRAPHICS dim3{ 80, 90 }
#define DIM_BLOCK_GRAPHICS dim3{ 16, 8 }

class Config {
    public:
        static constexpr uint8_t CHANNELS_COUNT_INPUT = 4;     // in bytes
        // does not have alpha channel
        static constexpr uint8_t CHANNELS_COUNT_BG_SAMPLE = 4;     // in bytes
};

#endif // !CONFIG_H

