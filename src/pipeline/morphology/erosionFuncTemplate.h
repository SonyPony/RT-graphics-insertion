#include <stdint.h>

//void Filter(int * src, int * dst, int * temp, int width, int height, int radio);
void FilterDilation(uint8_t * src, uint8_t * temp, int width, int height, int radio);