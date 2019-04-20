void NaiveErosion(int * src, int * dst, int width, int height, int radio);
void ErosionTwoSteps(int * src, int * dst, int * temp, int width, int height, int radio);
void ErosionTwoStepsShared(int * src, int * dst, int * temp, int width, int height, int radio);
void ErosionTemplateSharedTwoSteps(uint8_t * src, uint8_t * temp, int width, int height, int radio);