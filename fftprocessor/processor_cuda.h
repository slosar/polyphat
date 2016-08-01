#ifndef PROCESSOR_CUDA_H
#define PROCESSOR_CUDA_H
#include "stdint.h"
#include "settings.h"
uint8_t* alloc_sample_buffer();
float* alloc_power();
void cuda_test(uint8_t *buf, float* freq, float* power);
void ztest();

#endif
