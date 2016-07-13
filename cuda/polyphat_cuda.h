#ifndef POLYPHAT_CUDA_H
#define POLYPHAT_CUDA_H
void cuda_alloc(float *filter);
void cuda_release();
void cuda_exec(float* sample, float *output);
#endif
