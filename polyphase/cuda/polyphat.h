
#ifndef POLYPHASE_HEADER
#define POLYPHASE_HEADER

void getFreqArray (float dt, float* arr);
float* polyphase_alloc_buffer(int size);
void polyphase_charge();
void polyphase_release();
void polyPhase_cuda(float *sam, float *spec);

#endif



