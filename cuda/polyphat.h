
#ifndef POLYPHASE_HEADER
#define POLYPHASE_HEADER

void getFreqArray (float dt, float* arr);
void polyphase_charge();
void polyphase_release();
void polyPhase_cuda(float *sam, float *spec);

#endif



