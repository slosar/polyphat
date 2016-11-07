#pragma once

#include "stdint.h"

// modifiable settings
struct SETTINGS {
  // basic settings

  // digi card settings
  float sample_rate; // in samples/s
  uint64_t channel_mask;  // channel bit mask 
  int32_t ADC_range; // in mV
  int ext_clock_mode; // 0 for internal, 1 for external
  
  //
  uint32_t fft_size; // must be power of 2
  float nu_min, nu_max; // min and max frequency to output
  uint32_t fft_avg; // how many bins to average over
  int buf_mult; // buffer multiplier, we allocate
                //buf_mult*fft_size for transfer
  //
  int n_cuda_streams; // number of cuda streams
  int cuda_threads; // number of cuda threads
};

// Fixed defines

#define VERSION "0.01"


void init_settings(SETTINGS *settings, char* inifile);
void print_settings(SETTINGS *s);
