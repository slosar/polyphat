#ifndef SETTINGS_H
#define SETTINGS_H

/* Basic parameters of ADC */
#define SAMPLING_RATE 2.7e9
#define DELTA_T (1.0/SAMPLING_RATE)

/* SIZE OF FFT, 
we aim at 1ms = 2.7MSamples, 
closest is 2**21, corresponding to 0.8 ms */
#define FFT_SIZE 2097152

/* Size of a chunk we process in one go
At 2.7 MS, this 99.4 ms*/
#define NUM_FFT 128
#define BUFFER_SIZE (NUM_FFT*FFT_SIZE)

#endif

