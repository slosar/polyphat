#ifndef SETTINGS_H
#define SETTINGS_H
#include <math.h>

/*version*/
#define VERSION "0.1 PROTO"
/*#define DEBUGREDUCE*/

/* Basic parameters of ADC */
#define SAMPLING_RATE 2.7e9
#define DELTA_T (1.0/SAMPLING_RATE)

/* SIZE OF FFT, BUFFERS 
we aim at 1ms = 2.7MSamples, 
2**21 is 0.8 ms 
2**23 is 0.99ms
*/
#define BUFFER_SIZE (268435456)
#define NUM_FFT 1
#define FFT_SIZE (BUFFER_SIZE/NUM_FFT)
#define TRANSFORM_SIZE (FFT_SIZE/2+1)

/* Size of a chunk we process in one go
At 2.7 MS, this 99.4 ms*/

#define DELTA_NU (1./(DELTA_T*FFT_SIZE))

/* numin and numax in freq in MHZ */
#define NUMIN 600.0
#define NUMAX 1300.0
/* We average over this many FFT bins */
#define NUAVG (16384/NUM_FFT)
/* and we get this many bins */

void print_settings();
inline int num_nubins() { return (floor((NUMAX-NUMIN)*1e6/DELTA_NU/NUAVG)+1); }

#endif

