#ifndef GENERATOR_H
#define GENERATOR_H
#include "stdint.h"

/* Generates white noise 

Incredibly slow, presumably because default rand() is very slow.
Fix at some point...

*/

/* Generates white noise into buffer with amplitude
   given in unit_8 units (i.e. <128) */
void generate_white (uint8_t *buffer, float amplitude, unsigned int seed);

/* Generates a tone  with amplitude
   given in unit_8 units (i.e. <128) 
   and frequency in MHz and phase offset in radians.
   Note: Always adds a bit unit of noise to prevent
   discrete sampling errors coming in.
 */
void generate_tone (uint8_t *buffer, float amplitude, float freq, float phi, unsigned int seed);


/* write to a binary file */
void write_bin(uint8_t *buffer, char* fname);
/* read from a binary file */
void read_bin(uint8_t *buffer, char* fname);


#endif
