#pragma once
#include "settings.h"
#include "stdio.h"

#define MAXFNLEN 512
struct WRITER {
  char fname[MAXFNLEN];
  int pslen; // full length of PS info
  FILE* f;
};

struct BMXheader {
  int nChannels;
  int nPSbins; // 
};

void writerInit(WRITER *writer, SETTINGS *set);
void writerWritePS (WRITER *writer, float* ps);
void writerCleanUp(WRITER *writer);

