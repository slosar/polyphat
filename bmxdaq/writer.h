#pragma once
#include "settings.h"
#include "stdio.h"
#include "time.h"
#define MAXFNLEN 512

struct BMXHEADER {
  int nChannels;
  int nPSbins; // 
};

struct WRITER {
  char fname[MAXFNLEN];
  int pslen; // full length of PS info
  int save_every; // how many minutes we save.
  FILE* f;
  bool reopen;
  BMXHEADER header;
};


void writerInit(WRITER *writer, SETTINGS *set);
void writerWritePS (WRITER *writer, float* ps);
void writerCleanUp(WRITER *writer);

