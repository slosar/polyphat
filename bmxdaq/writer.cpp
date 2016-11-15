#include "writer.h"
#include "string.h"

void writerInit(WRITER *writer, SETTINGS *s) {
  printf ("\n\nInitializing digitizer\n");
  printf ("==========================\n");
  strcpy(writer->fname,"output.bin");
  writer->pslen=s->fft_size/2/s->fft_avg*(1+3*(s->channel_mask==3));
  printf ("Writing to: %s\n", writer->fname);
  printf ("Record size: %i\n", writer->pslen);
  BMXheader h;
  h.nChannels=1+s->channel_mask;
  h.nPSbins=s->fft_size/2/s->fft_avg;
  writer->f=fopen(writer->fname,"wb");
  fwrite (&h, sizeof(BMXheader),1,writer->f);
}
void writerWritePS (WRITER *writer, float* ps) {
  fwrite (ps, sizeof(float), writer->pslen, writer->f);
  fflush(writer->f);
}

void writerCleanUp(WRITER *writer) {
  fclose(writer->f);
}
