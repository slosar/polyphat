/*

The digitizer GPU driver.

Based on examples by Spectrum GmbH.

Anze Slosar, anze@bnl.gob

**************************************************************************
*/

#include "settings.h"
#include "digicard.h"
#include "gpucard.h"

// ----- standard c include files -----
#include <stdio.h>
#include <string.h>
#include <stdlib.h>




/*
**************************************************************************
main 
**************************************************************************
*/

int main(int argc,char **argv)
{ 
  char                szBuffer[1024];     // a character buffer for any messages
  SETTINGS settings;                      // settings
  DIGICARD dcard;                         // digitizer CARD
  GPUCARD gcard;                          // GPU card
  
  if(argc>=2) {
    char fname_ini[256];
    sprintf(fname_ini,"%s",argv[1]);
    init_settings(&settings,fname_ini);
  } else
    init_settings(&settings,NULL);

  // intialize
  print_settings(&settings);
  digiCardInit(&dcard,&settings);
  gpuCardInit(&gcard,&settings);
  //work
  digiWorkLoop(&dcard, &gcard, &settings);
  //shutdown
  digiCardCleanUp(&dcard, &settings);

  return 0;
}

