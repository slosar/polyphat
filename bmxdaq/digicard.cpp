#include "settings.h"
#include "digicard.h"
#include "gpucard.h"

// ----- include standard driver header from library -----
#include "spcm_examples/c_cpp/common/ostools/spcm_oswrap.h"
#include "spcm_examples/c_cpp/common/ostools/spcm_ostools.h"


// ----- standard c include files -----
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


/*
**************************************************************************
szTypeToName: doing name translation
**************************************************************************
*/

char* szTypeToName (int32 lCardType)
    {
    static char szName[50];
    switch (lCardType & TYP_SERIESMASK)
        {
        case TYP_M2ISERIES:     sprintf (szName, "M2i.%04x", (unsigned int) (lCardType & TYP_VERSIONMASK));      break;
        case TYP_M2IEXPSERIES:  sprintf (szName, "M2i.%04x-Exp", (unsigned int) (lCardType & TYP_VERSIONMASK));  break;
        case TYP_M3ISERIES:     sprintf (szName, "M3i.%04x", (unsigned int) (lCardType & TYP_VERSIONMASK));      break;
        case TYP_M3IEXPSERIES:  sprintf (szName, "M3i.%04x-Exp", (unsigned int) (lCardType & TYP_VERSIONMASK));  break;
        case TYP_M4IEXPSERIES:  sprintf (szName, "M4i.%04x-x8", (unsigned int) (lCardType & TYP_VERSIONMASK));   break;
        case TYP_M4XEXPSERIES:  sprintf (szName, "M4x.%04x-x4", (unsigned int) (lCardType & TYP_VERSIONMASK));   break;
        default:                sprintf (szName, "unknown type");                               break;
        }
    return szName;
    }


/*
**************************************************************************
Setup the digitizer card
**************************************************************************
*/

void digiCardInit (DIGICARD *card, SETTINGS *set) {
    // open card

  printf ("\n\nInitilizing digitizer\n");
  printf ("=====================\n");
  card->hCard = spcm_hOpen ((char*)"/dev/spcm0");
    if (!card->hCard) {
        printf ("no digitizer card found...\n");
        exit(1);
    }

    int32       lCardType, lSerialNumber, lFncType;
    // read type, function and sn and check for A/D card
    spcm_dwGetParam_i32 (card->hCard, SPC_PCITYP,         &lCardType);
    spcm_dwGetParam_i32 (card->hCard, SPC_PCISERIALNO,    &lSerialNumber);
    spcm_dwGetParam_i32 (card->hCard, SPC_FNCTYPE,        &lFncType);

    switch (lFncType)
        {
        case SPCM_TYPE_AI:  
            printf ("Found: %s sn %05d\n", szTypeToName (lCardType), lSerialNumber);
            break;

        default:
            printf ("Card: %s sn %05d not supported. \n", szTypeToName (lCardType), lSerialNumber);            
            exit(1);
        }


    // do a simple standard setup
    // always do two channels
    spcm_dwSetParam_i32 (card->hCard, SPC_CHENABLE,       set->channel_mask);                     // just 1 channel enabled
    spcm_dwSetParam_i32 (card->hCard, SPC_PRETRIGGER,     1024);                  // 1k of pretrigger data at start of FIFO mode
    spcm_dwSetParam_i32 (card->hCard, SPC_CARDMODE,       SPC_REC_FIFO_SINGLE);   // single FIFO mode
    spcm_dwSetParam_i32 (card->hCard, SPC_TIMEOUT,        5000);                 // timeout 5 s
    spcm_dwSetParam_i32 (card->hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);    // trigger set to software
    spcm_dwSetParam_i32 (card->hCard, SPC_TRIG_ANDMASK,   0);                     // ...
    if (set->ext_clock_mode) 
      spcm_dwSetParam_i32 (card->hCard, SPC_CLOCKMODE,      SPC_CM_EXTREFCLOCK);
    else
      spcm_dwSetParam_i32 (card->hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL

    spcm_dwSetParam_i64 (card->hCard, SPC_REFERENCECLOCK, (long long int)(set->sample_rate));
    spcm_dwSetParam_i64 (card->hCard, SPC_SAMPLERATE, (long long int)(set->sample_rate));

    spcm_dwSetParam_i32 (card->hCard, SPC_CLOCKOUT,       0);                     // no clock output

    long long int srate;
    spcm_dwGetParam_i64 (card->hCard, SPC_SAMPLERATE, &srate);
    printf ("Sampling rate set to %.1lf MHz\n", srate/1000000.);
    printf ("Allocating digitizer buffer...\n");
    /// now set the memory
    card->two_channel = (set->channel_mask==3);
    card->lNotifySize = set->fft_buffer_size*(1+card->two_channel);
    card->lBufferSize = card->lNotifySize*set->buf_mult;
    
    /// alocate buffer
    card->pnData = (int16*) pvAllocMemPageAligned ((uint64) card->lBufferSize);
    if (!card->pnData)
        {
        printf ("memory allocation failed\n");
        spcm_vClose (card->hCard);
        exit(1);
        }
    // define transfer
    spcm_dwDefTransfer_i64 (card->hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC,
			    card->lNotifySize, card->pnData, 0, card->lBufferSize);
    printf ("Digitizer card and buffer ready.\n");

}

float deltaT (timespec t1,timespec t2) {
  return ( t2.tv_sec - t1.tv_sec )
	  + ( t2.tv_nsec - t1.tv_nsec )/ 1e9;
}

void  digiWorkLoop(DIGICARD *dc, GPUCARD *gc, SETTINGS *set) {
  // start everything
  char        szErrorTextBuffer[ERRORTEXTLEN];
  uint32      dwError = spcm_dwSetParam_i32 (dc->hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA);
  int32       lStatus, lAvailUser, lPCPos, fill;

  // check for error
  if (dwError != ERR_OK)
    {
      spcm_dwGetErrorInfo_i32 (dc->hCard, NULL, NULL, szErrorTextBuffer);
      printf ("%s\n", szErrorTextBuffer);
      digiCardCleanUp(dc);
      exit(1);
    }


  struct timespec timeStart, timeNow;
  clock_gettime(CLOCK_REALTIME, &timeStart);

  while (1) {
    if ((dwError = spcm_dwSetParam_i32 (dc->hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
      {
	if (dwError == ERR_TIMEOUT)
	  printf ("DMA wait timeout\n");
	else
	  printf ("DMA wait error: %d\n", dwError);
	digiCardCleanUp(dc);
	exit(1);
      }

    spcm_dwGetParam_i32 (dc->hCard, SPC_M2STATUS,             &lStatus);
    spcm_dwGetParam_i32 (dc->hCard, SPC_DATA_AVAIL_USER_LEN,  &lAvailUser);
    spcm_dwGetParam_i32 (dc->hCard, SPC_DATA_AVAIL_USER_POS,  &lPCPos);
    spcm_dwGetParam_i32 (dc->hCard, SPC_FILLSIZEPROMILLE,  &fill);
    
    if (lAvailUser >= dc->lNotifySize)
      {
	clock_gettime(CLOCK_REALTIME, &timeNow);
	double accum = deltaT(timeStart, timeNow);
	printf("Time: %fs;  digitizer buffer fill %i/1000 \n", accum, fill);
	int8_t* bufstart=((int8_t*)dc->pnData+lPCPos);
	gpuProcessBuffer(gc,bufstart);
	
	// check for esape = abort
	if (bKbhit ())
	  if (cGetch () == 27)
	    break;
      }
  }
    
  printf("Sending stop command\n");
  // send the stop command
  dwError = spcm_dwSetParam_i32 (dc->hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);
  if (dwError != ERR_OK)
    {
      spcm_dwGetErrorInfo_i32 (dc->hCard, NULL, NULL, szErrorTextBuffer);
      printf ("%s\n", szErrorTextBuffer);
      digiCardCleanUp(dc);
      exit(1);
    }

}


void digiCardCleanUp(DIGICARD *card) {
    vFreeMemPageAligned (card->pnData, (uint64) card->lBufferSize);
    spcm_vClose (card->hCard);
}