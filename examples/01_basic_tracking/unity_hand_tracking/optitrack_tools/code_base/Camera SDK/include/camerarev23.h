
//======================================================================================================-----
//== NaturalPoint 2012
//======================================================================================================-----

//== This is the >>> OptiTrack eSync <<<

#ifndef __CAMERALIBRARY__CAMERAREV23_H__
#define __CAMERALIBRARY__CAMERAREV23_H__

//== INCLUDES ===========================================================================================----

#include "camerarev11.h"
#include "camerarevisions.h"

//== GLOBAL DEFINITIONS AND SETTINGS ====================================================================----

class cInputBase;

namespace CameraLibrary
{
	class CameraRev23 : public CameraRev11
    {
    public:
        CameraRev23();

        virtual bool IsCamera()       { return false; }  //== Reports of device is a camera ===---
        virtual bool IsSyncAuthority(){ return true;  }
		virtual void SetFrameRate(int Value);

        cSyncFeatures  SyncFeatures(); //== Return devices supported synchronization features =---

        virtual bool   SetParameter(const char* ParameterName, const char* ParameterValue);

    };

	const unsigned int kESYNC_LOCK_SRC_MASK	   = 0x000F;
	const unsigned int kESYNC_STS_ESYNC2_DEV   = 0x0010;
	const unsigned int kESYNC_STS_MASTER       = 0x0100;
	const unsigned int kESYNC_STS_EXT_LOCK     = 0x0200;
	const unsigned int kESYNC_STS_RECORDING    = 0x0400;
	const unsigned int kESYNC_STS_TC_VALID     = 0x0800;
	const unsigned int kESYNC_STS_INPUT_UPDATE = 0x1000;

	const unsigned int kESYNC1_INPUT_COUNT	   = 7;
	const unsigned int kESYNC2_INPUT_COUNT     = 9;  //== 8 real + 1 virtual ( CAST's Time Keeper )
	const unsigned int kESYNC_MAX_INPUT_COUNT  = 9;

    struct sRev23SyncTelemetry
    {                                        //== Engineer's Comments ===----

        unsigned short PacketType;           // 0x3000
        unsigned short PacketSize;           // total packet size
        unsigned int   FrameID;              // ubiquitous frameid
        unsigned int   TimeStampLower;       // upper 32-bits of the hardware timestamp
        unsigned int   TimeStampUpper;       // lower 32-bits of the hardware timestamp
        unsigned int   TimeStampFrequency;   // hardware timestamp frequency 
        unsigned int   TimeCode;             // hh:mm:ss:ff (ff in LSB, hh in MSB)
        unsigned int   TimeCodeSubFrame;     // sub-frame within the given timecode
        unsigned int   LockPeriod;           // the true capture period (reported in 10ns clocks); zero if free-running
		unsigned int   Status;				 // various status bits
        unsigned int   InputRates[kESYNC_MAX_INPUT_COUNT]; // measured rates on all inputs (reported as
                                                           // number of 10ns clocks in one period)

		bool IsMasterTimingDevice()	const { return (Status & kESYNC_STS_MASTER)       != 0; }
		bool ExternalLock()			const { return (Status & kESYNC_STS_EXT_LOCK)     != 0; }
		bool RecordStatus()			const { return (Status & kESYNC_STS_RECORDING)    != 0; }
		bool TimeCodeValid()		const { return (Status & kESYNC_STS_TC_VALID)     != 0; }
		bool InputsUpdated()		const { return (Status & kESYNC_STS_INPUT_UPDATE) != 0; }
		bool IsEsync2()			    const { return (Status & kESYNC_STS_ESYNC2_DEV)   != 0; }
		
		int LockSource() const 
		{
			int src = Status & kESYNC_LOCK_SRC_MASK;
			return src == kESYNC_LOCK_SRC_MASK ? -1 : src;
		}

		int InputCount() const { return IsEsync2() ? kESYNC2_INPUT_COUNT : kESYNC1_INPUT_COUNT; }
    };
}




#endif
