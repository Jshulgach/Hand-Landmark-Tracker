
//======================================================================================================-----
//== NaturalPoint 2012
//======================================================================================================-----

//== This is the >>> OptiTrack eSync 2 <<<

#ifndef __CAMERALIBRARY__CAMERAREV34_H__
#define __CAMERALIBRARY__CAMERAREV34_H__

//== INCLUDES ===========================================================================================----

#include "camerarev23.h"
#include "camerarevisions.h"

//== GLOBAL DEFINITIONS AND SETTINGS ====================================================================----

class cInputBase;

namespace CameraLibrary
{
	class CameraRev34 : public CameraRev23
    {
    public:
        CameraRev34();

        cSyncFeatures  SyncFeatures(); //== Return devices supported synchronization features =---

    };
}




#endif
