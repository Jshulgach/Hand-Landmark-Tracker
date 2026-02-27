
//======================================================================================================-----
//== Copyright NaturalPoint, All Rights Reserved
//======================================================================================================-----


#ifndef __CAMERALIBRARY__CAMERAREV36_H__
#define __CAMERALIBRARY__CAMERAREV36_H__

//== INCLUDES ===========================================================================================----

#include "camerarev30.h"
#include "camerarevisions.h"
#include "lock.h"

//== GLOBAL DEFINITIONS AND SETTINGS ====================================================================----

class cInputBase;

namespace CameraLibrary
{
	class CameraRev36 : public CameraRev30
    {
    public:
        CameraRev36();

        //== Camera Information =========================================================================----

        virtual void   GetDistortionModel(Core::DistortionModel &Model);    //== Distortion Model ========---

    };
}




#endif
