
//======================================================================================================-----
//== Copyright NaturalPoint, All Rights Reserved
//======================================================================================================-----


#ifndef __CAMERALIBRARY__CAMERAREV37_H__
#define __CAMERALIBRARY__CAMERAREV37_H__

//== INCLUDES ===========================================================================================----

#include "camerarev36.h"

namespace CameraLibrary
{
    class CameraRev37 : public CameraRev36
    {
    public:
        CameraRev37();

        //== Camera Information =========================================================================----

        virtual int  MaximumFrameRateValue();         //== Returns the maximum frame rate ===============----

    };
}
#endif

