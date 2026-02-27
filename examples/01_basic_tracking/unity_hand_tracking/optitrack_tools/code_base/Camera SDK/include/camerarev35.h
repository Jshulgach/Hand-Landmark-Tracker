
//======================================================================================================-----
//== NaturalPoint 2015
//======================================================================================================-----

//== This is the >>> TrackIR 5.5 <<<

#ifndef __CAMERALIBRARY__CAMERAREV35_H__
#define __CAMERALIBRARY__CAMERAREV35_H__

//== INCLUDES ===========================================================================================----

#include "Core/RandomNumberGenerator.h"
#include "camerarev18.h"

//== GLOBAL DEFINITIONS AND SETTINGS ====================================================================----

namespace CameraLibrary
{
    class CameraRev35 : public CameraRev18
    {
    public:
        CameraRev35();
        ~CameraRev35();
        
        virtual double  ImagerWidth();
        virtual double  ImagerHeight();
        virtual double  FocalLength();
        virtual void    GetDistortionModel( Core::DistortionModel &Model );

    };
}

#endif
