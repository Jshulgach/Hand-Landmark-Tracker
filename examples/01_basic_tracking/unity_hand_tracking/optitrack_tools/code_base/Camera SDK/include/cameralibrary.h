
//==================================================================================-----
//== Copyright NaturalPoint, All Rights Reserved
//==================================================================================-----

#ifndef __CAMERALIBRARY__CAMERALIBRARY_H__
#define __CAMERALIBRARY__CAMERALIBRARY_H__

#include "cameralibraryglobals.h"
#include "cameralibrarysettings.h"
#include "cameramanager.h"
#include "camera.h"
#include "camerarev4.h"
#include "camerarev5.h"
#include "camerarev6.h"
#include "camerarev7.h"
#include "camerarev8.h"
#include "camerarev9.h"
#include "camerarev10.h"
#include "camerarev11.h"
#include "camerarev12.h"
#include "camerarev13.h"
#include "camerarev14.h"
#include "camerarev15.h"
#include "camerarev16.h"
#include "camerarev17.h"
#include "camerarev18.h"
#include "camerarev21.h"
#include "camerarev23.h"
#include "camerarev24.h"
#include "camerarev25.h"
#include "camerarev26.h"
#include "camerarev27.h"
#include "camerarev29.h"
#include "camerarev30.h"
#include "camerarev31.h"
#include "camerarev32.h"
#include "camerarev33.h"
#include "camerarev34.h"
#include "camerarev35.h"
#include "camerarev36.h"
#include "frame.h"
#include "cameracommands.h"
#include "bitmap.h"
#include "modulefileoutput.h"
#include "modulelogger.h"
#include "moduleconsole.h"
#include "modulelinearwand.h"
#include "moduleprojection.h"
#include "modulesync.h"
#include "modulevector.h"
#include "modulevectorprocessing.h"
#include "cameratypes.h"
#include "singleton.h"

//== Camera Library Initialization / Deinitialization ===============================----

CLAPI void CameraLibraryStartup();
CLAPI void CameraLibraryShutdown();


class CLAPI cCameraLibrary {
public:
    cCameraLibrary(void);
};

class GlobalLibraryStartup
{
public:
    GlobalLibraryStartup();
    ~GlobalLibraryStartup();
    bool Alive() { return true; };
};

extern GlobalLibraryStartup gStartupLibrary;

#ifdef _DEBUG
#define CameraLibrary_EnableDevelopment() cCameraLibraryStartupSettings::X().EnableDevelopment()
#else
#define CameraLibrary_EnableDevelopment()
#endif


#endif
