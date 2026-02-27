
#if _MSC_VER // this is defined when compiling with Visual Studio
#define EXPORT_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define EXPORT_API // XCode does not need annotating exported functions, so define is empty
#endif

#include "cameralibrary.h"

// ------------------------------------------------------------------------
// Plugin itself

CameraLibrary::Camera *gCamera=0;

CameraLibrary::Frame *gFrame=0;

// Link following functions C-style (required for plugins)
extern "C"
{

// The functions we will call from Unity.
//

void EXPORT_API VRPNStartup()
{
}

void EXPORT_API OptiTrackTick()
{
	//== Acquire first camera we can get our hands on ==--

	if(gCamera==0)
	{
		gCamera=CameraLibrary::CameraManager::X().GetCamera();

		if(gCamera)
		{
			gCamera->SetIntensity(15);
			gCamera->SetExposure(25);
			gCamera->SetThreshold(200);
			gCamera->Start();
		}
	}


	//== Early out if there is no camera yet ==--
	if(gCamera==0)
		return;

	//== Try to grab a new frame ==--

	CameraLibrary::Frame *frame = gCamera->GetLatestFrame();

	if(frame)
	{
		//== We've got a new frame, lets get rid of the old one and
		//== hold onto this one until the next frame comes along

		if(gFrame!=0)
			gFrame->Release();

		gFrame=frame;
	}

	//== That's it! ==--
}

int EXPORT_API OptiTrackCameraCount()
{
	if(gCamera!=0)
		return 1;
	return 0;
}

int EXPORT_API OptiTrackObjectCount()
{
	if(gFrame==0)
		return 0;

	return gFrame->ObjectCount();
}

float EXPORT_API OptiTrackObjectX(int ObjectIndex)
{
	if(gFrame==0)
		return 0;

	if(ObjectIndex<0 || ObjectIndex>=gFrame->ObjectCount())
		return 0;

	return gFrame->Object(ObjectIndex)->X();
}

float EXPORT_API OptiTrackObjectY(int ObjectIndex)
{
	if(gFrame==0)
		return 0;

	if(ObjectIndex<0 || ObjectIndex>=gFrame->ObjectCount())
		return 0;

	return gFrame->Object(ObjectIndex)->Y();
}

float EXPORT_API OptiTrackObjectArea(int ObjectIndex)
{
	if(gFrame==0)
		return 0;

	if(ObjectIndex<0 || ObjectIndex>=gFrame->ObjectCount())
		return 0;

	return gFrame->Object(ObjectIndex)->Area();
}

}