
//=================================================================================-----
//== NaturalPoint 2010
//== Camera Library SDK Sample
//==
//== This sample brings up a connected camera and displays it's MJPEG output. 
//=================================================================================-----

#include "supportcode.h"       //== Boiler-plate code for application window init ===---
#include "cameralibrary.h"     //== Camera Library header file ======================---
using namespace CameraLibrary; 

int main(int argc, char* argv[])
{
	//== For OptiTrack Ethernet cameras, it's important to enable development mode if you
	//== want to stop execution for an extended time while debugging without disconnecting
	//== the Ethernet devices.  Lets do that now:

    CameraLibrary_EnableDevelopment();

	//== Initialize Camera SDK ==--

	CameraLibrary::CameraManager::X();

	//== At this point the Camera SDK is actively looking for all connected cameras and will initialize
	//== them on it's own.

	//== Now, lets pop a dialog that will persist until there is at least one camera that is initialized
	//== or until canceled.

	PopWaitingDialog();

    //== Get a connected camera ================----

    Camera *camera = CameraManager::X().GetCamera();

    //== If no device connected, pop a message box and exit ==--

    if(camera==0)
    {
        MessageBox(0,"Please connect a camera","No Device Connected", MB_OK);
        return 1;
    }

	//== Determine camera resolution to size application window ==----
	
    int cameraWidth  = camera->Width();
    int cameraHeight = camera->Height();

 	//== Open the application window =============================----
		
    if (!CreateAppWindow("Camera Library SDK - Sample",cameraWidth,cameraHeight,32,gFullscreen))
	    return 0;

    //== Create a texture to push the rasterized camera image ====----

    //== We're using textures because it's an easy & cpu light 
    //== way to utilize the 3D hardware to display camera
    //== imagery at high frame rates

    Surface  Texture(cameraWidth, cameraHeight);
    Bitmap * framebuffer = new Bitmap(cameraWidth, cameraHeight, Texture.PixelSpan()*4,
                               Bitmap::ThirtyTwoBit, Texture.GetBuffer());

    //== MJPEG Mode ==--

    camera->SetVideoType(Core::MJPEGMode);  //== Select MJPEG Video ===================---
    camera->SetAEC(true);             //== Enable Automatic Exposure Control ====---
    camera->SetAGC(true);             //== Enable Automatic Gain Control ========---

    //== Start camera output ==--

    camera->Start();

    //== Turn on some overlay text so it's clear things are     ===---
    //== working even if there is nothing in the camera's view. ===---

    camera->SetTextOverlay(true);

    //== Ok, start main loop.  This loop fetches and displays   ===---
    //== camera frames.                                         ===---

    while(1)
    {
        //== Fetch a new frame from the camera ===---

        Frame *frame = camera->GetFrame();

        if(frame)
        {
            //== Ok, we've received a new frame, lets do something
            //== with it.

            //== Lets have the Camera Library raster the camera's
            //== image into our texture.

            frame->Rasterize(framebuffer);

            //== Display Camera Image ============--

            if(!DrawGLScene(&Texture, frame->Width(), frame->Height()))  
                break;

            //== Escape key to exit application ==--

            if (keys[VK_ESCAPE])
                break;

            //== Release frame =========--

            frame->Release();
        }

	    Sleep(2);

        //== Service Windows Message System ==--

        if(!PumpMessages())
            break;
    }

    //== Close window ==--

    CloseWindow();

    //== Release camera ==--

    camera->Release();

    //== Shutdown Camera Library ==--

    CameraManager::X().Shutdown();

    //== Exit the application.  Simple! ==--

	return 1;
}

