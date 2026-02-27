
//=================================================================================-----
//== NaturalPoint 2010
//== Camera Library SDK Sample
//==
//== This sample brings up a connected camera and displays it's output frames.
//=================================================================================-----

#include "supportcode.h"       //== Boiler-plate code for application window init ===---
#include "cameralibrary.h"     //== Camera Library header file ======================---
#include "modulevector.h"
#include "modulevectorprocessing.h"
#include "coremath.h"

#include <gl/glu.h>

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

    int WindowWidth = 800;
    int WindowHeight = 450;

 	//== Open the application window =============================----
		
    if (!CreateAppWindow("Camera Library SDK - Single Camera Tracking Sample",WindowWidth,WindowHeight,32,gFullscreen))
	    return 0;

    //== Create a texture to push the rasterized camera image ====----

    //== We're using textures because it's an easy & cpu light 
    //== way to utilize the 3D hardware to display camera
    //== imagery at high frame rates

    Surface  Texture(cameraWidth, cameraHeight);
    Bitmap * framebuffer = new Bitmap(cameraWidth, cameraHeight, Texture.PixelSpan()*4,
                               Bitmap::ThirtyTwoBit, Texture.GetBuffer());

    //== Set Video Mode ==--

    //== We set the camera to Segment Mode here.  This mode is support by all of our products.
    //== Depending on what device you have connected you might want to consider a different
    //== video mode to achieve the best possible tracking quality.  All devices that support a
    //== mode that will achieve a better quality output with a mode other than Segment Mode are
    //== listed here along with what mode you should use if you're looking for the best head
    //== tracking:
    //==
    //==     V100:R1/R2    Precision Mode
    //==     TrackIR 5     Bit-Packed Precision Mode
    //==     V120          Precision Mode
    //==     TBar          Precision Mode
    //==     S250e         Precision Mode
    //==
    //== If you have questions about a new device that might be conspicuously missing here or
    //== have any questions about head tracking, email support or participate in our forums.

    camera->SetVideoType(Core::SegmentMode);
    
    //== Start camera output ==--

    camera->Start();

    //== Turn on some overlay text so it's clear things are     ===---
    //== working even if there is nothing in the camera's view. ===---

    camera->SetTextOverlay(false);

    cModuleVector *vec = cModuleVector::Create(); //new cModuleVector();
    cModuleVectorProcessing *vecprocessor = new cModuleVectorProcessing();

    Core::DistortionModel lensDistortion;

    camera->GetDistortionModel(lensDistortion);

    //== Plug distortion into vector module ==--

    cVectorSettings vectorSettings;
    vectorSettings = *vec->Settings();

    vectorSettings.Arrangement = cVectorSettings::VectorClip;
    vectorSettings.Enabled     = true;
    
    cVectorProcessingSettings vectorProcessorSettings;

    vectorProcessorSettings = *vecprocessor->Settings();

    vectorProcessorSettings.Arrangement = cVectorSettings::VectorClip;
    vectorProcessorSettings.ShowPivotPoint = false;
    vectorProcessorSettings.ShowProcessed  = false;

    vecprocessor->SetSettings(vectorProcessorSettings);

    //== Plug in focal length in (mm) by converting it from pixels -> mm

    vectorSettings.ImagerFocalLength =  (lensDistortion.HorizontalFocalLength/((float) camera->PhysicalPixelWidth()))*camera->ImagerWidth();

    vectorSettings.ImagerHeight = camera->ImagerHeight();
    vectorSettings.ImagerWidth  = camera->ImagerWidth();

    vectorSettings.PrincipalX   = camera->PhysicalPixelWidth()/2;
    vectorSettings.PrincipalY   = camera->PhysicalPixelHeight()/2;

    vectorSettings.PixelWidth   = camera->PhysicalPixelWidth();
    vectorSettings.PixelHeight  = camera->PhysicalPixelHeight();

    vec->SetSettings(vectorSettings);

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

            vec->BeginFrame();

            for(int i=0; i<frame->ObjectCount(); i++)
            {
                cObject *obj = frame->Object(i);

                float x = obj->X();
                float y = obj->Y();

                Core::Undistort2DPoint(lensDistortion,x,y);

                vec->PushMarkerData(x, y, obj->Area(), obj->Width(), obj->Height());
            }
            vec->Calculate();
            vecprocessor->PushData(vec);
            

            StartScene();

            glEnable(GL_BLEND);
            glColor4f(1,1,1,0.3f);
            glBegin(GL_LINES);
            glVertex3f(10,0,0);glVertex3f(-10, 0,0);
            glVertex3f(0,10,0);glVertex3f( 0,-10,0);
            glVertex3f(0,0,10);glVertex3f( 0, 0,-10);
            glEnd();

            if(vecprocessor->MarkerCount()>0)
            {
                glColor3f(0,1,1);
                glBegin(GL_LINES);
                
                for(int i=0; i<vecprocessor->MarkerCount(); i++)
                    for(int j=0; j<vecprocessor->MarkerCount(); j++)
                    {
                        if(i!=j)
                        {
                            float x,y,z;
                            vecprocessor->GetResult(i,x,y,z);
                            glVertex3f(x/200,y/200,z/200);
                            vecprocessor->GetResult(j,x,y,z);
                            glVertex3f(x/200,y/200,z/200);

                        }
                    }

                glEnd();

            }

            //== Display Camera Image ============--

            if(!DrawGLScene(&Texture))  
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

