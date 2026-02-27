
//=================================================================================-----
//== NaturalPoint 2010
//== Camera Library SDK Sample
//==
//== This sample brings up a connected camera and displays it's output frames.
//=================================================================================-----

#ifdef WIN32
#include "supportcode.h"       //== Boiler-plate code for application window init ===---
#else
#include "SDL/SDL.h"
#include "lock.h"
#endif

#include "cameralibrary.h"     //== Camera Library header file ======================---
using namespace CameraLibrary;

#ifdef WIN32 // Windows implementation

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

    //== Set Video Mode ==--

    camera->SetVideoType(Core::SegmentMode);
    
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

#else // Linux implementation

void DrawCamera(SDL_Surface *screen, CameraLibrary::Frame *frame)
{
	if ( SDL_MUSTLOCK(screen) ) {
		if ( SDL_LockSurface(screen) < 0 ) {
			return;
		}
	}

	Bitmap *bitmap = new Bitmap(screen->w,screen->h,screen->pitch,(Bitmap::ColorDepth)screen->format->BitsPerPixel, (unsigned char*)screen->pixels);
	frame->Rasterize(bitmap);
	delete bitmap;
}

void DrawPixel(SDL_Surface *screen, int x, int y, Uint8 R, Uint8 G, Uint8 B)
{
	Uint32 color = SDL_MapRGB(screen->format, R, G, B);

	if ( SDL_MUSTLOCK(screen) ) {
		if ( SDL_LockSurface(screen) < 0 ) {
			return;
		}
	}
	switch (screen->format->BytesPerPixel) {
		case 1: { /* Assuming 8-bpp */
			Uint8 *bufp;

			bufp = (Uint8 *)screen->pixels + y*screen->pitch + x;
			*bufp = color;
				}
				break;

		case 2: { /* Probably 15-bpp or 16-bpp */
			Uint16 *bufp;

			bufp = (Uint16 *)screen->pixels + y*screen->pitch/2 + x;
			*bufp = color;
				}
				break;

		case 3: { /* Slow 24-bpp mode, usually not used */
			Uint8 *bufp;

			bufp = (Uint8 *)screen->pixels + y*screen->pitch + x;
			*(bufp+screen->format->Rshift/8) = R;
			*(bufp+screen->format->Gshift/8) = G;
			*(bufp+screen->format->Bshift/8) = B;
				}
				break;

		case 4: { /* Probably 32-bpp */
			Uint32 *bufp;

			bufp = (Uint32 *)screen->pixels + y*screen->pitch/4 + x;
			*bufp = color;
				}
				break;
	}
}


int main( int argc, char* argv[])
{
	// Fire up SDL, this starts all subsystems; audio video etc.
	// SDL_Init(SDL_INIT_EVERYTHING);

	printf("top of main...\n");

	CameraLibrary::CameraManager::X();

	SDL_Init(SDL_INIT_VIDEO);
	putenv(strdup("SDL_VIDEO_CENTERED=1"));

	SDL_Surface *screen;

	screen = SDL_SetVideoMode(832, 832, 16, SDL_SWSURFACE);
	if ( screen == NULL ) {
		fprintf(stderr, "Unable to set 640x480 video: %s\n", SDL_GetError());
		exit(1);
	}

	SDL_WM_SetCaption("Camera SDK Sample", "Application");

	int done = 0;
	Uint8 *keys;

	CameraLibrary::CameraManager::X().WaitForInitialization();

	CameraLibrary::Camera *camera = CameraLibrary::CameraManager::X().GetCamera();

	if(camera)
	{
		printf("We have a camera! %s\n",camera->Name());
		camera->Start();
		camera->SetTextOverlay(true);
		camera->SetVideoType(CameraLibrary::MJPEGMode);
	}
	else
		printf("No cameras connected....\n");


	Frame *frame=0;

	while ( ! done )
	{
		if(camera)
		{

			bool newFrame = false;
			Frame *temp=camera->GetFrame();
			while(temp)
			{
				newFrame=true;
				if(frame)
					frame->Release();
				frame=temp;
				temp=camera->GetFrame();
			}

			if(frame && newFrame)
				DrawCamera(screen,frame);
		}


		SDL_Event event;

		while ( SDL_PollEvent(&event) )
		{
			switch(event.type)
			{
			case SDL_VIDEORESIZE:
				//screen = SDL_SetVideoMode(event.resize.w, event.resize.h, 16,
				//                    SDL_OPENGL|SDL_RESIZABLE);
				if ( screen ) {
				} else {
					// Uh oh, we couldn't set the new video mode??
				}
				break;
			case SDL_QUIT:
				done = 1;
				break;
			default:
				break;
			}
		}
		keys = SDL_GetKeyState(NULL);

		if ( keys[SDLK_ESCAPE] )
			done = 1;

		if ( SDL_MUSTLOCK(screen) )
			SDL_UnlockSurface(screen);

		SDL_UpdateRect(screen, 0, 0, screen->w, screen->h);

		timespec t = { 0, 60l * 1000l * 1000l };
		nanosleep(&t, NULL);
	}
	SDL_Quit();

	return 0;
}

#endif