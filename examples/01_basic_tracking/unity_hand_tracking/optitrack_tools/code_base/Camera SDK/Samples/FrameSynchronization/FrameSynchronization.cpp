
#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include <conio.h>

#include "cameralibrary.h"
using namespace CameraLibrary;

bool PopWaitingDialog();

int main(int argc, char* argv[])
{
    printf("==============================================================================\n");
    printf("== Frame Synchronization Example                     NaturalPoint OptiTrack ==\n");
    printf("==============================================================================\n\n");

    //== This example attaches a frame synchronizer for all connected cameras.  Once
    //== attached, it receives their synchronized frame data as a FrameGroup vs. the
    //== GetFrame approach used for a single camera.

    //== For OptiTrack Ethernet cameras, it's important to enable development mode if you
    //== want to stop execution for an extended time while debugging without disconnecting
    //== the Ethernet devices.  Lets do that now:

    CameraLibrary_EnableDevelopment();

	//== Next, lets wait until all connected cameras are initialized ==

    printf("Waiting for cameras to spin up...");
    
	//== Initialize Camera SDK ==--

	CameraLibrary::CameraManager::X();

	//== At this point the Camera SDK is actively looking for all connected cameras and will initialize
	//== them on it's own.

	//== Now, lets pop a dialog that will persist until there is at least one camera that is initialized
	//== or until canceled.

	PopWaitingDialog();
    
    //== Verify all cameras are initialized ==--

    if(CameraManager::X().AreCamerasInitialized())
        printf("complete\n\n");
    else
    {
        printf("failed (return key to exit)");
        getchar();
        return 1;
    }

    //== Connect to all connected cameras ==----

    Camera *camera[kMaxCameras];
    int     cameraCount=0;

    CameraList list;

    printf("Cameras:\n");

    for(int i=0; i<list.Count(); i++)
    {
        printf("Camera %d >> %s\n", i, list[i].Name());

        camera[i] = CameraManager::X().GetCamera(list[i].UID());

        if(camera[i]==0)
        {
            printf("unable to connected to camera...\n");
        }
        else
        {
            cameraCount++;
        }
    }
        
    if(cameraCount==0)
    {
        printf("no cameras (return key to exit)");
        getchar();
        return 1;
    }

    printf("\n");

    //== Create and attach frame synchronizer ==--

    cModuleSync * sync = cModuleSync::Create();

    for(int i=0; i<cameraCount; i++)
    {
        sync->AddCamera(camera[i]);
    }

    //== Start cameras ==--

    printf("Starting cameras... (any key to exit)\n\n");

    for(int i=0; i<cameraCount; i++)
    {
        camera[i]->Start();
        camera[i]->SetVideoType(Core::SegmentMode);
    }

    //== Pool for frame groups ==--
    
    while(!_kbhit())
    {
        FrameGroup *frameGroup = sync->GetFrameGroup();

        if(frameGroup)
        {
            //== Print something every 100 frames ==--

            static int frameThrottle = 0;
            static int frameCount    = 0;
            frameThrottle = (frameThrottle+1)%100;

            frameCount++;

            if(frameThrottle==0)
            {
                //== Ok, lets print something about this frame group ==--

                printf("Received Group Frame #%d (Contains %d frames)   ", frameCount, frameGroup->Count());

                if(sync->LastFrameGroupMode()==FrameGroup::Hardware)
                {
                    printf("Synchronized\n");
                }
                else
                {
                    printf("Unsynchronized\n");
                }

                for(int i=0; i<frameGroup->Count(); i++)
                {
                    Frame * frame = frameGroup->GetFrame(i);

                    printf("  - Camera #%d is reporting %d 2D objects\n", i, frame->ObjectCount());

                    frame->Release();
                }

                printf("\n");
            }

            frameGroup->Release();
        }

        Sleep(2);
    }

    //== Destroy synchronizer ==--

    sync->RemoveAllCameras();
    
    cModuleSync::Destroy( sync );

    //== Release cameras ==--

    for(int i=0; i<cameraCount; i++)
    {
        camera[i]->Release();
    }

    //== Disconnect devices and shutdown Camera Library ==--

    CameraManager::X().Shutdown();

	return 0;
}

//== Code to pop a simple dialog for 'waiting for cameras' using a message box and
//== no resources required for this sample application.

HHOOK hHook = NULL;

LRESULT CALLBACK CBTHookProc(int nCode, WPARAM wParam, LPARAM lParam)
{
	if (nCode < 0)
    {
		return CallNextHookEx(hHook, nCode, wParam, lParam);
    }

	if (nCode == HCBT_ACTIVATE)
	{
		HWND hWnd = reinterpret_cast<HWND>(wParam);
		SetWindowText(GetDlgItem(hWnd, IDOK), TEXT("Cancel"));
		return 0;
	}

	return CallNextHookEx(hHook, nCode, wParam, lParam);
}

VOID CALLBACK TimerProc(HWND hWnd, UINT uMsg, UINT idEvent, DWORD dwTime)
{
	CameraLibrary::CameraList list;

	bool found = false;

	for( int i=0; i<list.Count(); i++ )
	{
		if( list[i].State()==CameraLibrary::Initialized )
		{
			found = true;
		}
	}

	if(found==true)
	{
		HWND hWndActive = GetActiveWindow();

		if( hWndActive!=0 )
		{
			SendMessage(hWndActive, WM_COMMAND, IDCANCEL, 0);
		}
	}
}

bool PopWaitingDialog()
{
	//== hook in so we can create a message box that has only a 'Cancel' button ==--

	hHook = SetWindowsHookEx(WH_CBT, reinterpret_cast<HOOKPROC>(&CBTHookProc), NULL, GetCurrentThreadId());

	UINT_PTR nTimer = SetTimer(0, 100, 3000, TimerProc);
	int iResult = MessageBox( 0, _TEXT("waiting for connected cameras..."), _TEXT("Camera Initialization"), MB_OK );

	if( iResult == IDOK )
	{
		//== user has clicked the cancel button ==--
		UnhookWindowsHookEx(hHook);
		return false;
	}

	KillTimer(0, nTimer);

	return true;
}
