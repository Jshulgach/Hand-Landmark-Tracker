
#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

#include <windows.h>
#include <stdio.h>
#include <tchar.h>

#include "cameralibrary.h"
using namespace CameraLibrary;

bool PopWaitingDialog();

int _tmain(int argc, _TCHAR* argv[])
{
    printf("==============================================================================\n");
    printf("== Camera Library Camera Test                        NaturalPoint OptiTrack ==\n");
    printf("==============================================================================\n\n");

    printf("Waiting for cameras to spin up...");

	PopWaitingDialog();
    
    //== Initialize Camera Library ==----

    CameraManager::X().WaitForInitialization();
    
    //== Verify all cameras are initialized ==--

    if(CameraManager::X().AreCamerasInitialized())
        printf("complete\n\n");
    else
        printf("failed\n\n");

    //== List all connected cameras ==----
    {
        CameraList list;

        printf("Cameras:\n");

        for(int i=0; i<list.Count(); i++)
            printf("Camera %d >> %s\n", i, list[i].Name());
        
        if(list.Count()==0)
            printf("None\n");

        printf("\n");
    }

    //== List all connected OptiHubs ==----
    {
        HubList list;

        printf("OptiHubs:\n");

        for(int i=0; i<list.Count(); i++)
            printf("OptiHub %d >> %s\n", i, list[i].Name());

        if(list.Count()==0)
            printf("None\n");
    }

    printf("\n\nreturn key to exit...");
    getchar();

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
		return CallNextHookEx(hHook, nCode, wParam, lParam);

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

	for( int i=0; i<list.Count(); i++ )
	{
		if( list[i].State()==CameraLibrary::Initializing )
		{
			found = false;
			//== suppress continuing until all cameras are initialized ==--
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
