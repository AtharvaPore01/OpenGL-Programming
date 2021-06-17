//Header
#include<Windows.h>

//Function Decalaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//Global Variable Declaration
bool bIsFullScreen = FALSE;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
HWND ghwnd;

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MyApp");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;

	//Register Class
	RegisterClassEx(&wndclass);

	//CreateWindow
	hwnd = CreateWindow(szAppName,
		TEXT("Message Handling Window"),
		WS_OVERLAPPEDWINDOW,
		100,
		100,
		800,
		600,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	UpdateWindow(hwnd);

	//Message Loop
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void ToggleFullScreen(void);
	switch (iMsg)
	{
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;
		case 0x46:
			ToggleFullScreen();
			break;
		}
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void)
{
	/* https:/docs.microsoft.com/en-us/previous-versions/ms812499(v=msdn.10) */
	//Variable Declaration
	MONITORINFO mi = { sizeof(MONITORINFO) };
	DEVMODE dm = { sizeof(DEVMODE) };
	dm.dmSize = sizeof(dm);
	DWORD iModeNum = 0;
	long ReturnValueOfChangeDisplaySettings;


	if (bIsFullScreen == false)
	{
		//GettingCurrentSettings

		EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm);

		//Screen Size
		dm.dmPelsWidth = mi.rcMonitor.right - mi.rcMonitor.left;
		dm.dmPelsHeight = mi.rcMonitor.bottom - mi.rcMonitor.top;
		
		//Konte Fields Set Karayche Te Sangitla
		dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT;

		//Change DisplaySettings
		ReturnValueOfChangeDisplaySettings = ChangeDisplaySettings(&dm, CDS_FULLSCREEN);
		if (ReturnValueOfChangeDisplaySettings == DISP_CHANGE_SUCCESSFUL)
		{
			dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
			if (dwStyle & WS_OVERLAPPEDWINDOW)
			{
				mi = { sizeof(MONITORINFO) };

				if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
				{
					SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
					SetWindowPos(ghwnd,
						HWND_TOP,
						mi.rcMonitor.left,
						mi.rcMonitor.top,
						mi.rcMonitor.right - mi.rcMonitor.left,
						mi.rcMonitor.bottom - mi.rcMonitor.top,
						SWP_NOZORDER | SWP_FRAMECHANGED);
				}
			}
		}
		ShowCursor(FALSE);
		bIsFullScreen = TRUE;
	}
	else
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);
		ShowCursor(TRUE);
		bIsFullScreen = FALSE;
	}
	
	
/*
	if (bIsFullScreen == false)
	{
		//To Check Whether Window Has WS_OVERLAPPEDWINDOW
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle && WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				ChangeDisplaySettings(, CDS_FULLSCREEN);
			}
		}
	}
*/
}
