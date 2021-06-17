//Header
#include<Windows.h>
#include<stdio.h>
#include<gl/GL.h>
#include<gl/glu.h>
#include"MyResources.h"

//Library Functions
#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "glu32.lib")
#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "kernel32.lib")
#pragma comment (lib, "Winmm.lib")

//Macros
#define WIN_WIDTH 1366
#define WIN_HEIGHT 768
#define X (GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH)/2
#define Y (GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT)/2

//Funtion declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//Global Variable
HWND ghwnd;
DWORD dwStyle;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
FILE *gpFile = NULL;
bool bIsFullScreen = false;
bool gbActiveWindow = false;
HINSTANCE hInstance;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
GLfloat x_scale = 0.75f;
GLfloat y_scale = 0.95f;
GLfloat height = -0.35f;
GLfloat Bounce = 1.0f;
GLfloat y_translation = 2.0f;
bool bIsBallTouchTheGround = false;
bool bIsBallReachedToSpecificHeight = false;
GLUquadric *Quadric;

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	//Variable declaration
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MyApp");
	bool bDone = false;
	int iRet = 0;

	//function declaration
	int OGLInitialise(void);
	void OGLDisplay(void);
	void OGLUpdate(void);

	if(fopen_s(&gpFile, "AP_Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Can't Be Created"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	else
	{
		fprintf_s(gpFile, "Log File Is Created\n");
	}
	fprintf_s(gpFile, "1. In WinMain\n");
	//code
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
	wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
	wndclass.hInstance = hInstance;
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;

	//Register Class
	RegisterClassEx(&wndclass);

	//CreateWindow
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("Triangle And Rectangle"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		X,
		Y,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	fprintf_s(gpFile, "1.1 Going To OGLInitialise\n\n");
	iRet = OGLInitialise();

	if(iRet == -1)
	{
		fprintf_s(gpFile, "Choose Pixel Format Failed\n");
		DestroyWindow(hwnd);
	}
	else if(iRet == -2)
	{
		fprintf_s(gpFile, "Set Pixel Format Failed\n");
		DestroyWindow(hwnd);
	}
	else if(iRet == -3)
	{
		fprintf_s(gpFile, "wglCreateContext Failed\n");
		DestroyWindow(hwnd);
	}
	else if(iRet == -4)
	{
		fprintf_s(gpFile, "wglMakeCurrent Failed\n");
		DestroyWindow(hwnd);
	}
	else
	{
		fprintf_s(gpFile, "*******Initialization Is Successfully Done*******\n");
	}

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	//Game Loop
	while(bDone == false)
	{
		if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if(msg.message == WM_QUIT)
			{
				bDone = true;
				break;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}

		else
		{
			OGLDisplay();
			if(gbActiveWindow == true)
			{
				OGLUpdate();
			}
			
		}
	}

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{	
	fprintf_s(gpFile, "\n2. In WndProc\n");
	//function declaration
	void ToggleFullScreen(void);
	void OGLResize(int, int);
	void OGLUninitialise(void);

	//code
	switch(iMsg)
	{
	
	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;
	
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;
		}
		break;
/*
	case WM_CHAR:
		switch (wParam)
		{
		case 'F':
		case 'f':
			fprintf_s(gpFile, "2.1 Going To ToggleFullScreen()\n\n");
			ToggleFullScreen();
			break;
		}
		break;	
*/

	case WM_SIZE:
			fprintf_s(gpFile, "2.2 Going To OGLResize()\n\n");
			OGLResize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_ERASEBKGND:
			return(0);
			break;
	case WM_CLOSE:
			DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		OGLUninitialise();
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));	
}

int OGLInitialise(void)
{
	fprintf_s(gpFile, "3. In OGLInitialise()\n");
	//variable declaration
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	//function declaration
	void OGLResize(int, int);
	void ToggleFullScreen(void);

	//code
	memset((void *)&pfd, NULL, sizeof(PIXELFORMATDESCRIPTOR));
	fprintf_s(gpFile, "3.1 Memory Set For PIXELFORMATDESCRIPTOR And Filling The Structure Of PIXELFORMATDESCRIPTOR.\n");
	
	//PIXELFORMATDESCRIPTOR Initialisation
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd. dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cDepthBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;

	fprintf_s(gpFile, "3.2 Filled The Structure Of PIXELFORMATDESCRIPTOR.\n");
	fprintf_s(gpFile, "3.3 Getting Normal hdc\n");
	ghdc = GetDC(ghwnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);

	if(iPixelFormatIndex == 0)
	{
		return(-1);
	}
	if(SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		return(-2);
	}
	ghrc = wglCreateContext(ghdc);
	if(ghrc == NULL)
	{
		return(-3);
	}
	if(wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		return(-4);
	}
	fprintf_s(gpFile, "3.4 WarmUp Call To OpenGL\n");
	
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	ToggleFullScreen();
	fprintf_s(gpFile, "3.5 Going In OGLResize()\n\n");
	OGLResize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void ToggleFullScreen(void)
{
	fprintf_s(gpFile, "4. In ToggleFullScreen()\n");
	//Variable declaration
	MONITORINFO mi;

	//code
	if(bIsFullScreen == FALSE)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if(dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };
			if(GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
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
		ShowCursor(FALSE);
		bIsFullScreen = TRUE;
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
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
}

void OGLResize(int width, int height)
{
	fprintf_s(gpFile, "5. In OGLResize()\n");
	
	if(height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f,
		((GLfloat)width / (GLfloat)height),
		0.1f,
		100.0f);

}

void OGLDisplay(void)
{
	fprintf_s(gpFile, "6. In OGLDisplay()\n");
	//function declaration
	void Ball(void);
	void Surface(void);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -4.0f);

	Surface();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, y_translation, -4.0f);
	if(bIsBallReachedToSpecificHeight)
	{
		glScalef(x_scale, y_scale, 0.75f);
	}
	glScalef(0.75f, 0.95f, 0.75f);
	Ball();

	SwapBuffers(ghdc);
}

void OGLUpdate(void)
{
	if(bIsBallTouchTheGround == false)
	{
		y_translation = y_translation - 0.05f;
		if(y_translation < -0.55f)
		{
			bIsBallReachedToSpecificHeight = true;
			if((y_scale > 0.75f) && (x_scale < 0.95f))
			{
				x_scale = x_scale + 0.09f;
				y_scale = y_scale - 0.09f;	
			}
			 
		}
		if(y_translation < -0.86f)
		{
			//PlaySound(MAKEINTRESOURCE(MYAUDIO), hInstance, SND_ASYNC | SND_RESOURCE | SND_NODEFAULT);
			bIsBallTouchTheGround = true;
			Bounce = Bounce - 0.08f;
		}
	}
	if(bIsBallTouchTheGround == true)
	{
		if(y_translation < Bounce)
		{
			y_translation = y_translation + 0.07f;
			if(y_translation > Bounce)
			{
				bIsBallTouchTheGround = false;
			}	
		}
		if(y_translation > -0.35f)
		{
			bIsBallReachedToSpecificHeight = false;
		}
	}
}

void Ball(void)
{
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	Quadric = gluNewQuadric();
	glColor3f(0.9f, 0.0f, 0.0f);
	gluSphere(Quadric, 0.2f, 30, 30);
}

void Surface(void)
{
	glBegin(GL_QUADS);

	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	
	glColor3f(0.5f, 0.5f, 0.5f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	
	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);

	glColor3f(0.5f, 0.5f, 0.5f);
	glVertex3f(1.0f, -1.0f, 1.0f);

	glEnd();
}

void OGLUninitialise(void)
{
	fprintf_s(gpFile, "7. In OGLUninitialise()\n");
	if(bIsFullScreen == true)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}

	if(wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if(ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if(ghdc)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if(gpFile)
	{
		fprintf_s(gpFile, "Log File Is Closed Successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}
