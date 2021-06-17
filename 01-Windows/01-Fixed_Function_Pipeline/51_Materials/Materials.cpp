//Header
#include<Windows.h>
#include<stdio.h>
#include<gl/GL.h>
#include<gl/glu.h>

//Library Functions
#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "glu32.lib")
#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "kernel32.lib")

//Macros
#define WIN_WIDTH 800
#define WIN_HEIGHT 600
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
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

//light's Parameters
bool bLight = false;

GLfloat LightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat LightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat LightPosition[] = { 0.0f, 0.0f, 0.0f, 1.0f };  //{ 0.0f, 3.0f, 3.0f, 1.0f };

GLfloat light_model_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
GLfloat light_model_local_viewer[] = { 0.0f };

GLUquadric *quadric[24];

GLfloat anglOfXPosition = 0.0f;
GLfloat anglOfYPosition = 0.0f;
GLfloat anglOfZPosition = 0.0f;

GLint KeyPress = 0;

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

	if (fopen_s(&gpFile, "AP_Log.txt", "w") != 0)
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
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
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

	if (iRet == -1)
	{
		fprintf_s(gpFile, "Choose Pixel Format Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -2)
	{
		fprintf_s(gpFile, "Set Pixel Format Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -3)
	{
		fprintf_s(gpFile, "wglCreateContext Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -4)
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
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
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
			if (gbActiveWindow == true)
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
	switch (iMsg)
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

	case WM_CHAR:
		switch (wParam)
		{
		case 'F':
		case 'f':
			fprintf_s(gpFile, "2.1 Going To ToggleFullScreen()\n\n");
			ToggleFullScreen();
			break;
		case 'l':
		case 'L':
			if (bLight == false)
			{
				bLight = true;
				glEnable(GL_LIGHTING);
			}
			else
			{
				bLight = false;
				glDisable(GL_LIGHTING);
			}
			break;
		case 'x':
		case 'X':
			KeyPress = 1;
			anglOfXPosition = 0.0f;
			break;

		case 'y':
		case 'Y':
			KeyPress = 2;
			anglOfYPosition = 0.0f;
			break;

		case 'z':
		case 'Z':
			KeyPress = 3;
			anglOfZPosition = 0.0f;
			break;

		}
		break;


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

	//code
	memset((void *)&pfd, NULL, sizeof(PIXELFORMATDESCRIPTOR));
	fprintf_s(gpFile, "3.1 Memory Set For PIXELFORMATDESCRIPTOR And Filling The Structure Of PIXELFORMATDESCRIPTOR.\n");

	//PIXELFORMATDESCRIPTOR Initialisation
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
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

	if (iPixelFormatIndex == 0)
	{
		return(-1);
	}
	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		return(-2);
	}
	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		return(-3);
	}
	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		return(-4);
	}
	fprintf_s(gpFile, "3.4 WarmUp Call To OpenGL\n");

	glShadeModel(GL_SMOOTH);
	glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glEnable(GL_AUTO_NORMAL);
	glEnable(GL_NORMALIZE);

	glLightfv(GL_LIGHT0, GL_AMBIENT, LightAmbient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_model_ambient);
	glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, light_model_local_viewer);
	glEnable(GL_LIGHT0);

	for (int i = 0; i < 24; i++)
	{
		quadric[i] = gluNewQuadric();
	}

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
	if (bIsFullScreen == FALSE)
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

	if (height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	if (width <= height)
	{
		glOrtho(0.0f,
				15.5f,
				0.0f,
				(15.5f * ((GLfloat)height / (GLfloat)width)),
				-10.0f,
				10.0f);
	}
	else
	{
		glOrtho(0.0f,
				(15.5f * ((GLfloat)width / (GLfloat)height)),
				0.0f,
				15.5f,
				-10.0f,
				10.0f);
	}
	
}

void OGLDisplay(void)
{
	fprintf_s(gpFile, "6. In OGLDisplay()\n");
	//function declaration
	void draw24Spheres(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	if (KeyPress == 1)
	{
		glRotatef(anglOfXPosition, 1.0f, 0.0f, 0.0f);
		LightPosition[1] = anglOfXPosition;
	}
	else if (KeyPress == 2)
	{
		glRotatef(anglOfYPosition, 0.0f, 1.0f, 0.0f);
		LightPosition[2] = anglOfYPosition;
	}
	else
	{
		glRotatef(anglOfZPosition, 0.0f, 0.0f, 1.0f);
		LightPosition[0] = anglOfZPosition;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);
	
	//glTranslatef(0.0f, 0.0f, -0.55f);
	//glTranslatef(0.0f, 0.0f, -0.70f);
	//glTranslatef(0.0f, 0.0f, -0.75f);

	draw24Spheres();
	
	SwapBuffers(ghdc);
}

void OGLUpdate(void)
{
	//code
	anglOfXPosition = anglOfXPosition + 0.5f;
	if (anglOfXPosition > 360.0f)
	{
		anglOfXPosition = 0.0f;
	}
	anglOfYPosition = anglOfYPosition + 0.5f;
	if (anglOfYPosition > 360.0f)
	{
		anglOfYPosition = 0.0f;
	}
	anglOfZPosition = anglOfZPosition + 0.5f;
	if (anglOfZPosition > 360.0f)
	{
		anglOfZPosition = 0.0f;
	}

}

void OGLUninitialise(void)
{
	fprintf_s(gpFile, "7. In OGLUninitialise()\n");
	if (bIsFullScreen == true)
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

	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	for (int i = 0; i < 24; i++)
	{
		if (quadric[i])
		{
			gluDeleteQuadric(quadric[i]);
			quadric[i] = NULL;
		}
		
	}

	if (gpFile)
	{
		fprintf_s(gpFile, "Log File Is Closed Successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}


}

void draw24Spheres(void)
{
	//variable
	GLfloat MaterialAmbient[4];
	GLfloat MaterialSpecular[4];
	GLfloat MaterialDiffuse[4];
	GLfloat MaterialShininess[1];
	
	//code
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//-------------------------------------Emerald-------------------------------------//
	MaterialAmbient[0] = 0.0215f;
	MaterialAmbient[1] = 0.1745f;
	MaterialAmbient[2] = 0.0215f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.07568f;
	MaterialDiffuse[1] = 0.61424f;
	MaterialDiffuse[2] = 0.07568f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.633f;
	MaterialSpecular[1] = 0.727811f;
	MaterialSpecular[2] = 0.633f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.6f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 14.0f, 0.0f);
	gluSphere(quadric[0], 1.0f, 30, 30);

	//-------------------------------------Jade-------------------------------------//
	MaterialAmbient[0] = 0.135f;
	MaterialAmbient[1] = 0.2225f;
	MaterialAmbient[2] = 0.1575f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.54f;
	MaterialDiffuse[1] = 0.89f;
	MaterialDiffuse[2] = 0.63f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.316228f;
	MaterialSpecular[1] = 0.316228f;
	MaterialSpecular[2] = 0.316228f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.1f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 11.5f, 0.0f);
	gluSphere(quadric[1], 1.0f, 30, 30);

	//-------------------------------------Obsidian-------------------------------------//
	MaterialAmbient[0] = 0.05375f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.06625f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.18275f;
	MaterialDiffuse[1] = 0.17f;
	MaterialDiffuse[2] = 0.22525f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.332741f;
	MaterialSpecular[1] = 0.328634f;
	MaterialSpecular[2] = 0.346435f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.3f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 9.0f, 0.0f);
	gluSphere(quadric[2], 1.0f, 30, 30);

	//-------------------------------------Pearl-------------------------------------//
	MaterialAmbient[0] = 0.25f;
	MaterialAmbient[1] = 0.20725f;
	MaterialAmbient[2] = 0.20725f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 1.0f;
	MaterialDiffuse[1] = 0.829f;
	MaterialDiffuse[2] = 0.829f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.296648f;
	MaterialSpecular[1] = 0.296648f;
	MaterialSpecular[2] = 0.296648f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.088f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 6.5f, 0.0f);
	gluSphere(quadric[3], 1.0f, 30, 30);

	//-------------------------------------Ruby-------------------------------------//
	MaterialAmbient[0] = 0.1745f;
	MaterialAmbient[1] = 0.01175f;
	MaterialAmbient[2] = 0.01175f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.61424f;
	MaterialDiffuse[1] = 0.04136f;
	MaterialDiffuse[2] = 0.04136f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.727811f;
	MaterialSpecular[1] = 0.626959f;
	MaterialSpecular[2] = 0.626959f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.6f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 4.0f, 0.0f);
	gluSphere(quadric[4], 1.0f, 30, 30);

	//-------------------------------------Turquoise-------------------------------------//
	MaterialAmbient[0] = 0.1f;
	MaterialAmbient[1] = 0.18725f;
	MaterialAmbient[2] = 0.1745f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.396f;
	MaterialDiffuse[1] = 0.74151f;
	MaterialDiffuse[2] = 0.69102f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.297254f;
	MaterialSpecular[1] = 0.30829f;
	MaterialSpecular[2] = 0.306678f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.1f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 1.5f, 0.0f);
	gluSphere(quadric[5], 1.0f, 30, 30);

	//-------------------------------------Brass-------------------------------------//
	MaterialAmbient[0] = 0.329412f;
	MaterialAmbient[1] = 0.223529f;
	MaterialAmbient[2] = 0.027451f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.782392f;
	MaterialDiffuse[1] = 0.568627f;
	MaterialDiffuse[2] = 0.113725f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.992157f;
	MaterialSpecular[1] = 0.941176f;
	MaterialSpecular[2] = 0.807843f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.21794872f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 14.0f, 0.0f);
	gluSphere(quadric[6], 1.0f, 30, 30);

	//-------------------------------------Bronze-------------------------------------//
	MaterialAmbient[0] = 0.2125f;
	MaterialAmbient[1] = 0.1275f;
	MaterialAmbient[2] = 0.054f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.714f;
	MaterialDiffuse[1] = 0.4284f;
	MaterialDiffuse[2] = 0.18144f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.393548f;
	MaterialSpecular[1] = 0.271906f;
	MaterialSpecular[2] = 0.166721f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.2f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 11.5f, 0.0f);
	gluSphere(quadric[7], 1.0f, 30, 30);

	//-------------------------------------Chrome-------------------------------------//
	MaterialAmbient[0] = 0.25f;
	MaterialAmbient[1] = 0.25f;
	MaterialAmbient[2] = 0.25f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.4f;
	MaterialDiffuse[1] = 0.4f;
	MaterialDiffuse[2] = 0.4f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.774597f;
	MaterialSpecular[1] = 0.774597f;
	MaterialSpecular[2] = 0.774597f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.6f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 9.0f, 0.0f);
	gluSphere(quadric[8], 1.0f, 30, 30);

	//-------------------------------------Copper-------------------------------------//
	MaterialAmbient[0] = 0.19125f;
	MaterialAmbient[1] = 0.0735f;
	MaterialAmbient[2] = 0.0225f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.7038f;
	MaterialDiffuse[1] = 0.27048f;
	MaterialDiffuse[2] = 0.0828f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.256777f;
	MaterialSpecular[1] = 0.137622f;
	MaterialSpecular[2] = 0.086014f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.1f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 6.5f, 0.0f);
	gluSphere(quadric[9], 1.0f, 30, 30);

	//-------------------------------------Gold-------------------------------------//
	MaterialAmbient[0] = 0.24725f;
	MaterialAmbient[1] = 0.1995f;
	MaterialAmbient[2] = 0.0745f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.75164f;
	MaterialDiffuse[1] = 0.60648f;
	MaterialDiffuse[2] = 0.22648f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.628281f;
	MaterialSpecular[1] = 0.555802f;
	MaterialSpecular[2] = 0.366065f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.4f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 4.0f, 0.0f);
	gluSphere(quadric[10], 1.0f, 30, 30);


	//-------------------------------------Silver-------------------------------------//
	MaterialAmbient[0] = 0.19225f;
	MaterialAmbient[1] = 0.19225f;
	MaterialAmbient[2] = 0.19225f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.50754f;
	MaterialDiffuse[1] = 0.50754f;
	MaterialDiffuse[2] = 0.50754f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.508273f;
	MaterialSpecular[1] = 0.508273f;
	MaterialSpecular[2] = 0.508273f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.4f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 1.5f, 0.0f);
	gluSphere(quadric[11], 1.0f, 30, 30);

	//-------------------------------------Black Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.01f;
	MaterialDiffuse[1] = 0.01f;
	MaterialDiffuse[2] = 0.01f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.50f;
	MaterialSpecular[1] = 0.50f;
	MaterialSpecular[2] = 0.50f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 14.0f, 0.0f);
	gluSphere(quadric[12], 1.0f, 30, 30);

	//-------------------------------------Cyan Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.1f;
	MaterialAmbient[2] = 0.06f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.01f;
	MaterialDiffuse[1] = 0.50980392f;
	MaterialDiffuse[2] = 0.50980392f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.50196078f;
	MaterialSpecular[1] = 0.50196078f;
	MaterialSpecular[2] = 0.50196078f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 11.5f, 0.0f);
	gluSphere(quadric[13], 1.0f, 30, 30);

	//-------------------------------------Green Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.1f;
	MaterialDiffuse[1] = 0.35f;
	MaterialDiffuse[2] = 0.1f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.45f;
	MaterialSpecular[1] = 0.55f;
	MaterialSpecular[2] = 0.45f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 9.0f, 0.0f);
	gluSphere(quadric[14], 1.0f, 30, 30);

	//-------------------------------------Red Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.0f;
	MaterialDiffuse[2] = 0.0f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.7f;
	MaterialSpecular[1] = 0.6f;
	MaterialSpecular[2] = 0.6f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 6.5f, 0.0f);
	gluSphere(quadric[15], 1.0f, 30, 30);

	//-------------------------------------White Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.55f;
	MaterialDiffuse[1] = 0.55f;
	MaterialDiffuse[2] = 0.55f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.70f;
	MaterialSpecular[1] = 0.70f;
	MaterialSpecular[2] = 0.70f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 4.0f, 0.0f);
	gluSphere(quadric[16], 1.0f, 30, 30);

	//-------------------------------------White Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.0f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.60f;
	MaterialSpecular[1] = 0.60f;
	MaterialSpecular[2] = 0.50f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 1.5f, 0.0f);
	gluSphere(quadric[17], 1.0f, 30, 30);

	//-------------------------------------Black Rubber-------------------------------------//
	MaterialAmbient[0] = 0.02f;
	MaterialAmbient[1] = 0.02f;
	MaterialAmbient[2] = 0.02f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.01f;
	MaterialDiffuse[1] = 0.01f;
	MaterialDiffuse[2] = 0.01f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.4f;
	MaterialSpecular[1] = 0.4f;
	MaterialSpecular[2] = 0.4f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 14.0f, 0.0f);
	gluSphere(quadric[18], 1.0f, 30, 30);

	//-------------------------------------Cyan Rubber-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.05f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.4f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.5f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.04f;
	MaterialSpecular[1] = 0.7f;
	MaterialSpecular[2] = 0.7f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 11.5f, 0.0f);
	gluSphere(quadric[19], 1.0f, 30, 30);

	//-------------------------------------Green Rubber-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.4f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.4f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.04f;
	MaterialSpecular[1] = 0.7f;
	MaterialSpecular[2] = 0.04f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 9.0f, 0.0f);
	gluSphere(quadric[20], 1.0f, 30, 30);

	//-------------------------------------Red Rubber-------------------------------------//
	MaterialAmbient[0] = 0.05f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.4f;
	MaterialDiffuse[2] = 0.4f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.7f;
	MaterialSpecular[1] = 0.04f;
	MaterialSpecular[2] = 0.04f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 6.5f, 0.0f);
	gluSphere(quadric[21], 1.0f, 30, 30);

	//-------------------------------------White Rubber-------------------------------------//
	MaterialAmbient[0] = 0.05f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.05f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.5f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.7f;
	MaterialSpecular[1] = 0.7f;
	MaterialSpecular[2] = 0.7f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 4.0f, 0.0f);
	gluSphere(quadric[22], 1.0f, 30, 30);

	//-------------------------------------Yellow Rubber-------------------------------------//
	MaterialAmbient[0] = 0.05f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.4f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.7f;
	MaterialSpecular[1] = 0.7f;
	MaterialSpecular[2] = 0.04f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 1.5f, 0.0f);
	gluSphere(quadric[23], 1.0f, 30, 30);
}
