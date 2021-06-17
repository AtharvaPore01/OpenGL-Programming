//Headers
#include<Windows.h>
#include<stdio.h>
#define _USE_MATH_DEFINES 1
#include<math.h>
#include<gl/GL.h>
#include<gl/GLU.h>
#include"MyResources.h"

//Library Files
#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "glu32.lib")
#pragma comment (lib, "Winmm.lib")
//Macros
#define WIN_WIDTH 1280
#define WIN_HEIGHT 720

#define X (GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH)/2
#define Y (GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT)/2

//Function Decalration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//Global Variable
FILE *gpFile = NULL;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
HINSTANCE hInstance;
HWND ghwnd;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
bool gbActiveWindow = false;
bool bIsFullScreen = false;

//Variables for INDIA

//I
bool bITrue = false;

GLfloat x_I1 = -3.0f;
//A
bool bATrue = false;

GLfloat x_A = 3.0f;
GLfloat r_A = 0.0f;
GLfloat g_A = 0.0f;
GLfloat b_A = 0.0f;
GLfloat White_A = 0.0f;
//N
bool bNTrue = false;

GLfloat y_N = 3.0f;
//I2
bool bI2True = false;

GLfloat y_I2 = -3.0f;
//D
bool bDTrue = false;

GLfloat r = 0.0f;
GLfloat g = 0.0f;
GLfloat b = 0.0f;
//Plane
bool bPlaneTrue = false;
bool bTopPlaneAtOrigin = false;
//bool bStartPlane = false;
bool bClipTopPlane = false;
bool bClipBottomPlane = false;

GLfloat x_plane = -22.0f;

GLfloat top_plane_rotate_angle = -60.0f;
GLfloat bottom_plane_rotate_angle = 60.0f;

//TriColour
bool bColourDone = false;
//bool bColor = false;
bool bStartIncrementingTop = false;
bool bStartDecrementingBottom = false;

GLfloat x_CoordinateOfStrips_Orange = -12.0f;
GLfloat x_CoordinateOfStrips_White = -12.0f;
GLfloat x_CoordinateOfStrips_Green = -12.0f;

struct TopPlane
{
	GLfloat x;
	GLfloat y;
	GLfloat Radius = 10.0f;
	GLfloat angle = M_PI;
}top;

struct BottomPlane
{
	GLfloat x;
	GLfloat y;
	GLfloat Radius = 10.0f;
	GLfloat angle = M_PI;
}bottom;
struct TriColors
{
	//top
	GLfloat angle_1top = 3.14159f;
	GLfloat angle_2top = 3.14659f;

	GLfloat angle_3top = 4.71238f;
	GLfloat angle_4top = 4.71738f;
	//bottom
	GLfloat angle_1bottom = 3.13659f;
	GLfloat angle_2bottom = 3.14159f;
	
	GLfloat angle_3bottom = 1.57079f;
	GLfloat angle_4bottom = 1.56579f;
}clr;

struct RGB
{
	GLfloat r = 1.0f;
	GLfloat g = 0.5f;
	GLfloat b = 1.0f;
	GLfloat White = 1.0f;

	bool bMiddleDone = false;
	bool bTopDone = false;
	bool bBottomDone = false;

}top_clr, middle_clr, bottom_clr;


//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	//variable declaration
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MyApp");
	bool bDone = false;
	int iRet = 0;

	//function declaration
	void ToggleFullScreen(void);
	int OGLInitialise(void);
	void OGLDisplay(void);
	//File Opening
	if (fopen_s(&gpFile, "AP_Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Can't Be Created !!!"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "Log File Is Created.\n");
	}

	//code
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;

	//Register Class
	RegisterClassEx(&wndclass);

	//CreateWindow
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("aap_Dynamic India"),
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

	//Initialising...
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
		fprintf_s(gpFile, "Initialization Is Successfully Done\n");
	}
	ToggleFullScreen();
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
			if (gbActiveWindow == true)
			{
				//Here Call Update
			}
			OGLDisplay();
		}
	}
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	//function declaration
	void ToggleFullScreen(void);
	void OGLResize(int, int);
	void OGLUninitialise(void);

	//code
	switch (iMsg)
	{
	case WM_CREATE:
		PlaySound(MAKEINTRESOURCE(MYAUDIO), hInstance, SND_ASYNC | SND_RESOURCE | SND_NODEFAULT);
		break;
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
					ToggleFullScreen();
					break;
				}
				break;
		*/
	case WM_SIZE:
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
	//variable declaration
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	//Function Declaration
	void OGLResize(int, int);
	void ToggleFullScreen(void);

	//code
	memset((void *)&pfd, NULL, sizeof(PIXELFORMATDESCRIPTOR));

	//PIXELFORMATDESCRIPTOR Initialization
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cBlueBits = 8;
	pfd.cGreenBits = 8;
	pfd.cAlphaBits = 8;

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
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	OGLResize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void ToggleFullScreen(void)
{
	//Variable Declaration
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
	if (height == 0)
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
	//Function Declaration
	void OGL_I1(void);
	void OGL_N(void);
	void OGL_I2(void);
	void OGL_D(GLfloat, GLfloat, GLfloat);
	void OGL_A(void);
	void OGL_A_TriColourStrips(GLfloat, GLfloat, GLfloat, GLfloat);
	void TriColourStrips(void);
	void FighterPlane_Bottom(void);
	void FighterPlane_Middle(void);
	void FighterPlane_Top(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);

	//I1
	glTranslatef(x_I1, 0.0f, 0.0f);
	OGL_I1();
	if (x_I1 <= 0.0f)
	{
		fprintf_s(gpFile, "**************** I1 ****************\n");
		x_I1 = x_I1 + 0.00034;
		if (x_I1 > 0.0f)
		{
			bITrue = true;
		}
	}
	//A
	if (bITrue == true)
	{
		fprintf_s(gpFile, "**************** A ****************\n");
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		glTranslatef(x_A, 0.0f, 0.0f);
		OGL_A();
		if (x_A > 0.0f)
		{
			x_A = x_A - 0.00034;
			fprintf_s(gpFile, "x_A = %f\n", x_A);
			if (x_A < 0.0f)
			{
				bATrue = true;
			}
		}

	}
	//N
	if (bATrue == true)
	{
		fprintf_s(gpFile, "**************** N ****************\n");
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		glTranslatef(0.0f, y_N, 0.0f);
		OGL_N();
		if (y_N > 0.0f)
		{
			y_N = y_N - 0.000354;
			fprintf_s(gpFile, "y_N = %f\n", y_N);
			if (y_N < 0.0f)
			{
				bNTrue = true;
			}
		}

	}
	if (bNTrue == true)
	{
		fprintf_s(gpFile, "**************** I2 ****************\n");
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		glTranslatef(0.0f, y_I2, 0.0f);
		OGL_I2();
		if (y_I2 < 0.0f)
		{
			y_I2 = y_I2 + 0.000354;
			fprintf_s(gpFile, "y_I2 = %f\n", y_I2);
			if (y_I2 > 0.0f)
			{
				bI2True = true;
			}
		}
	}
	if (bI2True == true)
	{
		fprintf_s(gpFile, "**************** D ****************\n");
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		OGL_D(r, g, b);
		if ((r <= 1.0f) && (g <= 0.5f))
		{
			r = r + 0.0002f;
			fprintf_s(gpFile, "r = %f\n", r);
			g = g + 0.0001f;
			fprintf_s(gpFile, "g = %f\n", g);
			if ((r > 1.0f) && (g > 0.5f))
			{
				bDTrue = true;
			}
		}
	}

	if (bDTrue == true)
	{
		fprintf_s(gpFile, "\n\n************************Plane************************\n");

		if (bClipTopPlane == false)
		{
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(0.0f, 0.0f, -3.0f);
			glTranslatef(top.x, top.y, -20.0f);
			glRotatef(top_plane_rotate_angle, 0.0f, 0.0f, 1.0f);

			FighterPlane_Top();

		}
		
		if (top_clr.bTopDone == false)
		{
			glLoadIdentity();
			glTranslatef(0.0f, 0.0, -20.0f);
			glPointSize(5.0f);
			glBegin(GL_POINTS);
			for (GLfloat i = clr.angle_1top; i < clr.angle_2top; i = i + 0.000535f)
			{
				glColor3f(top_clr.r, top_clr.g, 0.0f);
				glVertex2f(top.Radius * cosf(i) - 7.0f, top.Radius * sinf(i) + 10.0f + 0.15);
				glColor3f(top_clr.r, top_clr.White, top_clr.b);
				glVertex2f(top.Radius * cosf(i) - 7.0f, top.Radius * sinf(i) + 10.0f);
				glColor3f(0.0f, top_clr.g, 0.0f);
				glVertex2f(top.Radius * cosf(i) - 7.0f, top.Radius * sinf(i) + 10.0f - 0.15);
			}
			glEnd();
			if (bStartIncrementingTop == true)
			{
				glBegin(GL_POINTS);
				for (GLfloat i = clr.angle_3top; i < clr.angle_4top; i = i + 0.00055f)
				{
					glColor3f(top_clr.r, top_clr.g, 0.0f);
					glVertex2f(top.Radius * cosf(i) + 6.5f, top.Radius * sinf(i) + 10.0f + 0.15);
					glColor3f(top_clr.r, top_clr.White, top_clr.b);
					glVertex2f(top.Radius * cosf(i) + 6.5f, top.Radius * sinf(i) + 10.0f);
					glColor3f(0.0f, top_clr.g, 0.0f);
					glVertex2f(top.Radius * cosf(i) + 6.5f, top.Radius * sinf(i) + 10.0f - 0.15);
				}
				glEnd();
			}
		}
		
		
		if (bClipBottomPlane == false)
		{
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(0.0f, 0.0f, -3.0f);
			glTranslatef(bottom.x, bottom.y, -20.0f);
			glRotatef(bottom_plane_rotate_angle, 0.0f, 0.0f, 1.0f);
			FighterPlane_Bottom();

		}
		if (bottom_clr.bBottomDone == false)
		{
			glLoadIdentity();
			glTranslatef(0.0f, 0.0, -20.0f);
			glPointSize(5.0f);

			glBegin(GL_POINTS);
			for (GLfloat i = clr.angle_1bottom; i > clr.angle_2bottom; i = i - 0.000235f)
			{
				glColor3f(bottom_clr.r, bottom_clr.g, 0.0f);
				glVertex2f(bottom.Radius * cosf(i) - 7.0f, bottom.Radius * sinf(i) - 10.0f + 0.15f);
				glColor3f(bottom_clr.r, bottom_clr.White, bottom_clr.b);
				glVertex2f(bottom.Radius * cosf(i) - 7.0f, bottom.Radius * sinf(i) - 10.0f);
				glColor3f(0.0f, bottom_clr.g, 0.0f);
				glVertex2f(bottom.Radius * cosf(i) - 7.0f, bottom.Radius * sinf(i) - 10.0f - 0.15f);
			}
			glEnd();
			if (bStartDecrementingBottom == true)
			{
				glBegin(GL_POINTS);
				for (GLfloat i = clr.angle_3bottom; i > clr.angle_4bottom; i = i - 0.00025f)
				{
					glColor3f(bottom_clr.r, bottom_clr.g, 0.0f);
					glVertex2f(bottom.Radius * cosf(i) + 6.5f, bottom.Radius * sinf(i) - 10.0f + 0.15f);
					glColor3f(bottom_clr.r, bottom_clr.White, bottom_clr.b);
					glVertex2f(bottom.Radius * cosf(i) + 6.5f, bottom.Radius * sinf(i) - 10.0f);
					glColor3f(0.0f, bottom_clr.g, 0.0f);
					glVertex2f(bottom.Radius * cosf(i) + 6.5f, bottom.Radius * sinf(i) - 10.0f - 0.15f);
				}
				glEnd();
			}
		}
		
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		glTranslatef(x_plane, 0.0f, -20.0f);

		FighterPlane_Middle();
		if (middle_clr.bMiddleDone == false)
		{
			glLineWidth(5.0f);
			glBegin(GL_LINES);
			glColor3f(middle_clr.r, middle_clr.g, 0.0f);
			glVertex2f(-1.20f, 0.15f);
			glVertex2f(x_CoordinateOfStrips_Orange, 0.15f);
			glColor3f(middle_clr.r, middle_clr.White, middle_clr.b);
			glVertex2f(-1.20f, 0.0f);
			glVertex2f(x_CoordinateOfStrips_Orange, 0.0f);
			glColor3f(0.0f, middle_clr.g, 0.0f);
			glVertex2f(-1.20f, -0.15f);
			glVertex2f(x_CoordinateOfStrips_Orange, -0.15f);
			glEnd();
		}
		if (x_plane <= 22.0f)
		{
			x_plane = x_plane + 0.00445f;
			x_CoordinateOfStrips_Orange = x_CoordinateOfStrips_Orange - 0.005f;
			//top
			if ((top.angle <= M_PI + M_PI_2) )
			{
				top.x = top.Radius * cosf(top.angle) - 8.0f;
				fprintf_s(gpFile, "************x = %f\n\n", top.x);
				top.y = top.Radius * sinf(top.angle) + 10.0f;
				fprintf_s(gpFile, "************y = %f\n\n", top.y);
				top.angle = top.angle + 0.0005;
				if (clr.angle_2top <= 4.71238)
				{
					clr.angle_2top = clr.angle_2top + 0.000475;
				}
				if (top.angle > M_PI + M_PI_2)
				{
					bClipTopPlane = true;
				}
				
			}
			//bottom
			if (bottom.angle >= M_PI_2)
			{
				bottom.x = bottom.Radius * cosf(bottom.angle) - 8.0f;
				bottom.y = bottom.Radius * sinf(bottom.angle) - 10.0f;
				bottom.angle = bottom.angle - 0.0005;
				if (clr.angle_2bottom >= M_PI_2)
				{
					clr.angle_2bottom = clr.angle_2bottom - 0.000475;
				}
				if (bottom.angle <= M_PI_2)
				{
					bClipBottomPlane = true;
				}
			}
			//angle
			if ((top_plane_rotate_angle <= 0.0f) && (bottom_plane_rotate_angle >= 0.0f))
			{
				top_plane_rotate_angle = top_plane_rotate_angle + 0.021;
				bottom_plane_rotate_angle = bottom_plane_rotate_angle - 0.021;
			}

			if ((x_plane > 8.0f))
			{
				bClipTopPlane = false;
				bClipBottomPlane = false;

				bStartIncrementingTop = true;
				bStartDecrementingBottom = true;
				if (top.angle <= 2 * M_PI)
				{
					top.x = top.Radius * cosf(top.angle) + 8.0f;
					top.y = top.Radius * sinf(top.angle) + 10.0f;
					top.angle = top.angle + 0.0005;
					if (clr.angle_4top <= 6.28318)
					{
						clr.angle_4top = clr.angle_4top + 0.00038;
					}
					if (top.angle > 2 * M_PI)
					{
						bClipTopPlane = true;
					}
				}
				if (bottom.angle >= 0.0f)
				{
					bottom.x = bottom.Radius * cosf(bottom.angle) + 8.0f;
					bottom.y = bottom.Radius * sinf(bottom.angle) - 10.0f;
					bottom.angle = bottom.angle - 0.0005;
					if (clr.angle_4bottom >= 0.0f)
					{
						clr.angle_4bottom = clr.angle_4bottom - 0.00038;
					}
					if (bottom.angle <= 0.0f)
					{
						bClipBottomPlane = true;
					}
				}
				top_plane_rotate_angle = top_plane_rotate_angle + 0.021;
				bottom_plane_rotate_angle = bottom_plane_rotate_angle - 0.021;
				if (x_plane > 22.0f)
				{
					bPlaneTrue = true;
				}
				
			}
		}
	}

	if (bPlaneTrue == true)
	{
		//top
		if (top_clr.bTopDone == false)
		{
			top_clr.r = top_clr.r - 0.005f;
			fprintf_s(gpFile, "top r = %f\n", top_clr.r);
			top_clr.g = top_clr.g - 0.005f;
			fprintf_s(gpFile, "top g = %f\n", top_clr.g);
			top_clr.b = top_clr.b - 0.005f;
			fprintf_s(gpFile, "top b = %f\n", top_clr.b);
			top_clr.White = top_clr.White - 0.005f;
			fprintf_s(gpFile, "top white = %f\n", top_clr.White);
		}
		//bottom
		if (bottom_clr.bBottomDone == false)
		{
			bottom_clr.r = bottom_clr.r - 0.005f;
			fprintf_s(gpFile, "bottom r = %f\n", bottom_clr.r);
			bottom_clr.g = bottom_clr.g - 0.005f;
			fprintf_s(gpFile, "bottom g = %f\n", bottom_clr.g);
			bottom_clr.b = bottom_clr.b - 0.005f;
			fprintf_s(gpFile, "bottom b = %f\n", bottom_clr.b);
			bottom_clr.White = bottom_clr.White - 0.005f;
			fprintf_s(gpFile, "bottom white = %f\n", bottom_clr.White);
		}
		//middle
		if (middle_clr.bMiddleDone == false)
		{
			
			middle_clr.r = middle_clr.r - 0.005f;
			fprintf_s(gpFile, "middle r = %f\n", middle_clr.r);
			middle_clr.g = middle_clr.g - 0.005f;
			fprintf_s(gpFile, "middle g = %f\n", middle_clr.g);
			middle_clr.b = middle_clr.b - 0.005f;
			fprintf_s(gpFile, "middle b = %f\n", middle_clr.b);
			middle_clr.White = middle_clr.White - 0.005f;
			fprintf_s(gpFile, "middle white = %f\n", middle_clr.White);
		}
		
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		OGL_A_TriColourStrips(r_A, g_A, b_A, White_A);
		if ((r_A <= 1.0f) && (g_A <= 0.5f) && (b_A <= 1.0f) && (White_A <= 1.0f))
		{
			
			r_A = r_A + 0.002f;
			fprintf_s(gpFile, "r = %f\n", r);
			g_A = g_A + 0.001f;
			fprintf_s(gpFile, "g = %f\n", g);
			b_A = b_A + 0.002f;
			fprintf_s(gpFile, "b_A = %f\n", b_A);
			White_A = White_A + 0.002f;
			fprintf_s(gpFile, "White_A = %f\n", White_A);
			if ((r > 1.0f) && (g > 0.5f) && (b > 1.0f) && (White_A > 1.0f))
			{
				bColourDone = true;
			}
			
		}
		if ((top_clr.r < 0.0f) && (top_clr.g < 0.0f) && (top_clr.b < 0.0f) && (top_clr.White < 0.0f))
		{
			fprintf_s(gpFile, "************************************************Top Flags Getting True.*************************************\n");
			top_clr.bTopDone = true;
			
		}
		if ((bottom_clr.r < 0.0f) && (bottom_clr.g < 0.0f) && (bottom_clr.b < 0.0f) && (bottom_clr.White < 0.0f))
		{
			fprintf_s(gpFile, "************************************************Bottom Flags Getting True.*************************************\n");
			bottom_clr.bBottomDone = true;
		}
		if ((middle_clr.r < 0.0f) && (middle_clr.g < 0.0f) && (middle_clr.b < 0.0f) && (middle_clr.White < 0.0f))
		{
			fprintf_s(gpFile, "************************************************Middle Flags Getting True.*************************************\n");
			middle_clr.bMiddleDone = true;
		}
		
	}
	SwapBuffers(ghdc);
}

void OGLUninitialise(void)
{
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
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);
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

	if (gpFile)
	{
		fprintf_s(gpFile, "Log File Is Closed Successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}

void OGL_I1(void)
{
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(-1.15f, 0.7f);
	glVertex2f(-1.25f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(-1.25f, -0.7f);
	glVertex2f(-1.15f, -0.7f);
	glEnd();
}
void OGL_N(void)
{
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(-0.95f, 0.7f);
	glVertex2f(-1.05f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(-1.05f, -0.7f);
	glVertex2f(-0.95f, -0.7f);
	glEnd();

	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(-0.55f, 0.7f);
	glVertex2f(-0.65f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(-0.65f, -0.7f);
	glVertex2f(-0.55f, -0.7f);
	glEnd();

	glLineWidth(15.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(-0.95f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(-0.65f, -0.7f);
	glEnd();
}
void OGL_I2(void)
{
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(0.35f, 0.7f);
	glVertex2f(0.25f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(0.25f, -0.7f);
	glVertex2f(0.35f, -0.7f);
	glEnd();
}
void OGL_D(GLfloat R, GLfloat G, GLfloat B)
{
	//top 
	glBegin(GL_QUADS);
	glColor3f(R, G, B);
	glVertex2f(0.15f, 0.7f);
	glVertex2f(-0.45f, 0.7f);
	glVertex2f(-0.45f, 0.6f);
	glVertex2f(0.15f, 0.6f);
	glEnd();
	//bottom
	glBegin(GL_QUADS);
	glColor3f(0.0f, G, 0.0f);
	glVertex2f(0.15f, -0.7f);
	glVertex2f(-0.45f, -0.7f);
	glVertex2f(-0.45f, -0.6f);
	glVertex2f(0.15f, -0.6f);
	glEnd();
	//left
	glBegin(GL_QUADS);
	glColor3f(R, G, B);
	glVertex2f(0.15f, 0.7f);
	glVertex2f(0.05f, 0.7f);
	glColor3f(0.0f, G, 0.0f);
	glVertex2f(0.05f, -0.7f);
	glVertex2f(0.15f, -0.7f);
	glEnd();
	//right
	glBegin(GL_QUADS);
	glColor3f(R, G, B);
	glVertex2f(-0.25f, 0.6f);
	glVertex2f(-0.35f, 0.6f);
	glColor3f(0.0f, G, 0.0f);
	glVertex2f(-0.35f, -0.6f);
	glVertex2f(-0.25f, -0.6f);
	glEnd();
}
void OGL_A(void)
{
	//left
	glLineWidth(60.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(0.75f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(0.45f, -0.7f);
	glEnd();
	//right
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(0.75f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(1.05f, -0.7f);
	glEnd();
}
void OGL_A_TriColourStrips(GLfloat R, GLfloat G, GLfloat B, GLfloat White)
{
	//middle strips
	glLineWidth(3.0f);
	glBegin(GL_LINES);
	glColor3f(R, G, 0.0f);
	glVertex2f(0.654f, 0.015f);
	glVertex2f(0.849f, 0.015f);
	glEnd();

	glBegin(GL_LINES);
	glColor3f(R, White, B);
	glVertex2f(0.65f, 0.0f);
	glVertex2f(0.849f, 0.0f);
	glEnd();

	glBegin(GL_LINES);
	glColor3f(0.0f, G, 0.0f);
	glVertex2f(0.648f, -0.015f);
	glVertex2f(0.85f, -0.015f);
	glEnd();

}

void FighterPlane_Top(void)
{
	//Body
	glBegin(GL_QUADS);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(-1.0f, 0.3f);
	glVertex2f(-1.0f, -0.3f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//BackEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(-0.3f, 0.0f);
	glVertex2f(-1.2f, 0.2f);
	glVertex2f(-1.2f, -0.2f);
	glEnd();

	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-1.0f, 0.15f);
	glVertex2f(-1.0f, -0.15f);
	glEnd();

	//FrontEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.8f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();


	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//Upper Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	//glColor3f(1.0f, 0.0f, 0.0f);
	glVertex2f(1.5f, 0.32f);
	glVertex2f(-0.6f, 1.5f);
	glVertex2f(-0.6f, 0.22f);
	glEnd();

	//Lower Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(1.5f, -0.32f);
	glVertex2f(-0.6f, -1.5f);
	glVertex2f(-0.6f, -0.22f);
	glEnd();
	/*------------------------------------IAF----------------------------------------*/
	//I
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-0.0f, 0.15f);
	glVertex2f(-0.0f, -0.15f);
	glEnd();

	//A
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.1f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.3f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.15f, 0.0f);
	glVertex2f(0.25f, 0.0f);
	glEnd();

	//F
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.4f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.55f, 0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.0f);
	glVertex2f(0.5f, 0.0f);
	glEnd();
}

void FighterPlane_Middle(void)
{
	//Body
	glBegin(GL_QUADS);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(-1.0f, 0.3f);
	glVertex2f(-1.0f, -0.3f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//BackEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(-0.3f, 0.0f);
	glVertex2f(-1.2f, 0.2f);
	glVertex2f(-1.2f, -0.2f);
	glEnd();

	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-1.0f, 0.15f);
	glVertex2f(-1.0f, -0.15f);
	glEnd();

	//FrontEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.8f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();


	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//Upper Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	//glColor3f(1.0f, 0.0f, 0.0f);
	glVertex2f(1.5f, 0.32f);
	glVertex2f(-0.6f, 1.5f);
	glVertex2f(-0.6f, 0.22f);
	glEnd();

	//Lower Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(1.5f, -0.32f);
	glVertex2f(-0.6f, -1.5f);
	glVertex2f(-0.6f, -0.22f);
	glEnd();
	/*------------------------------------IAF----------------------------------------*/
	//I
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-0.0f, 0.15f);
	glVertex2f(-0.0f, -0.15f);
	glEnd();

	//A
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.1f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.3f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.15f, 0.0f);
	glVertex2f(0.25f, 0.0f);
	glEnd();

	//F
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.4f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.55f, 0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.0f);
	glVertex2f(0.5f, 0.0f);
	glEnd();
}
void FighterPlane_Bottom(void)
{
	//Body
	glBegin(GL_QUADS);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(-1.0f, 0.3f);
	glVertex2f(-1.0f, -0.3f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//BackEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(-0.3f, 0.0f);
	glVertex2f(-1.2f, 0.2f);
	glVertex2f(-1.2f, -0.2f);
	glEnd();

	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-1.0f, 0.15f);
	glVertex2f(-1.0f, -0.15f);
	glEnd();

	//FrontEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.8f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();


	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//Upper Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	//glColor3f(1.0f, 0.0f, 0.0f);
	glVertex2f(1.5f, 0.32f);
	glVertex2f(-0.6f, 1.5f);
	glVertex2f(-0.6f, 0.22f);
	glEnd();

	//Lower Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(1.5f, -0.32f);
	glVertex2f(-0.6f, -1.5f);
	glVertex2f(-0.6f, -0.22f);
	glEnd();
	/*------------------------------------IAF----------------------------------------*/
	//I
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-0.0f, 0.15f);
	glVertex2f(-0.0f, -0.15f);
	glEnd();

	//A
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.1f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.3f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.15f, 0.0f);
	glVertex2f(0.25f, 0.0f);
	glEnd();

	//F
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.4f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.55f, 0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.0f);
	glVertex2f(0.5f, 0.0f);
	glEnd();
}
