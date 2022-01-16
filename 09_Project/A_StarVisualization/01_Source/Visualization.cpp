//Standard Header
#include<Windows.h>
#include<stdio.h>
#define _USE_MATH_DEFINES 1
#include<math.h>
#include<gl/GL.h>
#include<gl/glu.h>
#include <time.h>

//My Headers
#include"Graph.h"
#include"Stack.h"
#include"ServerList.h"
#include"Visualization.h"

//Libraries
#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "glu32.lib")
#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "kernel32.lib")
#pragma comment (lib, "Winmm.lib")

//Macros
#define WIN_WIDTH			1366
#define WIN_HEIGHT			768
#define X					(GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH)/2
#define Y					(GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT)/2

//Funtion declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//Global Variable
HWND ghwnd;
DWORD dwStyle;
HINSTANCE hInstance;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
FILE *gpFile = NULL;
Graph_t *graph = NULL;
GLYPHMETRICSFLOAT *gmf;
stack_t *openList = NULL;
ap_list *nodeList = NULL;
ap_list *start_nodeList = NULL;
stack_t *sortedList = NULL;
bool bIsFullScreen = false;
bool gbActiveWindow = false;
unsigned int base;
int width = WIN_WIDTH;
int height = WIN_HEIGHT;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

//Light Variables
bool bLight = false;

GLfloat LightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat LightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat LightPosition[] = { 0.0f, 0.0f, 0.0f, 1.0f };

GLfloat light_model_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
GLfloat light_model_local_viewer[] = { 0.0f };

GLfloat MaterialDiffuse_white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat MaterialDiffuse_grey[] = { 0.5019607f, 0.5019607f, 0.5019607f, 1.0f };
GLfloat MaterialDiffuse_black[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat MaterialDiffuse_pink[] = { 1.0f, 0.0f, 0.8f, 0.0f };
GLfloat MaterialDiffuse_red[] = { 1.0f, 0.0f, 0.0f, 1.0f };
GLfloat MaterialDiffuse_green[] = { 0.0f, 1.0f, 0.0f, 1.0f };
GLfloat MaterialDiffuse_orange[] = { 1.0f, 0.6470588f, 0.0f, 1.0f };
GLfloat MaterialDiffuse_yellow[] = { 1.0f, 1.0f, 0.0f, 1.0f };
GLfloat MaterialDiffuse_purple_low[] = { 0.9f, 0.4f, 1.0f, 1.0f };
GLfloat MaterialDiffuse_purple_high[] = { 0.9f, 0.2f, 0.7f, 1.0f };
GLfloat MaterialDiffuse_purple_ultrahigh[] = { 0.9f, 0.0f, 0.9f, 1.0f };

GLfloat MaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat MaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat MaterialShininess[] = { 50.0f };

GLUquadric *quadric = NULL;
GLUquadric *quadric_cylinder = NULL;

char str_Algo[] = { "A* Algorithm's Visualization" };
char str_Statement[] = { "Algorithm Statement : " };
char str_Statement_1[] = { "The Algorithm Will Find Out The Shortest Path Amongs All Available Paths." };

char str_credits[] = { "Done By : " };
char str_credits_1[] = { "Atharva A. Pore" };
char str_credits_2[] = { "P I X E L Group" };

char str_inst_1[] = { "Press Right Arrow Key For Algorithm's Steps." };
char str_inst_2[] = { "Press Space Bar For Algorithm's Visualization." };
char str_inst_3[] = { "Press 'N' or 'n' To Add The Nodes." };
char str_inst_4[] = { "Press 'S' or 's' To Select Start And End." };
char str_inst_5[] = { "Press 'P' or 'p' To Start Finding A Path." };


bool bFlashScreen = true;
bool bFlash = true;
bool bRightArrow = false;

GLfloat angle = 0.0f;

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	//Variable Declaration
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

	//code
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
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
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("Template Window"),
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
			//Here Call Display Though For This App We Are Calling In WM_PAINT
		}
	}

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	//Function Declaration
	void ToggleFullScreen(void);
	void OGLResize(int, int);
	void OGLUninitialise(void);

	//variable declaration
	Ret_t ret;
	//Weight_t w;
	//Ret_t ret_e;


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
		case VK_SPACE:
			b_n = true;
			bRightArrow = false;
			bFlashScreen = false;
			if (bLight == false)
			{
				PlaySound(MAKEINTRESOURCE(MYAUDIO_1), hInstance, SND_ASYNC | SND_RESOURCE | SND_NODEFAULT);

				bLight = true;

				glEnable(GL_LIGHTING);
			}
			
			break;
		case VK_RIGHT:
			bRightArrow = true;
			break;
		}
		break;

	case WM_CHAR:
		switch (wParam)
		{
		
		case 'N':
		case 'n':
			
			if (count != 8)
			{
				PlaySound(MAKEINTRESOURCE(MYAUDIO), hInstance, SND_ASYNC | SND_RESOURCE | SND_NODEFAULT);
				count = count + 1;
				bVertexAdded = true;
				show_node_counter = 2.0f;
				if (count == 8)
				{
					b_s = true;
					bEdgeAdded = true;
				}
			}

			break;
		case 'E':
		case 'e':
			if (count == 8)
			{
				
				PrintGraph(graph);

			}
			break;

		case 'S':
		case 's':
			b_p = true;
			b_s = false;
			bSPressed = true;

			ret = find_shortest_path(A, D);

			if (ret == SUCCESS)
			{
				fprintf_s(gpFile, "*****SUCCESS*******\n");
				Path_Display(sortedList);
			}
			else
			{
				fprintf_s(gpFile, "FAILURE\n");
			}
			break;
		case 'P':
		case 'p':
			b_p = false;
			bNode = true;
			nodeList_run = nodeList->next;
			start_list_run = start_nodeList->next;
			PrintNode(nodeList);
			PrintNode(start_nodeList);
			break;
		case 'A':
		case 'a':
			bVisual = true;
			break;
		}
		break;


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
	//Variable Declaration
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;


	//Function Declaration
	void OGLResize(int, int);
	void ToggleFullScreen();

	//code
	memset((void *)&pfd, NULL, sizeof(PIXELFORMATDESCRIPTOR));

	//PIXELFORMATDISCRIPTER Initialization
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
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

	//Graph
	graph = CreateGraph();
	ret_v = add_vertex_in_graph();
	ret_v = add_vertex_in_graph();
	if (ret_v == SUCCESS)
	{
		ret_e = add_edge_between_vertices();
	}

	//3D
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//Lights
	glEnable(GL_AUTO_NORMAL);
	glEnable(GL_NORMALIZE);

	glLightfv(GL_LIGHT0, GL_AMBIENT, LightAmbient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_model_ambient);
	glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, light_model_local_viewer);

	glEnable(GL_LIGHT0);
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	//blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//create fonts
	font_init();

	//fullscreen
	ToggleFullScreen();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	OGLResize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void ToggleFullScreen()
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

unsigned int Create_Font(char *fontName, int fontSize, float depth)
{
	//variable declaration
	HFONT hFont;

	//code
	base = glGenLists(256);

	if (strcmp(fontName, "symbol") == 0)
	{
		hFont = CreateFont(fontSize,
			0,
			0,
			0,
			FW_BOLD,
			FALSE,
			FALSE,
			FALSE,
			SYMBOL_CHARSET,
			OUT_TT_PRECIS,
			CLIP_DEFAULT_PRECIS,
			CLEARTYPE_QUALITY,
			VARIABLE_PITCH,
			TEXT("Consolas"));
	}
	else
	{
		hFont = CreateFont(fontSize,
			0,
			0,
			0,
			FW_BOLD,
			FALSE,
			FALSE,
			FALSE,
			ANSI_CHARSET,
			OUT_TT_PRECIS,
			CLIP_DEFAULT_PRECIS,
			CLEARTYPE_QUALITY,
			VARIABLE_PITCH,
			TEXT("Consolas"));
	}
	if (!hFont)
	{
		return(-1);
	}

	SelectObject(ghdc, hFont);
	//wglUseFontOutlines(ghdc, 0, 255, 1000, 0.0f, 0.1f, WGL_FONT_POLYGONS, &gmf);
	wglUseFontOutlines(ghdc, 0, 255, base, 0.0f, depth, WGL_FONT_POLYGONS, gmf);

	return(base);
}

void RenderFont(GLfloat x_position, GLfloat y_position, GLfloat z_position, char *str)
{
	//code
	if ((base == 0) || (!str))
	{
		return;
	}

	glTranslatef(x_position, y_position, z_position);

	glPushAttrib(GL_LIST_BIT);

	glListBase(base);
	glCallLists((int)strlen(str), GL_UNSIGNED_BYTE, str);

	glPopAttrib();
}

void oglRenderFont(GLfloat x_position, GLfloat y_position, char *str, GLfloat angle)
{
	//code
	if ((base == 0) || (!str))
	{
		return;
	}

	glTranslatef(x_position, y_position, TRANSLATE_Z);
	glRotatef(angle, 0.0f, 0.0f, 1.0f);

	glPushAttrib(GL_LIST_BIT);

	glListBase(base);
	glCallLists((int)strlen(str), GL_UNSIGNED_BYTE, str);

	glPopAttrib();
}

void font_init(void)
{
	Create_Font(str_A, 3, 0.5f);
	Create_Font(str_B, 3, 0.5f);
	Create_Font(str_C, 3, 0.5f);
	Create_Font(str_D, 3, 0.5f);
	Create_Font(str_E, 3, 0.5f);
	Create_Font(str_F, 3, 0.5f);

	Create_Font(str_AtoB, 3, 0.1f);
	Create_Font(str_AtoE, 3, 0.1f);
	Create_Font(str_AtoF, 3, 0.1f);
	Create_Font(str_BtoA, 3, 0.1f);
	Create_Font(str_BtoC, 3, 0.1f);
	Create_Font(str_CtoB, 3, 0.1f);
	Create_Font(str_CtoD, 3, 0.1f);
	Create_Font(str_CtoE, 3, 0.1f);
	Create_Font(str_EtoA, 3, 0.1f);
	Create_Font(str_EtoC, 3, 0.1f);
	Create_Font(str_EtoD, 3, 0.1f);
	Create_Font(str_EtoF, 3, 0.1f);
	Create_Font(str_FtoA, 3, 0.1f);
	Create_Font(str_FtoD, 3, 0.1f);
	Create_Font(str_FtoE, 3, 0.1f);

	Create_Font(str_1, 3, 0.1f);
	Create_Font(str_2, 3, 0.1f);
	Create_Font(str_3, 3, 0.1f);
	Create_Font(str_5, 3, 0.1f);

	Create_Font(str_Algo, 15, 0.1f);
	Create_Font(str_inst_1, 5, 0.1f);
	Create_Font(str_inst_2, 5, 0.1f);
	Create_Font(str_inst_3, 5, 0.1f);
	Create_Font(str_inst_4, 5, 0.1f);
	Create_Font(str_inst_5, 5, 0.1f);
	Create_Font(str_Statement, 5, 0.1f);
	Create_Font(str_Statement_1, 5, 0.1f);
	Create_Font(str_credits, 5, 0.1f);
	Create_Font(str_credits_2, 5, 0.1f);
	Create_Font(str_credits_1, 5, 0.1f);

	//algo
	Create_Font(str_Algo_steps, 5, 0.1f);
	Create_Font(str_Algo_steps_1, 5, 0.1f);
	Create_Font(str_Algo_steps_2, 5, 0.1f);
	Create_Font(str_Algo_steps_3, 5, 0.1f);
	Create_Font(str_Algo_steps_3_1, 5, 0.1f);
	Create_Font(str_Algo_steps_4, 5, 0.1f);
	Create_Font(str_Algo_steps_5, 5, 0.1f);
	Create_Font(str_Algo_steps_5_1, 5, 0.1f);
	Create_Font(str_Algo_steps_5_2, 5, 0.1f);
	Create_Font(str_Algo_steps_5_3, 5, 0.1f);
	Create_Font(str_Algo_steps_5_4, 5, 0.1f);
	Create_Font(str_Algo_steps_5_5, 5, 0.1f);
	Create_Font(str_Algo_steps_5_6, 5, 0.1f);
	Create_Font(str_Algo_steps_5_7, 5, 0.1f);
	Create_Font(str_Algo_steps_6, 5, 0.1f);
	Create_Font(str_Algo_steps_7, 5, 0.1f);
	Create_Font(str_Algo_steps_8, 5, 0.1f);
	Create_Font(str_Algo_steps_8_1, 5, 0.1f);
	Create_Font(str_Algo_steps_8_2, 5, 0.1f);
	Create_Font(str_Algo_steps_8_3, 5, 0.1f);
	Create_Font(str_Algo_steps_8_4, 5, 0.1f);
	Create_Font(str_Algo_steps_8_5, 5, 0.1f);
	Create_Font(str_Algo_steps_9, 5, 0.1f);
	Create_Font(str_Algo_steps_9_1, 5, 0.1f);
	Create_Font(str_Algo_steps_9_2, 5, 0.1f);
	Create_Font(str_Algo_steps_9_3, 5, 0.1f);
	Create_Font(str_Algo_steps_9_4, 5, 0.1f);
	Create_Font(str_Algo_steps_10, 5, 0.1f);

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
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glViewport(0, ((GLsizei)height) / 2, ((GLsizei)width), ((GLsizei)height) / 2);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (bFlashScreen)
	{
		glDisable(GL_LIGHT0);
		
		if (bRightArrow == false)
		{
			glColor3f(0.5f, 0.8f, 0.9f);
			RenderFont(-6.0f, 7.0f, -20.0f, str_Algo);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 0.0f, 0.0f);
			RenderFont(-16.0f, 7.0f, -35.0f, str_Statement);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-16.0f, 6.0f, -35.0f, str_Statement_1);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 0.0f, 0.0f);
			RenderFont(-16.0f, -6.0f, -35.0f, str_credits);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-16.0f, -7.0f, -35.0f, str_credits_1);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-16.0f, -8.0f, -35.0f, str_credits_2);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-8.0f, -16.0f, -40.0f, str_inst_1);
		}
		
		if (bRightArrow)
		{
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 0.0f, 0.0f);
			RenderFont(-22.0f, 15.5f, -40.0f, str_Algo_steps);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-22.0f, 14.5f, -40.0f, str_Algo_steps_1);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-22.0f, 13.5f, -40.0f, str_Algo_steps_2);
			
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-22.0f, 12.5f, -40.0f, str_Algo_steps_3);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, 11.5f, -40.0f, str_Algo_steps_3_1);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-22.0f, 10.5f, -40.0f, str_Algo_steps_4);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-22.0f, 9.5f, -40.0f, str_Algo_steps_5);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 1.0f);
			RenderFont(-20.0f, 8.5f, -40.0f, str_Algo_steps_5_1);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, 7.5f, -40.0f, str_Algo_steps_5_2);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, 6.5f, -40.0f, str_Algo_steps_5_3);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-18.0f, 5.5f, -40.0f, str_Algo_steps_5_4);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, 4.5f, -40.0f, str_Algo_steps_5_5);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-20.0f, 3.5f, -40.0f, str_Algo_steps_5_6);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 1.0f);
			RenderFont(-20.0f, 2.5f, -40.0f, str_Algo_steps_5_7);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-22.0f, 1.5f, -40.0f, str_Algo_steps_6);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 0.0f);
			RenderFont(-22.0f, 0.5f, -40.0f, str_Algo_steps_7);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-22.0f, -0.5f, -40.0f, str_Algo_steps_8);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, -1.5f, -40.0f, str_Algo_steps_8_1);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-18.0f, -2.5f, -40.0f, str_Algo_steps_8_2);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-18.0f, -3.5f, -40.0f, str_Algo_steps_8_3);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, -4.5f, -40.0f, str_Algo_steps_8_4);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, -5.5f, -40.0f, str_Algo_steps_8_5);
			
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-22.0f, -6.5f, -40.0f, str_Algo_steps_9);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, -7.5f, -40.0f, str_Algo_steps_9_1);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, -8.5f, -40.0f, str_Algo_steps_9_2);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-18.0f, -9.5f, -40.0f, str_Algo_steps_9_3);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 1.0f, 0.0f);
			RenderFont(-20.0f, -10.5f, -40.0f, str_Algo_steps_9_4);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 0.0f, 1.0f);
			RenderFont(-22.0f, -11.5f, -40.0f, str_Algo_steps_10);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 0.0f, 0.0f);
			RenderFont(-10.0f, -16.0f, -40.0f, str_inst_2);
			
			
		}

	}
	else
	{
		
		glEnable(GL_LIGHT0);
		if (b_n)
		{
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 0.0f, 0.0f);
			RenderFont(-30.0f, -20.0f, -50.0f, str_inst_3);
		}
		if (b_p)
		{
			b_n = false;
			

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 0.0f, 0.0f);
			RenderFont(-30.0f, -20.0f, -50.0f, str_inst_5);
		}
		if (b_s)
		{
			b_n = false;
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 0.0f, 0.0f);
			RenderFont(-30.0f, -20.0f, -50.0f, str_inst_4);
		}
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		topViewPort();
	}
	SwapBuffers(ghdc);

}

void OGLUpdate(void)
{
	if (alpha != 1.0f && bVertexAdded == true)
	{
		alpha = alpha + 0.05f;
	}

	if (bEdgeAdded == true)
	{

		Red = 1.0f;
		Green = 1.0f;
		Blue = 1.0f;

	}

	if (bNode)
	{
		if (node_counter != 2.0f)
		{
			node_counter = node_counter + 0.001f;
		}
		if (show_node_counter != 2.0f)
		{
			show_node_counter = show_node_counter + 0.001f;
		}
	}

	if (count_node_B == 2 && count_node_C == 2 && count_node_E == 4 && count_node_F == 3 && count_node_G == 2 && count_node_H == 2)
	{
		if (path_counter != 2.0f)
		{
			path_counter = path_counter + 0.001f;
		}
	}

	if (AtoB == true || AtoE == true || AtoF == true)
	{
		if (string_counter != 2.0f)
		{
			string_counter = string_counter + 0.004f;
		}
	}

	if (bFlashScreen)
	{
		angle = angle + 0.05f;
		if (angle >= 360.0f)
		{
			angle = 0.0f;
		}
	}

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

	if (base != 0)
	{
		glDeleteLists(base, 255);
	}

}

//																	1.	basic mandetory functions
Ret_t add_vertex_in_graph(void)
{
	//variable
	Ret_t ret;

	//code
	for (int i = 0; i <= 5; i++)
	{
		ret = AddVertex(graph, x, y);
		if (ret == SUCCESS)
		{
			//fprintf_s(gpFile, "vertex added\n");
		}
	}
	return(ret);
}

Ret_t add_edge_between_vertices(void)
{
	//variable
	int i;
	Ret_t ret;
	Vertex_t start = 0;
	Vertex_t end = 0;
	Weight_t w = 0;

	//code
	for (i = 0; i < (sizeof(edge) / sizeof(edge[0])); i++)
	{
		start = edge[i].v_Start;
		end = edge[i].v_End;
		w = edge[i].v_weight;
		ret = AddEdge(graph, start, end, w);
		if (ret == SUCCESS)
		{
			//count_e = count_e + 1;
		}
	}

	return(ret);
}

Ret_t find_shortest_path(Vertex_t start, Vertex_t end)
{
	//variables
	Ret_t a_star_ret;
	vNode_t *run = NULL;
	vNode_t *vStartNode = NULL;
	vNode_t *vEndNode = NULL;

	//code
	//search The incoming nodes

	vStartNode = vSearchNode(graph->pV_Head_Node, start);
	if (vStartNode == NULL)
	{
		return(INVALID_VERTEX);
	}

	vEndNode = vSearchNode(graph->pV_Head_Node, end);
	if (vEndNode == NULL)
	{
		return(INVALID_VERTEX);
	}

	if (vStartNode != NULL && vEndNode != NULL)
	{
		a_star_ret = a_star_algorithm(vStartNode, vEndNode);
	}
	return(a_star_ret);
}

Weight_t get_weight(void)
{
	//variable
	Weight_t random_weight;

	//code
	random_weight = get_integer_random_value(LOWER_LIMIT, UPPER_LIMIT, COUNT);
	return(random_weight);
}

//																		2.	drawing functions

void draw_sphere(x_t x, y_t y)
{
	//code
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(x, y, TRANSLATE_Z);
	quadric = gluNewQuadric();
	gluSphere(quadric, RADIUS, 30, 30);
}

void draw_cylinder(x_t x, y_t y, double h, float angle)
{

	glTranslatef(x, y, TRANSLATE_Z);
	glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
	glRotatef(angle, 1.0f, 0.0f, 0.0f);
	quadric_cylinder = gluNewQuadric();
	gluCylinder(quadric_cylinder, 0.1f, 0.1f, h, 30, 30);
}

void draw_edge(red_t r, green_t g, blue_t b, x_t x1, y_t y1, x_t x2, y_t y2)
{
	//variable declaration

	//code
	glTranslatef(0.0f, 0.0f, TRANSLATE_Z);
	glLineWidth(3.0f);
	glBegin(GL_LINES);

	glColor3f(r, g, b);
	glVertex2f(x1, y1);
	glVertex2f(x2, y2);

	glEnd();
}

void show_edge(void)
{

	//a-b
	glLoadIdentity();
	draw_cylinder(-8.0f, 8.0f, 11.0f, 135.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	oglRenderFont(-12.0f, 5.0f, str_2, 0.0f);

	//b-c
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(8.0f, 8.0f, 16.0f, 180.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	oglRenderFont(0.0f, 9.0f, str_1, 0.0f);

	//c-d
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(16.0f, 0.0f, 11.0f, -135.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(12.0f, 5.0f,TRANSLATE_Z, str_2);

	//d-f
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(16.0f, 0.0f, 11.0f, 135.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(12.0f, -3.2f,TRANSLATE_Z, str_1);

	//f-e
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(8.0f, -8.0f, 11.0f, -135.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(4.0f, -3.5f,TRANSLATE_Z, str_1);

	

	//e-c
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(8.0f, 8.0f, 11.0f, 135.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(5.0f, 4.0f,TRANSLATE_Z, str_1);

	//a-e
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(0.0f, 0.0f, 16.0f, 180.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(-5.0f, 0.3f,TRANSLATE_Z, str_1);

	//a-f
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(7.0f, -8.0f, 24.0f, 199.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(-5.0f, -5.5f,TRANSLATE_Z, str_3);
	//e-d
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(0.0f, 0.0f, 16.0f, -360.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(8.0f, 0.7f,TRANSLATE_Z, str_5);

	

	//a-h
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(-11.0f, -8.0f, 10.0f, 240.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(Red, Green, Blue);
	RenderFont(-13.0f, -7.0f,TRANSLATE_Z, str_2);
	
	//a-g
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(-7.0f, 4.0f, 9.0f, 155.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(Red, Green, Blue);
	RenderFont(-10.0f, 3.5f,TRANSLATE_Z, str_1);

	//g-e
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(-7.0f, 4.0f, 8.0f, 390.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(Red, Green, Blue);
	RenderFont(-2.0f, 1.5f,TRANSLATE_Z, str_2);

	//h-f
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(7.0f, -8.0f, 18.0f, 180.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(Red, Green, Blue);
	RenderFont(-0.0f, -9.0f,TRANSLATE_Z, str_2);
}

void show_nodes(count_t count)
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if (count >= 1)
	{

		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
		draw_sphere(-16.0f, 0.0f);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(-18.3f, -0.3f,TRANSLATE_Z, str_A);

	}

	if (count >= 2)
	{
		if (b_B == false)
		{
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
			draw_sphere(-8.0f, 8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(-8.3f, 9.7f,TRANSLATE_Z, str_B);
		}
	}

	if (count >= 3)
	{
		if (b_C == false)
		{
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
			draw_sphere(8.0f, 8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(7.7f, 9.7f,TRANSLATE_Z, str_C);
		}
	}

	if (count >= 4)
	{

		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
		draw_sphere(16.0f, 0.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(17.7f, -0.3f,TRANSLATE_Z, str_D);
	}

	if (count >= 5)
	{
		if (b_F == false)
		{
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
			draw_sphere(8.0f, -8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(7.7f, -10.3f,TRANSLATE_Z, str_F);
		}
	}

	if (count >= 6)
	{
		if (b_E == false)
		{
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
			draw_sphere(0.0f, 0.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(-1.0f, 1.0f,TRANSLATE_Z, str_E);
			bDrawEdge = true;
		}
	}

	if (count >= 7)
	{
		if (b_G == false)
		{
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
			draw_sphere(-7.0f, 4.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-7.3f, 5.3f,TRANSLATE_Z, str_G);
		}
	}

	if (count >= 8)
	{
		if (b_H == false)
		{
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
			draw_sphere(-11.0f, -8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-11.3f, -6.8f,TRANSLATE_Z, str_H);
		}
	}
}

//algorithm functions
Ret_t a_star_algorithm(vNode_t *start, vNode_t *end)
{
	//variable
	vNode_t *vCurrent = NULL;
	vNode_t *vPrev_Current = NULL;
	vNode_t *v_hRun_Node = NULL;

	hNode_t *hRun = NULL;
	hNode_t *hRunNext = NULL;

	Ret_t ret;
	int weight = 0;
	int local = 0;
	int global = 0;

	int top_node_global_value = 0;
	int deleted_top = 0;



	//code
	openList = create_stack();
	sortedList = create_list();
	nodeList = create_list();
	start_nodeList = create_list();

	start->local = 0;
	ret = heuristic_cost_estimate(graph, start->vertex, end->vertex, &weight);
	if (ret == SUCCESS)
	{
		//s = start->vertex;
		//e = end->vertex;

		start->global = weight;
	}
	vCurrent = start;

	ret = push(openList, vCurrent->global);
	if (ret == SUCCESS)
	{
		fprintf_s(gpFile, "First Node's Vertex Pushed Successfully\n");
	}
	else
	{
		return(FAILURE);
	}

	hRun = vCurrent->ph_Head_Node->next;
	while (hRun != vCurrent->ph_Head_Node)
	{
		hRunNext = hRun->next;

		v_hRun_Node = vSearchNode(graph->pV_Head_Node, hRun->vertex);

		ret = heuristic_cost_estimate(graph, vCurrent->vertex, hRun->vertex, &weight);

		if (ret == SUCCESS)
		{
			ret = insert_end(start_nodeList, vCurrent->vertex);
			ret = insert_end(nodeList, hRun->vertex);

			local = vCurrent->local + weight;
			if (local < v_hRun_Node->local)
			{

				v_hRun_Node->local = local;
				v_hRun_Node->parent = vCurrent;
				if (v_hRun_Node == end)
				{
					v_hRun_Node->global = local;
				}
				else
				{
					ret = heuristic_cost_estimate(graph, hRun->vertex, end->vertex, &weight);

					if (ret == SUCCESS)
					{
						//s = hRun->vertex;
						//e =end->vertex;

						global = weight + local;
						v_hRun_Node->global = global;
						if (v_hRun_Node != end && !v_hRun_Node->bVisited)
						{
							ret = push(openList, v_hRun_Node->global);
						}
					}
				}
			}
		}
		hRun = hRunNext;
		if (hRun == vCurrent->ph_Head_Node)
		{
			ret = top(openList, &top_node_global_value);
			if (ret == SUCCESS)
			{
				if (vCurrent->global == top_node_global_value)
				{
					ret = pop(openList, &deleted_top);
					if (ret == SUCCESS)
					{
						fprintf_s(gpFile, "Top Deleted SUccessfully.\n");
					}
				}
			}

			ret = sort_stack(openList);

			if (ret == SUCCESS)
			{
				ret = top(openList, &top_node_global_value);
				if (ret == SUCCESS)
				{
					vPrev_Current = vCurrent;
					vCurrent = vCurrent->next;
					while (vCurrent->global != top_node_global_value)
					{
						vCurrent = vCurrent->next;
						if (vCurrent->global == top_node_global_value)
						{
							vPrev_Current->bVisited = true;
							break;
						}
					}
				}
			}

			else
			{
				v_hRun_Node = end;

				ret = insert_beginning(sortedList, v_hRun_Node->vertex);
				while (v_hRun_Node->parent != NULL)
				{
					ret = insert_beginning(sortedList, v_hRun_Node->parent->vertex);
					if (ret == SUCCESS)
					{
						v_hRun_Node = v_hRun_Node->parent;
						continue;
					}
				}
				break;
			}

			hRun = vCurrent->ph_Head_Node->next;
			if (vCurrent == graph->pV_Head_Node)
			{
				break;
			}
			else
			{
				continue;
			}
		}
	}

	return(SUCCESS);
}

Ret_t heuristic_cost_estimate(Graph_t *g, Vertex_t start, Vertex_t end, Weight_t *weight)
{
	fprintf_s(gpFile, "\n************Entering In heuristic_cost_estimate function************\n");
	//variable 
	vNode_t *vStart = NULL;
	vNode_t *vEnd = NULL;
	vNode_t *vCurrent = NULL;

	hNode_t* hRun = NULL;
	hNode_t* hRunNext = NULL;
	hNode_t* hhRunNext = NULL;

	hNode_t* hRunCurrent = NULL;
	hNode_t *h_is_node_available = NULL;

	int prev_weight = 0;

	int condition_count = 0;
	int condition_count_1 = 0;
	int continue_bit = 0;
	//code

	//First Search Start And End Node In The Vertex List(vList) And Get Stored In To 
	//vStart And vEnd Nodes.
	vStart = vSearchNode(g->pV_Head_Node, start);
	if (vStart == NULL)
	{
		fprintf_s(gpFile, "1.\tStart Vertex Invalid\n");
		return(INVALID_VERTEX);
	}
	fprintf_s(gpFile, "1.\tStart Vertex\t:\t%d\n", vStart->vertex);

	vEnd = vSearchNode(g->pV_Head_Node, end);
	if (vEnd == NULL)
	{
		fprintf_s(gpFile, "2.\tEnd Vertex Invalid\n");
		return(INVALID_VERTEX);
	}
	fprintf_s(gpFile, "1.\tEnd Vertex\t:\t%d\n", vEnd->vertex);

	//Now Take The Horizontal List's Variable And Assign The ph_head_node Of Start Vertex To hRun
	vCurrent = vStart;
	hRun = vCurrent->ph_Head_Node->next;

	while (hRun != vCurrent->ph_Head_Node)
	{
		hRunNext = hRun->next;
		*weight = hRun->W;
		prev_weight = *weight;

		hhRunNext = hRun;
		//hhRunNext = hRunNext->next;

		if (condition_count_1 == 0)
		{
			if (hhRunNext->vertex == end)
			{
				*weight = hhRunNext->W;
				condition_count_1 = 1;
				hRun = vCurrent->ph_Head_Node;
				//break;
			}

			while (hhRunNext->vertex != end)
			{
				hhRunNext = hhRunNext->next;
				if (hhRunNext->vertex == end)
				{
					*weight = hhRunNext->W;
					condition_count_1 = 1;
					hRun = vCurrent->ph_Head_Node;
					break;
				}
				if (hhRunNext == vCurrent->ph_Head_Node)
				{
					condition_count_1 = 2;
					break;
				}
			}
		}

		if (condition_count_1 == 2)
		{
			if (vCurrent == NULL)
			{
				return(INVALID_VERTEX);
			}

			if (vCurrent == vEnd)
			{
				*weight = prev_weight;
				break;
			}

			else if (vCurrent != vEnd)
			{
				hRunCurrent = vCurrent->ph_Head_Node->next;

				while (hRunCurrent != vCurrent->ph_Head_Node)
				{
					if (hRunCurrent->vertex != vEnd->vertex)
					{
						hRunCurrent = hRunCurrent->next;
						continue;
					}
					else if (hRunCurrent->vertex == vEnd->vertex)
					{
						*weight = hRunCurrent->W + prev_weight;
						condition_count = 1;
						break;
					}
				}
			}
			if (condition_count == 1)
			{
				hRun = vCurrent->ph_Head_Node;
			}
			else
			{
				hRun = hRunNext;
				//prev_weight = prev_weight + *weight;
				vCurrent = vSearchNode(g->pV_Head_Node, hRun->vertex);

			}
		}
	}

	fprintf_s(gpFile, "************Exiting from heuristic_cost_estimate function************\n");
	return(SUCCESS);
}

Ret_t sort_stack(stack_t *stack)
{
	//variable
	ap_data *array = NULL;
	ap_len n = 0;
	ap_ret ret;
	int array_value = 0;
	ap_data poped_data = 0;
	//code
	if ((array = to_array(stack, &n)) == NULL)
	{
		return(STACK_EMPTY);
	}
	else
	{
		if ((ret = sort(array, n)) == SUCCESS)
		{
			for (int i = 0; i < n; i++)
			{
				pop(openList, &poped_data);
				array_value = array[i];
				push(openList, array_value);
			}
			return(ret);
		}
	}
	return(FAILURE);
}

void Path_Display(ap_list *pList)
{
	ap_node *run;
	fprintf_s(gpFile, "[Beginning]<->");
	run = pList->next;
	while (run != pList)
	{
		fprintf_s(gpFile, " [%d]<->", run->data);
		run = run->next;
	}
	fprintf_s(gpFile, "[End]\n");
}

void PrintNode(ap_list *pList)
{
	ap_node *run = NULL;
	fprintf_s(gpFile, "[Beginning]<->");
	run = pList->next;
	while (run != pList)
	{
		fprintf_s(gpFile, " [%d]<->", run->data);
		run = run->next;
	}
	fprintf_s(gpFile, "[End]\n");
}

//Auxilary Routines

void topViewPort(void)
{
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
	if (bEdgeAdded)
	{
		show_edge();
	}


	if (bNode)
	{
		if (node_counter >= 2.0f)
		{
			if (start_list_run != start_nodeList)
			{
				if (start_list_run->data == 0)
				{
					if (nodeList_run != nodeList)
					{
						if (nodeList_run->data == 1)
						{
							b_B = true;
							AtoB = true;
							node_counter = 0.0f;
							count_node_B = count_node_B + 1;
						}
						if (nodeList_run->data == 2)
						{
							b_C = true;

							node_counter = 0.0f;
							count_node_C = count_node_C + 1;
						}
						if (nodeList_run->data == 4)
						{
							b_E = true;
							AtoE = true;
							node_counter = 0.0f;
							count_node_E = count_node_E + 1;
						}
						if (nodeList_run->data == 5)
						{
							b_F = true;
							AtoF = true;
							node_counter = 0.0f;
							count_node_F = count_node_F + 1;
						}
						if (nodeList_run->data == 0 || nodeList_run->data == 3)
						{
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 6)
						{
							b_G = true;
							AtoG = true;
							node_counter = 0.0f;
							count_node_G = count_node_G + 1;
						}
						if (nodeList_run->data == 7)
						{
							b_H = true;
							AtoH = true;
							node_counter = 0.0f;
							count_node_H = count_node_H + 1;
						}

					}
					nodeList_run = nodeList_run->next;
				}
				if (start_list_run->data == 1)
				{
					if (nodeList_run != nodeList)
					{
						if (nodeList_run->data == 1)
						{
							b_B = true;

							node_counter = 0.0f;
							count_node_B = count_node_B + 1;
						}
						if (nodeList_run->data == 2)
						{
							b_C = true;
							BtoC = true;
							node_counter = 0.0f;
							count_node_C = count_node_C + 1;
						}
						if (nodeList_run->data == 4)
						{
							b_E = true;

							node_counter = 0.0f;
							count_node_E = count_node_E + 1;
						}
						if (nodeList_run->data == 5)
						{
							b_F = true;

							node_counter = 0.0f;
							count_node_F = count_node_F + 1;
						}
						if (nodeList_run->data == 0)
						{
							BtoA = true;
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 3)
						{
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 6)
						{
							b_G = true;
							node_counter = 0.0f;
							count_node_G = count_node_G + 1;
						}
						if (nodeList_run->data == 7)
						{
							b_H = true;
							node_counter = 0.0f;
							count_node_H = count_node_H + 1;
						}

					}
					nodeList_run = nodeList_run->next;
				}
				if (start_list_run->data == 2)
				{
					if (nodeList_run != nodeList)
					{
						if (nodeList_run->data == 1)
						{
							b_B = true;
							CtoB = true;
							node_counter = 0.0f;
							count_node_B = count_node_B + 1;
						}
						if (nodeList_run->data == 2)
						{
							b_C = true;

							node_counter = 0.0f;
							count_node_C = count_node_C + 1;
						}
						if (nodeList_run->data == 4)
						{
							b_E = true;
							CtoE = true;
							node_counter = 0.0f;
							count_node_E = count_node_E + 1;
						}
						if (nodeList_run->data == 5)
						{
							b_F = true;

							node_counter = 0.0f;
							count_node_F = count_node_F + 1;
						}
						if (nodeList_run->data == 3)
						{
							CtoD = true;
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 0)
						{
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 6)
						{
							b_G = true;
							node_counter = 0.0f;
							count_node_G = count_node_G + 1;
						}
						if (nodeList_run->data == 7)
						{
							b_H = true;
							node_counter = 0.0f;
							count_node_H = count_node_H + 1;
						}
					}
					nodeList_run = nodeList_run->next;
				}
				if (start_list_run->data == 4)
				{
					if (nodeList_run != nodeList)
					{
						if (nodeList_run->data == 1)
						{
							b_B = true;

							node_counter = 0.0f;
							count_node_B = count_node_B + 1;
						}
						if (nodeList_run->data == 2)
						{
							b_C = true;
							EtoC = true;
							node_counter = 0.0f;
							count_node_C = count_node_C + 1;
						}
						if (nodeList_run->data == 4)
						{
							b_E = true;

							node_counter = 0.0f;
							count_node_E = count_node_E + 1;
						}
						if (nodeList_run->data == 5)
						{
							b_F = true;
							EtoF = true;
							node_counter = 0.0f;
							count_node_F = count_node_F + 1;
						}
						if (nodeList_run->data == 0)
						{
							EtoA = true;
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 3)
						{
							EtoD = true;
							node_counter = 0.0f;
							show_node_counter = 0.0f;

						}
						if (nodeList_run->data == 6)
						{
							b_G = true;
							EtoG = true;
							node_counter = 0.0f;
							count_node_G = count_node_G + 1;
						}
						if (nodeList_run->data == 7)
						{
							b_H = true;

							node_counter = 0.0f;
							count_node_H = count_node_H + 1;
						}

					}
					nodeList_run = nodeList_run->next;
				}
				if (start_list_run->data == 5)
				{
					if (nodeList_run != nodeList)
					{
						if (nodeList_run->data == 0)
						{
							FtoA = true;
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}

						if (nodeList_run->data == 1)
						{
							b_B = true;

							node_counter = 0.0f;
							count_node_B = count_node_B + 1;
						}
						if (nodeList_run->data == 2)
						{
							b_C = true;

							node_counter = 0.0f;
							count_node_C = count_node_C + 1;
						}
						if (nodeList_run->data == 4)
						{
							b_E = true;
							FtoE = true;
							node_counter = 0.0f;
							count_node_E = count_node_E + 1;
						}
						if (nodeList_run->data == 5)
						{
							b_F = true;

							node_counter = 0.0f;
							count_node_F = count_node_F + 1;
						}
						if (nodeList_run->data == 3)
						{
							FtoD = true;
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 6)
						{
							b_G = true;
							node_counter = 0.0f;
							count_node_G = count_node_G + 1;
						}
						if (nodeList_run->data == 7)
						{
							b_H = true;
							FtoH = true;
							node_counter = 0.0f;
							count_node_H = count_node_H + 1;
						}


					}
					nodeList_run = nodeList_run->next;
				}
				if (start_list_run->data == 6)
				{
					if (nodeList_run != nodeList)
					{
						if (nodeList_run->data == 0)
						{
							GtoA = true;
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 1)
						{
							b_B = true;

							node_counter = 0.0f;
							count_node_B = count_node_B + 1;
							
						}
						if (nodeList_run->data == 2)
						{
							b_C = true;

							node_counter = 0.0f;
							count_node_C = count_node_C + 1;
						}
						if (nodeList_run->data == 3)
						{
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 4)
						{
							b_E = true;
							GtoE = true;
							node_counter = 0.0f;
							count_node_E = count_node_E + 1;
						}
						if (nodeList_run->data == 5)
						{
							b_F = true;

							node_counter = 0.0f;
							count_node_F = count_node_F + 1;
						}
						if (nodeList_run->data == 6)
						{
							b_G = true;

							node_counter = 0.0f;
							count_node_G = count_node_G + 1;
						}
						if (nodeList_run->data == 7)
						{
							b_H = true;

							node_counter = 0.0f;
							count_node_H = count_node_H + 1;
						}
					}
					nodeList_run = nodeList_run->next;
				}
				if (start_list_run->data == 7)
				{
					if (nodeList_run != nodeList)
					{
						if (nodeList_run->data == 0)
						{
							HtoA = true;
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 1)
						{
							b_B = true;

							node_counter = 0.0f;
							count_node_B = count_node_B + 1;
						}
						if (nodeList_run->data == 2)
						{
							b_C = true;

							node_counter = 0.0f;
							count_node_C = count_node_C + 1;
						}
						if (nodeList_run->data == 3)
						{
							node_counter = 0.0f;
							show_node_counter = 0.0f;
						}
						if (nodeList_run->data == 4)
						{
							b_E = true;

							node_counter = 0.0f;
							count_node_E = count_node_E + 1;
						}
						if (nodeList_run->data == 5)
						{
							b_F = true;
							HtoF = true;
							node_counter = 0.0f;
							count_node_F = count_node_F + 1;
						}
						if (nodeList_run->data == 6)
						{
							b_G = true;

							node_counter = 0.0f;
							count_node_G = count_node_G + 1;
						}
						if (nodeList_run->data == 7)
						{
							b_H = true;

							node_counter = 0.0f;
							count_node_H = count_node_H + 1;
						}
					}
					nodeList_run = nodeList_run->next;
				}
			}
			start_list_run = start_list_run->next;

			if (nodeList_run == nodeList)
			{
				bNode = false;
			}

		}

	}

	if (bVertexAdded == true)
	{
		show_nodes(count);
	}

	if (bSPressed)
	{
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_red);
		
		draw_sphere(-16.0f, 0.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(-16.3f, -0.3f,TRANSLATE_Z, str_A);

		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_green);
		draw_sphere(16.0f, 0.0f);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(15.7f, -0.3f,TRANSLATE_Z, str_D);
	}


	if (b_B)
	{
		if (count_node_B == 1)
		{

			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_low);
			draw_sphere(-8.0f, 8.0f);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(-8.3f, 7.7f,TRANSLATE_Z, str_B);
		}
		if (count_node_B == 2)
		{
		
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_yellow);
			draw_sphere(-8.0f, 8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(-8.3f, 7.7f,TRANSLATE_Z, str_B);
		}
	}
	if (b_C)
	{
		if (count_node_C == 1)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_low);
			draw_sphere(8.0f, 8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(7.7f, 7.7f,TRANSLATE_Z, str_C);
		}
		if (count_node_C == 2)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_yellow);
			draw_sphere(8.0f, 8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(7.7f, 7.7f,TRANSLATE_Z, str_C);
		}
	}
	if (b_E)
	{
		if (count_node_E == 1)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_low);
			draw_sphere(0.0f, 0.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(-0.3f, -0.3f,TRANSLATE_Z, str_E);
		}
		if (count_node_E == 2)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_high);
			draw_sphere(0.0f, 0.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(-0.3f, -0.3f,TRANSLATE_Z, str_E);
		}
		if (count_node_E == 3)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_ultrahigh);
			draw_sphere(0.0f, 0.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(-0.3f, -0.3f,TRANSLATE_Z, str_E);
		}
		if (count_node_E == 4)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_yellow);
			draw_sphere(0.0f, 0.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(-0.3f, -0.3f,TRANSLATE_Z, str_E);
		}

	}
	if (b_F)
	{
		if (count_node_F == 1)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_low);
			draw_sphere(8.0f, -8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(7.7f, -8.3f,TRANSLATE_Z, str_F);
		}
		if (count_node_F == 2)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_high);
			draw_sphere(8.0f, -8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(7.7f, -8.3f,TRANSLATE_Z, str_F);

		}
		if (count_node_F == 3)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_yellow);
			draw_sphere(8.0f, -8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(0.0f, 0.0f, 0.0f);
			RenderFont(7.7f, -8.3f,TRANSLATE_Z, str_F);
		}
	}

	if (b_G)
	{
		if (count_node_G == 1)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_low);
			draw_sphere(-7.0f, 4.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-7.3f, 5.3f,TRANSLATE_Z, str_G);
		}
		if (count_node_G == 2)
		{
		
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_yellow);
			draw_sphere(-7.0f, 4.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-7.3f, 5.3f,TRANSLATE_Z, str_G);
		}
	}
	if (b_H)
	{
		if (count_node_H == 1)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_purple_low);
			draw_sphere(-11.0f, -8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-11.3f, -6.8f,TRANSLATE_Z, str_H);
		}
		if (count_node_H == 2)
		{
			
			glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_yellow);
			draw_sphere(-11.0f, -8.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glColor3f(1.0f, 1.0f, 1.0f);
			RenderFont(-11.3f, -6.8f,TRANSLATE_Z, str_H);

		}
	}

	if (AtoB)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(-14.0f, 4.5f, str_AtoB, 45.0f);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(-8.3f, 9.7f,TRANSLATE_Z, str_B);

		if (AtoE)
		{
			AtoB = false;
		}
	}

	if (AtoE)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(-10.0f, 0.5f, str_AtoE, 0.0f);

		fonts();
		if (AtoF)
		{
			AtoE = false;
		}
	}
	if (AtoF)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(-9.0f, -3.5f, str_AtoF, -20.0f);

		fonts();
		if (AtoG)
		{
			AtoF = false;
		}
	}

	//atog
	if (AtoG)
	{
		glLoadIdentity();
		glColor3f(Red, Green, Blue);
		oglRenderFont(-12.0f, 0.7f, str_AtoG, 25.0f);

		fonts();
		if (AtoH)
		{
			AtoG = false;
		}
	}
	//atoh
	if (AtoH)
	{
		glLoadIdentity();
		glColor3f(Red, Green, Blue);
		oglRenderFont(-15.0f, -3.0f, str_AtoH, -60.0f);

		fonts();
		if (FtoA)
		{
			AtoH = false;
		}
	}
	//then true ftoa

	if (FtoA)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(-9.0f, -3.5f, str_FtoA, -20.0f);

		fonts();
		if (FtoD)
		{
			FtoA = false;
		}
	}
	if (FtoD)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(11.0f, -6.5f, str_FtoD, 45.0f);

		fonts();
		if (FtoE)
		{
			FtoD = false;
		}
	}
	if (FtoE)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(1.0f, -2.5f, str_FtoE, -45.0f);
		fonts();
		if (FtoH)
		{
			FtoE = false;
		}
	}
	if (FtoH)
	{
		glLoadIdentity();
		glColor3f(Red, Green, Blue);
		oglRenderFont(-5.0f, -9.0f, str_FtoH, 0.0f);

		fonts();
		if (HtoA)
		{
			FtoH = false;
		}
	}
	//htoa
	if (HtoA)
	{
		glLoadIdentity();
		glColor3f(Red, Green, Blue);
		oglRenderFont(-15.0f, -3.0f, str_HtoA, -60.0f);

		fonts();
		if (HtoF)
		{
			HtoA = false;
		}
	}
	//htof
	if (HtoF)
	{
		glLoadIdentity();
		glColor3f(Red, Green, Blue);
		oglRenderFont(-5.0f, -9.0f, str_HtoF, 0.0f);

		fonts();
		if (BtoA)
		{
			HtoF = false;
		}
	}
	//then true btoa

	if (BtoA)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(-14.0f, 4.5f, str_BtoA, 45.0f);

		fonts();
		if (BtoC)
		{
			BtoA = false;
		}
	}
	if (BtoC)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(0.0f, 7.0f, str_BtoC, 0.0f);

		fonts();
		if (CtoB)
		{
			BtoC = false;
		}
	}
	if (CtoB)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(0.0f, 7.0f, str_CtoB, 0.0f);

		fonts();
		if (CtoD)
		{
			CtoB = false;
		}
	}
	if (CtoD)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(13.0f, 4.0f, str_CtoD, -45.0f);

		fonts();
		if (CtoE)
		{
			CtoD = false;
		}
	}
	if (CtoE)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(2.5f, 3.0f, str_CtoE, 45.0f);
		fonts();
		if (EtoA)
		{
			CtoE = false;
		}
	}

	if (EtoA)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(-10.0f, 0.5f, str_EtoA, 0.0f);
		fonts();
		if (EtoC)
		{
			EtoA = false;
		}
	}
	if (EtoC)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(2.5f, 3.0f, str_EtoC, 45.0f);
		fonts();
		if (EtoD)
		{
			EtoC = false;
		}
	}
	if (EtoD)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(8.0f, -0.9f, str_EtoD, 0.0f);
		fonts();
		if (EtoF)
		{
			EtoD = false;
		}
	}
	if (EtoF)
	{
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(1.0f, -2.5f, str_EtoF, -45.0f);
		fonts();
		if (EtoG)
		{
			EtoF = false;
		}
	}

	//etog
	if (EtoG)
	{
		glLoadIdentity();
		glColor3f(Red, Green, Blue);
		oglRenderFont(-5.0f, 3.0f, str_EtoG, -30.0f);

		fonts();
		if (GtoA)
		{
			EtoG = false;
		}
	}
	//then true gtoa
	if (GtoA)
	{
		glLoadIdentity();
		glColor3f(Red, Green, Blue);
		oglRenderFont(-12.0f, 0.7f, str_GtoA, 25.0f);

		fonts();
		if (GtoE)
		{
			GtoA = false;
		}
	}
	//then true gtoe
	if (GtoE)
	{
		glLoadIdentity();
		glColor3f(Red, Green, Blue);
		oglRenderFont(-5.0f, 3.0f, str_GtoE, -30.0f);

		fonts();
		if (bVisual)
		{
			GtoE = false;
		}
	}
	//ing gto check for bVisual

	if (path_counter >= 2.0f)
	{
		bVisual = true;
	}
	if (bVisual)
	{
		
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_white);
		fonts();
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_grey);
		grey_other_nodes();
		//A
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_orange);
		draw_sphere(-16.0f, 0.0f);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(-16.3f, -0.3f,TRANSLATE_Z, str_A);

		//a-e
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_orange);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		draw_cylinder(0.0f, 0.0f, 16.0f, 180.0f);
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(-10.0f, 0.5f, str_AtoE, 0.0f);

		//E
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_orange);
		draw_sphere(0.0f, 0.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(-0.3f, -0.3f,TRANSLATE_Z, str_E);

		//e-f
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_orange);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		draw_cylinder(8.0f, -8.0f, 11.0f, -135.0f);
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(1.0f, -2.5f, str_EtoF, -45.0f);

		//F
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_orange);
		draw_sphere(8.0f, -8.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(7.7f, -8.3f,TRANSLATE_Z, str_F);

		//f-d
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_orange);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		draw_cylinder(16.0f, 0.0f, 11.0f, 135.0f);
		glLoadIdentity();
		glColor3f(1.0f, 1.0f, 1.0f);
		oglRenderFont(11.0f, -6.5f, str_FtoD, 45.0f);

		//D
		glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse_orange);
		draw_sphere(16.0f, 0.0f);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(0.0f, 0.0f, 0.0f);
		RenderFont(15.7f, -0.3f,TRANSLATE_Z, str_D);
	}

}

void grey_other_nodes(void)
{

	//a-b
	glLoadIdentity();
	draw_cylinder(-8.0f, 8.0f, 11.0f, 135.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(-12.0f, 5.0f,TRANSLATE_Z, str_2);

	//b-c
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(8.0f, 8.0f, 16.0f, 180.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(0.0f, 9.0f,TRANSLATE_Z, str_1);

	//c-d
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(16.0f, 0.0f, 11.0f, -135.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(12.0f, 5.0f,TRANSLATE_Z, str_2);

	//e-c
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(8.0f, 8.0f, 11.0f, 135.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(5.0f, 4.0f,TRANSLATE_Z, str_1);

	//a-f
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(7.0f, -8.0f, 24.0f, 199.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(-5.0f, -5.5f,TRANSLATE_Z, str_3);
	//e-d
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(0.0f, 0.0f, 16.0f, -360.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(8.0f, 0.7f,TRANSLATE_Z, str_5);



	//a-h
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(-11.0f, -8.0f, 10.0f, 240.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(Red, Green, Blue);
	RenderFont(-13.0f, -7.0f,TRANSLATE_Z, str_2);

	//a-g
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(-7.0f, 4.0f, 9.0f, 155.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(Red, Green, Blue);
	RenderFont(-10.0f, 3.5f,TRANSLATE_Z, str_1);

	//g-e
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(-7.0f, 4.0f, 8.0f, 390.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(Red, Green, Blue);
	RenderFont(-2.0f, 1.5f,TRANSLATE_Z, str_2);

	//h-f
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	draw_cylinder(7.0f, -8.0f, 18.0f, 180.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(Red, Green, Blue);
	RenderFont(-0.0f, -9.0f,TRANSLATE_Z, str_2);

	//nodes
	draw_sphere(-8.0f, 8.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(-8.3f, 9.7f,TRANSLATE_Z, str_B);
	
	draw_sphere(8.0f, 8.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(7.7f, 9.7f,TRANSLATE_Z, str_C);
	

	draw_sphere(-7.0f, 4.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(-7.3f, 5.3f,TRANSLATE_Z, str_G);
	
	draw_sphere(-11.0f, -8.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderFont(-11.3f, -6.8f,TRANSLATE_Z, str_H);


}

void fonts(void)
{
	//A
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(-18.3f, -0.3f,TRANSLATE_Z, str_A);

	//B
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(-8.3f, 9.7f,TRANSLATE_Z, str_B);

	//C
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(7.7f, 9.7f,TRANSLATE_Z, str_C);

	//D
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(17.7f, -0.3f,TRANSLATE_Z, str_D);

	//F
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(7.7f, -10.3f,TRANSLATE_Z, str_F);

	//E
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(-1.0f, 1.0f,TRANSLATE_Z, str_E);

	//G
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(-7.3f, 5.3f,TRANSLATE_Z, str_G);

	//H
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(0.0f, 0.0f, 0.0f);
	RenderFont(-11.3f, -6.8f,TRANSLATE_Z, str_H);
}



Weight_t get_integer_random_value(lower_limit_t lower, upper_limit_t upper, count_t count)
{
	//variable
	Weight_t random_weight;
	int random_function_iterator = 0;
	srand((unsigned int)time(0));
	//code
	do
	{

		random_weight = (rand() % (upper - lower + 1) + lower);
		random_function_iterator = random_function_iterator + 1;

	} while (random_function_iterator != count);

	return(random_weight);
}

void PrintGraph(Graph_t *g)
{
	//variable
	vNode_t *pv_run = NULL;
	hNode_t *ph_run = NULL;

	for (pv_run = g->pV_Head_Node->next; pv_run != g->pV_Head_Node; pv_run = pv_run->next)
	{
		fprintf_s(gpFile, "[%d]:\t\t", pv_run->vertex);
		for (ph_run = pv_run->ph_Head_Node->next; ph_run != pv_run->ph_Head_Node; ph_run = ph_run->next)
			fprintf_s(gpFile, "[%d]<->", ph_run->vertex);
		fprintf_s(gpFile, "[end]\n");
	}
}
