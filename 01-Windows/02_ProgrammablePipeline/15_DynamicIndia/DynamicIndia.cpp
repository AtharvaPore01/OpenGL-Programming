//Standard Header Files
#include <Windows.h>
#include <stdio.h>

//OpenGL Related Header Files
#include <gl/glew.h>
#include <GL/GL.h>
#include "vmath.h"

//our header files
#include "MyResources.h"
#include "dynamicIndia.h"

//Library Function
#pragma comment (lib, "opengl32.lib")	//open-gl standard library
#pragma comment (lib, "glew32.lib")		//library of graphic library extension
#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "kernel32.lib")
#pragma comment (lib, "Winmm.lib")		//for play sound function

//Macros
#define WIN_WIDTH 1366
#define WIN_HEIGHT 768
#define X (GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH) / 2
#define Y (GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT) / 2

//enum
enum
{
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOODR_0
};

//Global Variables
HWND ghwnd;
DWORD dwStyle;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
FILE *gpFile = NULL;
bool bIsFullScreen = false;
bool gbActiveWindow = false;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

//Funtion declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//global variables related to shaders
GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

//GLuint vao;
//GLuint vbo;
GLuint vao_I;
GLuint vao_N;
GLuint vao_D;
GLuint vao_i;
GLuint vao_A;
GLuint vao_middleStrips;
GLuint vao_plane;
GLuint vao_triColorStrip;

GLuint vbo_I_position;
GLuint vbo_I_color;
GLuint vbo_N_position;
GLuint vbo_N_color;
GLuint vbo_D_position;
GLuint vbo_D_color;
GLuint vbo_i_position;
GLuint vbo_i_color;
GLuint vbo_A_position;
GLuint vbo_A_color;
GLuint vbo_middleStrips_position;
GLuint vbo_middleStrips_color;

GLuint vbo_plane_position;
GLuint vbo_plane_color;

GLuint vbo_triColorStrip_position;
GLuint vbo_triColorStrip_color;

GLuint mvpUniform;
vmath::mat4 perspectiveProjectionMatrix;

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

	//fcuntion declaration
	int oglInitialise(void);
	void oglDisplay(void);
	void oglUpdate(void);

	if (fopen_s(&gpFile, "AP_Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Can't Be Created"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	else
	{
		fprintf_s(gpFile, "Log File Created Successfully.\n");
	}

	fprintf_s(gpFile, "\nIn WinMain\n");

	//code
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.cbClsExtra = 0;
	wndclass.style = CS_VREDRAW | CS_HREDRAW | CS_OWNDC;
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hInstance = hInstance;
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	//Register class
	RegisterClassEx(&wndclass);

	//Create Window
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("DynamicIndia"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		X,
		Y,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	//Assign hwnd to ghwnd
	ghwnd = hwnd;

	//call initialise
	iRet = oglInitialise();

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

	//show window
	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	//game loop
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
			oglDisplay();
			if (gbActiveWindow == true)
			{
				oglUpdate();
			}
		}
	}

	return((int)msg.wParam);
}
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	fprintf_s(gpFile, "\n2. In WndProc\n");
	//variable declaration
	static HINSTANCE hInstance;

	//function declaration
	void oglToggleFullScreen(void);
	void oglResize(int, int);
	void oglUninitialise(void);

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

	case WM_CHAR:
/*
		switch (wParam)
		{
		case 'F':
		case 'f':
			fprintf_s(gpFile, "2.1 Going To ToggleFullScreen()\n\n");
			oglToggleFullScreen();
			break;
		}
*/
		break;


	case WM_SIZE:
		fprintf_s(gpFile, "2.2 Going To OGLResize()\n\n");
		oglResize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_ERASEBKGND:
		return(0);
		break;
	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		oglUninitialise();
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

int oglInitialise(void)
{
	//file i/o statement
	fprintf_s(gpFile, "\n In oglInitialise. \n");

	//variable declaration
	//variable declaration
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum result;
	static HWND hwnd;

	//variables related to error checking
	GLint iShaderCompileStatus = 0;
	GLint iProgramLinkStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	//function declaration
	void oglResize(int, int);
	void oglUninitialise(void);
	void oglToggleFullScreen(void);
	void oglInitData(void);

	//code
	memset((void *)&pfd, NULL, sizeof(PIXELFORMATDESCRIPTOR));

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

	result = glewInit();
	if (result != GLEW_OK)
	{
		fprintf_s(gpFile, "Error : glew Initialisation Failed.\n");
		oglUninitialise();
		DestroyWindow(hwnd);
	}

	/* Vertex Shader Code */

	//define vertex shader object
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//write vertex shader code
	const GLchar *vertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec4 vColor;"
		"out vec4 out_color;"
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_color = vColor;"
		"}";

	//specify above source code to vertex shader object
	glShaderSource(gVertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

	//compile the vertex shader
	glCompileShader(gVertexShaderObject);

	/***Steps For Error Checking***/
	/*
		1.	Call glGetShaderiv(), and get the compile status of that object.
		2.	check that compile status, if it is GL_FALSE then shader has compilation error.
		3.	if(GL_FALSE) call again the glGetShaderiv() function and get the
			infoLogLength.
		4.	if(infoLogLength > 0) then call glGetShaderInfoLog() function to get the error
			information.
		5.	Print that obtained logs in file.
	*/

	//error checking
	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetShaderInfoLog(gVertexShaderObject,
					iInfoLogLength,
					&Written,
					szInfoLog);

				fprintf_s(gpFile, "Vertex Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	/* Fragment Shader Code */

	//define fragment shader object
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//write shader code
	const GLchar *fragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"out vec4 FragColor;" \
		"in vec4 out_color;"
		"void main(void)" \
		"{" \
		"FragColor = out_color;" \
		"}";
	//specify above shader code to fragment shader object
	glShaderSource(gFragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

	//compile the shader
	glCompileShader(gFragmentShaderObject);

	//error checking
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{

			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetShaderInfoLog(gFragmentShaderObject,
					iInfoLogLength,
					&Written,
					szInfoLog);
				fprintf_s(gpFile, "Fragment Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	//create shader program object
	gShaderProgramObject = glCreateProgram();

	//Attach Vertex Shader
	glAttachShader(gShaderProgramObject, gVertexShaderObject);

	//Attach Fragment Shader
	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	//pre linking bonding to vertex attributes
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_COLOR, "vColor");

	//link the shader porgram
	glLinkProgram(gShaderProgramObject);

	//error checking

	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);

	if (iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &Written, szInfoLog);
				fprintf_s(gpFile, "Program Link Error : \n %s\n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	//post linking retriving uniform location
	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");

	//vertices declaration
	const GLfloat I_vertices[] =
	{
		-1.15f, 0.7f, 0.0f,
		-1.25f, 0.7f, 0.0f,
		-1.25f, -0.7f, 0.0f,
		-1.15f, -0.7f, 0.0f
	};

	const GLfloat N_vertices[] =
	{
		-0.95f, 0.7f, 0.0f,
		-1.05f, 0.7f, 0.0f,
		-1.05f, -0.7f, 0.0f,
		-0.95f, -0.7f, 0.0f,

		-0.55f, 0.7f, 0.0f,
		-0.65f, 0.7f, 0.0f,
		-0.65f, -0.7f, 0.0f,
		-0.55f, -0.7f, 0.0f,

		-0.95f, 0.7f, 0.0f,
		-0.95f, 0.5f, 0.0f,
		-0.65f, -0.7f, 0.0f,
		-0.65f, -0.5f, 0.0f
	};

	const GLfloat D_vertices[] =
	{
		//top
		0.15f, 0.7f, 0.0f,
		-0.45f, 0.7f, 0.0f,
		-0.45f, 0.6f, 0.0f,
		0.15f, 0.6f, 0.0f,

		//bottom
		0.15f, -0.7f, 0.0f,
		-0.45f, -0.7f, 0.0f,
		-0.45f, -0.6f, 0.0f,
		0.15f, -0.6f, 0.0f,

		//left
		0.15f, 0.7f, 0.0f,
		0.05f, 0.7f, 0.0f,
		0.05f, -0.7f, 0.0f,
		0.15f, -0.7f, 0.0f,

		//right
		-0.25f, 0.6f, 0.0f,
		-0.35f, 0.6f, 0.0f,
		-0.35f, -0.6f, 0.0f,
		-0.25f, -0.6f, 0.0f
	};

	const GLfloat i_vertices[] =
	{
		0.35f, 0.7f, 0.0f,
		0.25f, 0.7f, 0.0f,
		0.25f, -0.7f, 0.0f,
		0.35f, -0.7f, 0.0f
	};

	const GLfloat A_vertices[] =
	{
		//left
		0.75f, 0.7f, 0.0f,
		0.75f, 0.5f, 0.0f,
		0.55f, -0.7f, 0.0f,
		0.45f, -0.7f, 0.0f,
		//right
		0.75f, 0.7f, 0.0f,
		0.75f, 0.5f, 0.0f,
		0.95f, -0.7f, 0.0f,
		1.05f, -0.7f, 0.0f
	};

	const GLfloat planeVertices[] =
	{
		/* Vertices */
		//body
		2.0f, 0.35f, 0.0f,									//0
		-1.0f, 0.3f, 0.0f,									//1
		-1.0f, -0.3f, 0.0f,									//2										
		2.0f, -0.35f, 0.0f,									//3											

		//exahaust
		-0.3f, 0.0f, 0.0f,									//4
		-1.2f, 0.4f, 0.0f,									//5
		-1.2f, -0.4f, 0.0f,									//6

		//orange 
		-1.2f, 0.3f, 0.0f,
		-2.5f, 0.3f, 0.0f,
		-2.5f, 0.1f, 0.0f,
		-1.2f, 0.1f, 0.0f,

		//white
		-1.2f, 0.1f, 0.0f,
		-2.5f, 0.1f, 0.0f,
		-2.5f, -0.1f, 0.0f,
		-1.2f, -0.1f, 0.0f,

		//green
		-1.2f, -0.1f, 0.0f,
		-2.5f, -0.1f, 0.0f,
		-2.5f, -0.3f, 0.0f,
		-1.2f, -0.3f, 0.0f,

		//separator line between exhaust and body
		-1.0f, 0.15f, 0.0f,									//7
		-1.0f, -0.15f, 0.0f,								//8

		//front tip
		2.8f, 0.0f, 0.0f,									//9
		2.0f, 0.35f, 0.0f,									//10
		2.0f, -0.35f, 0.0f,									//11

		//sperator line between front tip and body
		2.0f, 0.35f, 0.0f,									//12
		2.0f, -0.35f, 0.0f,									//13

		//upper wing
		1.5f, 0.32f, 0.0f,									//14
		-0.6f, 1.5f, 0.0f,									//15
		-0.6f, 0.22f, 0.0f,									//16

		//lower wing
		1.5f, -0.32f, 0.0f,									//17
		-0.6f, -1.5f, 0.0f,									//18
		-0.6f, -0.22f, 0.0f,								//19

		//IAF Letters
		/* Vertices */
		//1. I
		-0.0f, 0.15f, 0.0f,									//20
		-0.0f, -0.15f, 0.0f,								//21

		//2. A
		0.2f, 0.15f, 0.0f,									//22
		0.1f, -0.15f, 0.0f,									//23

		0.2f, 0.15f, 0.0f,									//24
		0.3f, -0.15f, 0.0f,									//25

		0.15f, 0.0f, 0.0f,									//26
		0.25f, 0.0f, 0.0f,									//27

		//3. F
		0.4f, 0.15f, 0.0f,									//28
		0.4f, -0.15f, 0.0f,									//29

		0.4f, 0.15f, 0.0f,									//30
		0.55f, 0.15f, 0.0f,									//31

		0.4f, 0.0f, 0.0f,									//32
		0.5f, 0.0f, 0.0f									//33
	};

	const GLfloat middleStrips_vertices[] =
	{
		//orange 
		-1.2f, 0.3f, 0.0f,
		-2.5f, 0.3f, 0.0f,
		-2.5f, 0.1f, 0.0f,
		-1.2f, 0.1f, 0.0f,

		//white
		-1.2f, 0.1f, 0.0f,
		-2.5f, 0.1f, 0.0f,
		-2.5f, -0.1f, 0.0f,
		-1.2f, -0.1f, 0.0f,

		//green
		-1.2f, -0.1f, 0.0f,
		-2.5f, -0.1f, 0.0f,
		-2.5f, -0.3f, 0.0f,
		-1.2f, -0.3f, 0.0f,
	};

	//color declaration
	const GLfloat I_color[] =
	{
		1.0f, 0.5f, 0.0f,
		1.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f
	};

	const GLfloat N_color[] =
	{
		1.0f, 0.5f, 0.0f,
		1.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,

		1.0f, 0.5f, 0.0f,
		1.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,

		1.0f, 0.5f, 0.0f,
		1.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
	};

	const GLfloat i_color[] =
	{
		1.0f, 0.5f, 0.0f,
		1.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
	};

	const GLfloat A_color[] =
	{
		1.0f, 0.5f, 0.0f,
		1.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,

		1.0f, 0.5f, 0.0f,
		1.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 0.0f,
	};

	const GLfloat planeColor[]
	{
		
		/* color */
		//body
		0.7294117f, 0.8862745f, 0.9333333f,									//0
		0.7294117f, 0.8862745f, 0.9333333f,									//1
		0.7294117f, 0.8862745f, 0.9333333f,									//2										
		0.7294117f, 0.8862745f, 0.9333333f,									//3											

		//exahaust
		0.7294117f, 0.8862745f, 0.9333333f,									//4
		0.7294117f, 0.8862745f, 0.9333333f,									//5
		0.7294117f, 0.8862745f, 0.9333333f,									//6

		//orange
		1.0f, 0.5f, 0.0f,	//right top										//7
		0.0f, 0.0f, 0.0f,	//right bottom									//8
		0.0f, 0.0f, 0.0f,	//left top										//9
		1.0f, 0.5f, 0.0f,	//left top										//10

		//white
		1.0f, 1.0f, 1.0f,	//right top							//11	
		0.0f, 0.0f, 0.0f,	//right bottom		//12
		0.0f, 0.0f, 0.0f,	//left top			13
		1.0f, 1.0f, 1.0f,	//left top			14

		//green
		0.0f, 0.5f, 0.0f,	//right top			15
		0.0f, 0.0f, 0.0f,	//right bottom		16
		0.0f, 0.0f, 0.0f,	//left top			17
		0.0f, 0.5f, 0.0f,	//left top			18


		//separator line between exhaust and body
		0.0f, 0.0f, 0.0f,													//19
		0.0f, 0.0f, 0.0f,													//20

		//front tip
		0.7294117f, 0.8862745f, 0.9333333f,									//21
		0.7294117f, 0.8862745f, 0.9333333f,									//22
		0.7294117f, 0.8862745f, 0.9333333f,									//23

		//sperator line between front tip and body
		0.0f, 0.0f, 0.0f,													//24
		0.0f, 0.0f, 0.0f,													//25

		//upper wing
		0.7294117f, 0.8862745f, 0.9333333f,									//26
		0.7294117f, 0.8862745f, 0.9333333f,									//27
		0.7294117f, 0.8862745f, 0.9333333f,									//28

		//lower wing
		0.7294117f, 0.8862745f, 0.9333333f,									//29
		0.7294117f, 0.8862745f, 0.9333333f,									//30
		0.7294117f, 0.8862745f, 0.9333333f,									//31

		//IAF Letters
		/* color */
		//1. I
		0.0f, 0.0f, 0.0f,													//32
		0.0f, 0.0f, 0.0f,													//33

		//2. A
		0.0f, 0.0f, 0.0f,													//34
		0.0f, 0.0f, 0.0f,													//35

		0.0f, 0.0f, 0.0f,													//36
		0.0f, 0.0f, 0.0f,													//37

		0.0f, 0.0f, 0.0f,													//38
		0.0f, 0.0f, 0.0f,													//39

		//3. F
		0.0f, 0.0f, 0.0f,													//40
		0.0f, 0.0f, 0.0f,													//41

		0.0f, 0.0f, 0.0f,													//42
		0.0f, 0.0f, 0.0f,													//43

		0.0f, 0.0f, 0.0f,													//44
		0.0f, 0.0f, 0.0f,													//45
	};

	//create vao and vbo

	//I
	glGenVertexArrays(1, &vao_I);
	glBindVertexArray(vao_I);

	//vertices
	glGenBuffers(1, &vbo_I_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_I_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(I_vertices), I_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//color
	glGenBuffers(1, &vbo_I_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_I_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(I_color), I_color, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//N
	glGenVertexArrays(1, &vao_N);
	glBindVertexArray(vao_N);

	//vertices
	glGenBuffers(1, &vbo_N_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_N_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(N_vertices), N_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//color
	glGenBuffers(1, &vbo_N_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_N_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(N_color), N_color, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//D
	glGenVertexArrays(1, &vao_D);
	glBindVertexArray(vao_D);

	//vertices
	glGenBuffers(1, &vbo_D_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_D_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(D_vertices), D_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//color
	glGenBuffers(1, &vbo_D_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_D_color);
	glBufferData(GL_ARRAY_BUFFER, 16 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//i
	glGenVertexArrays(1, &vao_i);
	glBindVertexArray(vao_i);

	//vertices
	glGenBuffers(1, &vbo_i_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_i_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(i_vertices), i_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//color
	glGenBuffers(1, &vbo_i_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_i_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(i_color), i_color, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//A
	glGenVertexArrays(1, &vao_A);
	glBindVertexArray(vao_A);

	//vertices 
	glGenBuffers(1, &vbo_A_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_A_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(A_vertices), A_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//color
	glGenBuffers(1, &vbo_A_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_A_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(A_color), A_color, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//middleStrips
	glGenVertexArrays(1, &vao_middleStrips);
	glBindVertexArray(vao_middleStrips);

	//vertices 
	glGenBuffers(1, &vbo_middleStrips_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_middleStrips_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(middleStrips_vertices), middleStrips_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//color
	glGenBuffers(1, &vbo_middleStrips_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_middleStrips_color);
	glBufferData(GL_ARRAY_BUFFER, 12 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//plane 
	glGenVertexArrays(1, &vao_plane);
	glBindVertexArray(vao_plane);

	//vertices 
	glGenBuffers(1, &vbo_plane_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_plane_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//color
	glGenBuffers(1, &vbo_plane_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_plane_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(planeColor), planeColor, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//tri-color strip
	glGenVertexArrays(1, &vao_triColorStrip);
	glBindVertexArray(vao_triColorStrip);

	//vertices 
	glGenBuffers(1, &vbo_triColorStrip_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_position);
	glBufferData(GL_ARRAY_BUFFER, 3 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//color
	glGenBuffers(1, &vbo_triColorStrip_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_color);
	glBufferData(GL_ARRAY_BUFFER, 3 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//init dynamic india data
	oglInitData();

	//clear the window
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//make fullscreen by default
	oglToggleFullScreen();

	//depth
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	//make orthograhic projection matrix a identity matrix
	perspectiveProjectionMatrix = vmath::mat4::identity();

	//warm up resize call
	oglResize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void oglToggleFullScreen(void)
{
	fprintf_s(gpFile, " \n In ToggleFullScreen() \n");
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
void oglResize(int width, int height)
{
	fprintf_s(gpFile, "\n In OGLResize \n");
	if (height == 0)
	{
		height = 1;
	}
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = vmath::perspective(45.0f, ((GLfloat)width / (GLfloat)height), 0.1f, 100.0f);

}
void oglDisplay(void)
{
	//function declaration
	void oglDraw_I(void);
	void oglDraw_N(void);
	void oglDraw_D(void);
	void oglDraw_i(void);
	void oglDraw_A(void);
	void oglDraw_middleStrips(void);
	void oglDraw_plane(void);
	void oglDrawSmoke_top(int loop_no);
	void oglDrawSmoke_bottom(int loop_no);
	void oglDrawSmoke_middle(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderProgramObject);

	//declaration of metrices
	vmath::mat4 modelViewMatrix;
	vmath::mat4 modelViewProjectionMatrix;
	vmath::mat4 translationMatrix;
	vmath::mat4 rotationMatrix;		//for plane rotation

	//init above metrices to identity
	translationMatrix = vmath::mat4::identity();
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();

	//do necessary transformations here
	translationMatrix = vmath::translate(f_Translate_I, 0.0f, -3.0f);

	//do necessary matrix multiplication
	modelViewMatrix = modelViewMatrix * translationMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	oglDraw_I();

	if (b_I_Done)
	{
		//init above metrices to identity
		translationMatrix = vmath::mat4::identity();
		modelViewMatrix = vmath::mat4::identity();
		modelViewProjectionMatrix = vmath::mat4::identity();

		//do necessary transformations here
		translationMatrix = vmath::translate(f_Translate_A, 0.0f, -3.0f);

		//do necessary matrix multiplication
		modelViewMatrix = modelViewMatrix * translationMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		oglDraw_A();

		//init above metrices to identity
		translationMatrix = vmath::mat4::identity();
		modelViewMatrix = vmath::mat4::identity();
		modelViewProjectionMatrix = vmath::mat4::identity();

		//do necessary transformations here
		translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
		translationMatrix = vmath::translate(6.8f, 0.0f, -20.0f);

		//do necessary matrix multiplication
		modelViewMatrix = modelViewMatrix * translationMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		oglDraw_middleStrips();
	}

	if (b_A_Done)
	{
		//init above metrices to identity
		translationMatrix = vmath::mat4::identity();
		modelViewMatrix = vmath::mat4::identity();
		modelViewProjectionMatrix = vmath::mat4::identity();

		//do necessary transformations here
		translationMatrix = vmath::translate(0.0f, f_Translate_N, -3.0f);

		//do necessary matrix multiplication
		modelViewMatrix = modelViewMatrix * translationMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		oglDraw_N();
	}

	if (b_N_Done)
	{
		//init above metrices to identity
		translationMatrix = vmath::mat4::identity();
		modelViewMatrix = vmath::mat4::identity();
		modelViewProjectionMatrix = vmath::mat4::identity();

		//do necessary transformations here
		translationMatrix = vmath::translate(0.0f, f_Translate_i, -3.0f);

		//do necessary matrix multiplication
		modelViewMatrix = modelViewMatrix * translationMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		oglDraw_i();
	}
	
	if (b_i_Done)
	{
		//init above metrices to identity
		translationMatrix = vmath::mat4::identity();
		modelViewMatrix = vmath::mat4::identity();
		modelViewProjectionMatrix = vmath::mat4::identity();

		//do necessary transformations here
		translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

		//do necessary matrix multiplication
		modelViewMatrix = modelViewMatrix * translationMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		oglDraw_D();
	}

	if (b_D_Done)
	{
		/* TOP PLANE */

		//if (b_top_plane_smoke_done == false)
		//{
		//	translationMatrix = vmath::mat4::identity();
		//	//modelViewMatrix = vmath::mat4::identity();
		//	//modelViewProjectionMatrix = vmath::mat4::identity();
		//	translationMatrix = vmath::translate(0.2f, -0.45f, -20.0f);

		//	//do necessary matrix multiplication
		//	modelViewMatrix = modelViewMatrix * translationMatrix;
		//	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//	//send necessary matrics to shaders in respective uniforms
		//	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		//	oglDrawSmoke_top(1);

		//	if (b_start_incrementing == true)
		//	{
		//		oglDrawSmoke_top(2);
		//	}
		//}
		if (b_clip_top_plane == false)
		{
			//init above metrices to identity
			translationMatrix = vmath::mat4::identity();
			rotationMatrix = vmath::mat4::identity();
			modelViewMatrix = vmath::mat4::identity();
			modelViewProjectionMatrix = vmath::mat4::identity();
			
			//do necessary transformations here
			//translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
			translationMatrix = vmath::translate(top._x, top._y, -20.0f);
			rotationMatrix = vmath::rotate(top.rotation_angle, 0.0f, 0.0f, 1.0f);

			//do necessary matrix multiplication
			modelViewMatrix = modelViewMatrix * translationMatrix;
			modelViewMatrix = modelViewMatrix * rotationMatrix;
			modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

			//send necessary matrics to shaders in respective uniforms
			glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
			
			oglDraw_plane();

			//oglDrawSmoke();
		}


		//init above metrices to identity
		translationMatrix = vmath::mat4::identity();
		modelViewMatrix = vmath::mat4::identity();
		modelViewProjectionMatrix = vmath::mat4::identity();

		//do necessary transformations here
		translationMatrix = vmath::translate(0.0f, 0.0f, -20.0f);

		//do necessary matrix multiplication
		modelViewMatrix = modelViewMatrix * translationMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		//if (b_bottom_plane_smoke_done == false)
		//{
		//	translationMatrix = vmath::mat4::identity();
		//	//modelViewMatrix = vmath::mat4::identity();
		//	//modelViewProjectionMatrix = vmath::mat4::identity();
		//	translationMatrix = vmath::translate(0.0f, 0.0f, -20.0f);

		//	//do necessary matrix multiplication
		//	modelViewMatrix = modelViewMatrix * translationMatrix;
		//	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//	//send necessary matrics to shaders in respective uniforms
		//	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		//	oglDrawSmoke_bottom(1);

		//	if (b_start_decrementing == true)
		//	{
		//		oglDrawSmoke_bottom(2);
		//	}
		//}

		if (b_clip_bottom_plane == false)
		{
			//init above metrices to identity
			translationMatrix = vmath::mat4::identity();
			rotationMatrix = vmath::mat4::identity();
			modelViewMatrix = vmath::mat4::identity();
			modelViewProjectionMatrix = vmath::mat4::identity();

			//do necessary transformations here
			translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
			translationMatrix = vmath::translate(bottom._x, bottom._y, -20.0f);
			rotationMatrix = vmath::rotate(bottom.rotation_angle, 0.0f, 0.0f, 1.0f);

			//do necessary matrix multiplication
			modelViewMatrix = modelViewMatrix * translationMatrix;
			modelViewMatrix = modelViewMatrix * rotationMatrix;
			modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

			//send necessary matrics to shaders in respective uniforms
			glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

			oglDraw_plane();
		}

		//translationMatrix = vmath::mat4::identity();
		////modelViewMatrix = vmath::mat4::identity();
		////modelViewProjectionMatrix = vmath::mat4::identity();
		//translationMatrix = vmath::translate(0.0f, 0.0f, -20.0f);

		////do necessary matrix multiplication
		//modelViewMatrix = modelViewMatrix * translationMatrix;
		//modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		////send necessary matrics to shaders in respective uniforms
		//glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		//oglDrawSmoke_middle();

		//init above metrices to identity
		translationMatrix = vmath::mat4::identity();
		modelViewMatrix = vmath::mat4::identity();
		modelViewProjectionMatrix = vmath::mat4::identity();

		//do necessary transformations here
		translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
		translationMatrix = vmath::translate(x_plane_pos, 0.0f, -20.0f);

		//do necessary matrix multiplication
		modelViewMatrix = modelViewMatrix * translationMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		oglDraw_plane();
	}

	

	//unuse program
	glUseProgram(0);
	SwapBuffers(ghdc);
}

void oglUpdate(void)
{
	//init translation Of I
	if (f_Translate_I <= 0.0f)
	{
		f_Translate_I = f_Translate_I + 0.0045f;
		if (f_Translate_I > 0.0f)
		{
			b_I_Done = true;
		}
	}

	//init translation A
	if (b_I_Done)
	{
		if (f_Translate_A > 0.0f)
		{
			f_Translate_A = f_Translate_A - 0.0041f;
			if (f_Translate_A < 0.0f)
			{
				b_A_Done = true;
			}
		}
	}

	//init translation N
	if (b_A_Done == true)
	{
		if (f_Translate_N > 0.0f)
		{
			f_Translate_N = f_Translate_N - 0.006f;
			if (f_Translate_N < 0.0f)
			{
				b_N_Done = true;
			}
		}
	}

	//init translation i
	if (b_N_Done == true)
	{
		if (f_Translate_i < 0.0f)
		{
			f_Translate_i = f_Translate_i + 0.00455f;
			if (f_Translate_i > 0.0f)
			{
				b_i_Done = true;
			}
		}
	}

	//init color transition of D
	if (b_i_Done == true)
	{
		if ((f_DRedColor <= 1.0f) && (f_DGreenColor <= 0.5f))
		{
			f_DRedColor += 0.00032f;
			f_DGreenColor += 0.00016f;

			if ((f_DRedColor > 1.0f) && (f_DGreenColor > 0.5f))
			{
				b_D_Done = true;
			}
		}
	}

	//plane transition
	if (b_D_Done == true)
	{
		if (x_plane_pos <= 22.0f)
		{
			x_plane_pos = x_plane_pos + 0.0445f;

			/* top plane */
			if (top.angle <= (M_PI + M_PI_2))
			{
				top._x = top.radius * cosf(top.angle) - 8.0f;
				top._y = top.radius * sinf(top.angle) + 10.0f;
				top.angle += 0.005f;
				if (top.angle > (M_PI + M_PI_2))
				{
					b_clip_top_plane = true;
				}
				if (top_angle_2 <= 4.71238f)
				{
					//top_angle_2 = top_angle_2 + 0.00475f;
					top_angle_2 = top_angle_2 + 0.00325f;
				}
			}
			//rotation angle calculation
			if (top.rotation_angle <= 0.0f)
			{
				top.rotation_angle += 0.21f;
			}

			/* bottom plane */
			if (bottom.angle >= M_PI_2)
			{
				bottom._x = bottom.radius * cosf(bottom.angle) - 8.0f;
				bottom._y = bottom.radius * sinf(bottom.angle) - 10.0f;
				bottom.angle -= 0.005f;
				if (bottom.angle <= M_PI_2)
				{
					b_clip_bottom_plane = true;
				}
				if (bottom_angle_2 <= M_PI_2)
				{
					bottom_angle_2 = bottom_angle_2 + 0.000475f;
				}
			}
			//rotation angle calculation
			if (bottom.rotation_angle >= 0.0f)
			{
				bottom.rotation_angle -= 0.21f;
			}

			if (x_plane_pos > 8.0f)
			{
				b_clip_top_plane = false;
				b_clip_bottom_plane = false;
				
				b_start_incrementing = true;
				b_start_decrementing = true;

				if (top.angle <= 2 * M_PI)
				{
					top._x = top.radius * cosf(top.angle) + 8.0f;
					top._y = top.radius * sinf(top.angle) + 10.0f;
					top.angle += 0.005f;

					if (top_angle_4 <= 6.28318f)
					{
						top_angle_4 += 0.00038f;
					}

					if (top.angle > 2 * M_PI)
					{
						b_clip_top_plane = true;
					}
					//angle related calculation
					top.rotation_angle += 0.21f;
				}
				
				
				if (bottom.angle >= 0.0f)
				{
					bottom._x = bottom.radius * cosf(bottom.angle) + 8.0f;
					bottom._y = bottom.radius * sinf(bottom.angle) - 10.0f;
					bottom.angle -= 0.005f;
					if (bottom.angle <= 0.0f)
					{
						b_clip_bottom_plane = true;
					}
					if (bottom_angle_4 >= 0.0f)
					{
						bottom_angle_4 -= 0.00038f;
					}
					//angle related calculation
					bottom.rotation_angle -= 0.21f;

				}
				if (x_plane_pos > 22.0f)
				{
					b_PlaneTrue = true;
				}
			}
		}
	}

	if (b_PlaneTrue)
	{
		if ((f_ARedColor <= 1.0f) && (f_AGreenColor <= 0.5f) && (f_ABlueColor <= 1.0f) && (f_AWhiteColor <= 1.0f))
		{
			f_ARedColor = f_ARedColor + 0.002f;
			f_AGreenColor = f_AGreenColor + 0.001f;
			f_ABlueColor = f_ABlueColor + 0.002f;
			f_AWhiteColor = f_AWhiteColor + 0.002f;
			
			/*if ((r > 1.0f) && (g > 0.5f) && (b > 1.0f) && (White_A > 1.0f))
			{
				bColourDone = true;
			}*/
		}
	}
}

void oglInitData(void)
{
	//code
	top.rotation_angle = -60.0f;
	bottom.rotation_angle = 60.0f;
	x_plane_pos = -22.0f;		//plane starting position
}

void oglDraw_I(void)
{
	//code
	glBindVertexArray(vao_I);

	//draw scene
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	//unbind vao
	glBindVertexArray(0);
}

void oglDraw_N(void)
{
	//code
	glBindVertexArray(vao_N);

	//draw scene
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);

	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);

	//unbind vao
	glBindVertexArray(0);
}

void oglDraw_D(void)
{
	//code
	glBindVertexArray(vao_D);

	GLfloat D_color[] =
	{
		f_DRedColor, f_DGreenColor, 0.0f,
		f_DRedColor, f_DGreenColor, 0.0f,
		f_DRedColor, f_DGreenColor, 0.0f,
		f_DRedColor, f_DGreenColor, 0.0f,

		0.0f, f_DGreenColor, 0.0f,
		0.0f, f_DGreenColor, 0.0f,
		0.0f, f_DGreenColor, 0.0f,
		0.0f, f_DGreenColor, 0.0f,

		f_DRedColor, f_DGreenColor, 0.0f,
		f_DRedColor, f_DGreenColor, 0.0f,
		0.0f, f_DGreenColor, 0.0f,
		0.0f, f_DGreenColor, 0.0f,

		f_DRedColor, f_DGreenColor, 0.0f,
		f_DRedColor, f_DGreenColor, 0.0f,
		0.0f, f_DGreenColor, 0.0f,
		0.0f, f_DGreenColor, 0.0f
	};

	glBindBuffer(GL_ARRAY_BUFFER, vbo_D_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(D_color), D_color, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//draw scene
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);

	//unbind vao
	glBindVertexArray(0);
}

void oglDraw_i(void)
{
	//code
	glBindVertexArray(vao_i);

	//draw scene
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	//unbind vao
	glBindVertexArray(0);
}

void oglDraw_A(void)
{
	//code
	glBindVertexArray(vao_A);

	//draw scene
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	
	//unbind vao
	glBindVertexArray(0);
}

void oglDraw_middleStrips(void)
{
	//code
	glBindVertexArray(vao_middleStrips);

	const GLfloat middleStrips_color[] =
	{
		f_ARedColor, f_AGreenColor, 0.0f,
		f_ARedColor, f_AGreenColor, 0.0f,
		f_ARedColor, f_AGreenColor, 0.0f,
		f_ARedColor, f_AGreenColor, 0.0f,

		f_ARedColor, f_AWhiteColor, f_ABlueColor,
		f_ARedColor, f_AWhiteColor, f_ABlueColor,
		f_ARedColor, f_AWhiteColor, f_ABlueColor,
		f_ARedColor, f_AWhiteColor, f_ABlueColor,

		0.0f, f_AGreenColor, 0.0f,
		0.0f, f_AGreenColor, 0.0f,
		0.0f, f_AGreenColor, 0.0f,
		0.0f, f_AGreenColor, 0.0f
	};

	glBindBuffer(GL_ARRAY_BUFFER, vbo_middleStrips_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(middleStrips_color), middleStrips_color, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//draw scene
	glLineWidth(3.0f);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);

	//unbind vao
	glBindVertexArray(0);
}
/*
void oglDrawSmoke_top(int loop_no)
{
	//variable declaration
	GLfloat top_plane_smoke_vertices[9];
	GLfloat top_plane_smoke_color[9];
	//GLfloat bottom_plane_smoke_vertices[9];
	//GLfloat bottom_plane_smoke_color[9];

	//code
	if (loop_no == 1)	//decrementing flag
	{
		for (float i = top_angle_1; i < top_angle_2; i = i + 0.000535f)
		{

			top_plane_smoke_color[0] = f_red;
			top_plane_smoke_color[1] = f_green;
			top_plane_smoke_color[2] = 0.0f;
			top_plane_smoke_color[3] = f_red;
			top_plane_smoke_color[4] = f_white;
			top_plane_smoke_color[5] = f_blue;
			top_plane_smoke_color[6] = 0.0f;
			top_plane_smoke_color[7] = f_green;
			top_plane_smoke_color[8] = 0.0f;

			glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_color);
			glBufferData(GL_ARRAY_BUFFER, sizeof(top_plane_smoke_color), top_plane_smoke_color, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			top_plane_smoke_vertices[0] = top.radius * cosf(i) - 8.0f;
			top_plane_smoke_vertices[1] = top.radius * sinf(i) + 10.0f + 0.15f;
			top_plane_smoke_vertices[2] = 0.0f;
			top_plane_smoke_vertices[3] = top.radius * cosf(i) - 8.0f;
			top_plane_smoke_vertices[4] = top.radius * sinf(i) + 10.0f;
			top_plane_smoke_vertices[5] = 0.0f;
			top_plane_smoke_vertices[6] = top.radius * cosf(i) - 8.0f;
			top_plane_smoke_vertices[7] = top.radius * sinf(i) + 10.0f - 0.15f;
			top_plane_smoke_vertices[8] = 0.0f;

			glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_position);
			glBufferData(GL_ARRAY_BUFFER, sizeof(top_plane_smoke_vertices), top_plane_smoke_vertices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glPointSize(5.0f);
			glBindVertexArray(vao_triColorStrip);
			glDrawArrays(GL_POINTS, 0, 3);
			glBindVertexArray(0);
		}
	}

	if (loop_no == 2)		//incrementing flag
	{
		for (float i = top_angle_3; i < top_angle_4; i = i + 0.00055f)
		{

			top_plane_smoke_color[0] = f_red;
			top_plane_smoke_color[1] = f_green;
			top_plane_smoke_color[2] = 0.0f;
			top_plane_smoke_color[3] = f_red;
			top_plane_smoke_color[4] = f_white;
			top_plane_smoke_color[5] = f_blue;
			top_plane_smoke_color[6] = 0.0f;
			top_plane_smoke_color[7] = f_green;
			top_plane_smoke_color[8] = 0.0f;

			glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_color);
			glBufferData(GL_ARRAY_BUFFER, sizeof(top_plane_smoke_color), top_plane_smoke_color, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			top_plane_smoke_vertices[0] = top.radius * cosf(i) + 7.5f;
			top_plane_smoke_vertices[1] = top.radius * sinf(i) + 10.0f + 0.15f;
			top_plane_smoke_vertices[2] = 0.0f;
			top_plane_smoke_vertices[3] = top.radius * cosf(i) + 7.5f;
			top_plane_smoke_vertices[4] = top.radius * sinf(i) + 10.0f;
			top_plane_smoke_vertices[5] = 0.0f;
			top_plane_smoke_vertices[6] = top.radius * cosf(i) + 7.5f;
			top_plane_smoke_vertices[7] = top.radius * sinf(i) + 10.0f - 0.15f;
			top_plane_smoke_vertices[8] = 0.0f;

			glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_position);
			glBufferData(GL_ARRAY_BUFFER, sizeof(top_plane_smoke_vertices), top_plane_smoke_vertices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glPointSize(5.0f);
			glBindVertexArray(vao_triColorStrip);
			glDrawArrays(GL_POINTS, 0, 3);
			glBindVertexArray(0);
		}
	}

}

void oglDrawSmoke_middle(void)
{
	GLfloat middle_plane_smoke_vertices[9];
	GLfloat middle_plane_smoke_color[9];

	middle_plane_smoke_color[0] = f_red;
	middle_plane_smoke_color[1] = f_green;
	middle_plane_smoke_color[2] = 0.0f;
	middle_plane_smoke_color[3] = f_red;
	middle_plane_smoke_color[4] = f_white;
	middle_plane_smoke_color[5] = f_blue;
	middle_plane_smoke_color[6] = 0.0f;
	middle_plane_smoke_color[7] = f_green;
	middle_plane_smoke_color[8] = 0.0f;

	glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(middle_plane_smoke_color), middle_plane_smoke_color, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	middle_plane_smoke_vertices[0] = middle_plane_smoke;
	middle_plane_smoke_vertices[1] = middle_plane_smoke + 0.15f;
	middle_plane_smoke_vertices[2] = 0.0f;
	middle_plane_smoke_vertices[3] = middle_plane_smoke;
	middle_plane_smoke_vertices[4] = middle_plane_smoke - 10.0f;
	middle_plane_smoke_vertices[5] = 0.0f;
	middle_plane_smoke_vertices[6] = middle_plane_smoke;
	middle_plane_smoke_vertices[7] = middle_plane_smoke - 0.15f;
	middle_plane_smoke_vertices[8] = 0.0f;

	glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(middle_plane_smoke_vertices), middle_plane_smoke_vertices, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glPointSize(5.0f);
	glBindVertexArray(vao_triColorStrip);
	glDrawArrays(GL_POINTS, 0, 3);
	glBindVertexArray(0);
}

void oglDrawSmoke_bottom(int loop_no)
{
	//variable declaration
	GLfloat bottom_plane_smoke_vertices[9];
	GLfloat bottom_plane_smoke_color[9];

	//code
	if (loop_no == 1)		//decrementing flag
	{
		for (float i = bottom_angle_1; i < bottom_angle_2; i = i + 0.000535f)
		{

			bottom_plane_smoke_color[0] = f_red;
			bottom_plane_smoke_color[1] = f_green;
			bottom_plane_smoke_color[2] = 0.0f;
			bottom_plane_smoke_color[3] = f_red;
			bottom_plane_smoke_color[4] = f_white;
			bottom_plane_smoke_color[5] = f_blue;
			bottom_plane_smoke_color[6] = 0.0f;
			bottom_plane_smoke_color[7] = f_green;
			bottom_plane_smoke_color[8] = 0.0f;

			glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_color);
			glBufferData(GL_ARRAY_BUFFER, sizeof(bottom_plane_smoke_color), bottom_plane_smoke_color, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			bottom_plane_smoke_vertices[0] = bottom.radius * cosf(i) - 8.0f;
			bottom_plane_smoke_vertices[1] = bottom.radius * sinf(i) - 10.0f + 0.15f;
			bottom_plane_smoke_vertices[2] = 0.0f;
			bottom_plane_smoke_vertices[3] = bottom.radius * cosf(i) - 8.0f;
			bottom_plane_smoke_vertices[4] = bottom.radius * sinf(i) - 10.0f;
			bottom_plane_smoke_vertices[5] = 0.0f;
			bottom_plane_smoke_vertices[6] = bottom.radius * cosf(i) - 8.0f;
			bottom_plane_smoke_vertices[7] = bottom.radius * sinf(i) - 10.0f - 0.15f;
			bottom_plane_smoke_vertices[8] = 0.0f;

			glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_position);
			glBufferData(GL_ARRAY_BUFFER, sizeof(bottom_plane_smoke_vertices), bottom_plane_smoke_vertices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glPointSize(5.0f);
			glBindVertexArray(vao_triColorStrip);
			glDrawArrays(GL_POINTS, 0, 3);
			glBindVertexArray(0);
		}
	}

	if (loop_no == 2)		//incrementing flag
	{
		for (float i = bottom_angle_3; i < bottom_angle_4; i = i + 0.00055f)
		{

			bottom_plane_smoke_color[0] = f_red;
			bottom_plane_smoke_color[1] = f_green;
			bottom_plane_smoke_color[2] = 0.0f;
			bottom_plane_smoke_color[3] = f_red;
			bottom_plane_smoke_color[4] = f_white;
			bottom_plane_smoke_color[5] = f_blue;
			bottom_plane_smoke_color[6] = 0.0f;
			bottom_plane_smoke_color[7] = f_green;
			bottom_plane_smoke_color[8] = 0.0f;

			glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_color);
			glBufferData(GL_ARRAY_BUFFER, sizeof(bottom_plane_smoke_color), bottom_plane_smoke_color, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			bottom_plane_smoke_vertices[0] = bottom.radius * cosf(i) + 7.5f;
			bottom_plane_smoke_vertices[1] = bottom.radius * sinf(i) - 10.0f + 0.15f;
			bottom_plane_smoke_vertices[2] = 0.0f;
			bottom_plane_smoke_vertices[3] = bottom.radius * cosf(i) + 7.5f;
			bottom_plane_smoke_vertices[4] = bottom.radius * sinf(i) - 10.0f;
			bottom_plane_smoke_vertices[5] = 0.0f;
			bottom_plane_smoke_vertices[6] = bottom.radius * cosf(i) + 7.5f;
			bottom_plane_smoke_vertices[7] = bottom.radius * sinf(i) - 10.0f - 0.15f;
			bottom_plane_smoke_vertices[8] = 0.0f;

			glBindBuffer(GL_ARRAY_BUFFER, vbo_triColorStrip_position);
			glBufferData(GL_ARRAY_BUFFER, sizeof(bottom_plane_smoke_vertices), bottom_plane_smoke_vertices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glPointSize(5.0f);
			glBindVertexArray(vao_triColorStrip);
			glDrawArrays(GL_POINTS, 0, 3);
			glBindVertexArray(0);
		}
	}
}
*/

void oglDraw_plane(void)
{
	//code
	glBindVertexArray(vao_plane);

	//plane
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	//body
	glDrawArrays(GL_TRIANGLES, 4, 3);		//exahaust
	glDrawArrays(GL_TRIANGLE_FAN, 7, 4);	//orange flag
	glDrawArrays(GL_TRIANGLE_FAN, 11, 4);	//white	flag
	glDrawArrays(GL_TRIANGLE_FAN, 15, 4);	//green	flag
	glDrawArrays(GL_LINES, 19, 2);			//separator line between exhaust and body
	glDrawArrays(GL_TRIANGLES, 21, 3);		//front tip
	glDrawArrays(GL_LINES, 24, 2);			//sperator line between front tip and body
	glDrawArrays(GL_TRIANGLES, 26, 3);		//upper wing
	glDrawArrays(GL_TRIANGLES, 29, 3);		//lower wing

	//IAF
	glDrawArrays(GL_LINES, 32, 2);			//I

	glDrawArrays(GL_LINES, 34, 2);			//A
	glDrawArrays(GL_LINES, 36, 2);
	glDrawArrays(GL_LINES, 38, 2);

	glDrawArrays(GL_LINES, 40, 2);			//F
	glDrawArrays(GL_LINES, 42, 2);
	glDrawArrays(GL_LINES, 44, 2);
	
	//unbind vao
	glBindVertexArray(0);
}

void oglUninitialise(void)
{
	fprintf_s(gpFile, "\nIn OGLUninitialise\n");

	//code
	if (vbo_I_position)
	{
		glDeleteBuffers(1, &vbo_I_position);
		vbo_I_position = 0;
	}
	if (vbo_I_color)
	{
		glDeleteBuffers(1, &vbo_I_color);
		vbo_I_color = 0;
	}

	if (vbo_N_position)
	{
		glDeleteBuffers(1, &vbo_N_position);
		vbo_N_position = 0;
	}
	if (vbo_N_color)
	{
		glDeleteBuffers(1, &vbo_N_color);
		vbo_N_color = 0;
	}

	if (vbo_D_position)
	{
		glDeleteBuffers(1, &vbo_D_position);
		vbo_D_position = 0;
	}
	if (vbo_D_color)
	{
		glDeleteBuffers(1, &vbo_D_color);
		vbo_D_color = 0;
	}

	if (vbo_i_position)
	{
		glDeleteBuffers(1, &vbo_i_position);
		vbo_i_position = 0;
	}
	if (vbo_i_color)
	{
		glDeleteBuffers(1, &vbo_i_color);
		vbo_i_color = 0;
	}

	if (vbo_A_position)
	{
		glDeleteBuffers(1, &vbo_A_position);
		vbo_A_position = 0;
	}
	if (vbo_A_color)
	{
		glDeleteBuffers(1, &vbo_A_color);
		vbo_A_color = 0;
	}

	if (vbo_plane_position)
	{
		glDeleteBuffers(1, &vbo_plane_position);
		vbo_plane_position = 0;
	}
	if (vbo_plane_color)
	{
		glDeleteBuffers(1, &vbo_plane_color);
		vbo_plane_color = 0;
	}

	if (vao_I)
	{
		glDeleteVertexArrays(1, &vao_I);
		vao_I = 0;
	}

	if (vao_N)
	{
		glDeleteVertexArrays(1, &vao_N);
		vao_N = 0;
	}

	if (vao_D)
	{
		glDeleteVertexArrays(1, &vao_D);
		vao_D = 0;
	}

	if (vao_i)
	{
		glDeleteVertexArrays(1, &vao_i);
		vao_i = 0;
	}

	if (vao_A)
	{
		glDeleteVertexArrays(1, &vao_A);
		vao_A = 0;
	}

	if (vao_plane)
	{
		glDeleteVertexArrays(1, &vao_plane);
		vao_plane = 0;
	}

	//safe release

	if (gShaderProgramObject)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;

		glUseProgram(gShaderProgramObject);

		//ask program how many shaders are attached
		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);

		GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

		if (pShaders)
		{
			glGetAttachedShaders(gShaderProgramObject, shaderCount, &shaderCount, pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				//detach shader
				glDetachShader(gShaderProgramObject, pShaders[shaderNumber]);
				//delete shader
				glDeleteShader(pShaders[shaderNumber]);
				pShaders[shaderNumber] = 0;
			}
			free(pShaders);
		}
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);
	}

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

	if (gpFile)
	{
		fprintf_s(gpFile, "Log File Is Closed Successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}
