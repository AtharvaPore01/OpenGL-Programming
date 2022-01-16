//Standard Header Files
#include <Windows.h>
#include <stdio.h>

//OpenGL Related Header Files
#include <gl/glew.h>
#include <GL/GL.h>
#include "vmath.h"

//Library Function
#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "glew32.lib")
#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "kernel32.lib")

//Macros
#define WIN_WIDTH 800
#define WIN_HEIGHT 600
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

GLuint vao_red;
GLuint vao_green;
GLuint vao_blue;
GLuint vao_circumscribed_circle;
GLuint vao_circumscribed_circle_triangle;

GLuint vao_circumscribed_rectangle;
GLuint vao_circumscribed_rectangle_circle;

GLuint vbo_red_line_position;
GLuint vbo_red_line_color;
GLuint vbo_green_line_position;
GLuint vbo_green_line_color;
GLuint vbo_blue_line_position;

GLuint vbo_circumscribed_circle_position_triangle;
GLuint vbo_circumscribed_circle_position_circle;
GLuint vbo_circumscribed_circle_color;
GLuint vbo_circumscribed_circle_color_triangle;

GLuint vbo_circumscribed_rectangle_position;
GLuint vbo_circumscribed_rectangle_position_circle;
GLuint vbo_circumscribed_rectangle_color;
GLuint vbo_circumscribed_rectangle_color_circle;

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
		TEXT("All Geometry Shapes"),
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
				//update call
			}
		}
	}

	return((int)msg.wParam);
}
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	fprintf_s(gpFile, "\n2. In WndProc\n");
	//function declaration
	void oglToggleFullScreen(void);
	void oglResize(int, int);
	void oglUninitialise(void);

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
			oglToggleFullScreen();
			break;
		}
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
		"in vec4 vColor;" \
		"out vec4 out_color;"
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_color = vColor;" \
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
		"in vec4 out_color;" \
		"out vec4 FragColor;" \
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
	//line vertices declaration
	const GLfloat blueLines[] =
	{
		-0.95f, 1.0f, 0.0f,
		-0.95f, -1.0f, 0.0f,

		-0.90f, 1.0f, 0.0f,
		-0.90f, -1.0f, 0.0f,

		-0.85f, 1.0f, 0.0f,
		-0.85f, -1.0f, 0.0f,

		-0.80f, 1.0f, 0.0f,
		-0.80f, -1.0f, 0.0f,

		-0.75f, 1.0f, 0.0f,
		-0.75f, -1.0f, 0.0f,

		-0.70f, 1.0f, 0.0f,
		-0.70f, -1.0f, 0.0f,

		-0.65f, 1.0f, 0.0f,
		-0.65f, -1.0f, 0.0f,

		-0.60f, 1.0f, 0.0f,
		-0.60f, -1.0f, 0.0f,

		-0.55f, 1.0f, 0.0f,
		-0.55f, -1.0f, 0.0f,

		-0.50f, 1.0f, 0.0f,
		-0.50f, -1.0f, 0.0f,

		-0.45f, 1.0f, 0.0f,
		-0.45f, -1.0f, 0.0f,

		-0.40f, 1.0f, 0.0f,
		-0.40f, -1.0f, 0.0f,

		-0.35f, 1.0f, 0.0f,
		-0.35f, -1.0f, 0.0f,

		-0.30f, 1.0f, 0.0f,
		-0.30f, -1.0f, 0.0f,

		-0.25f, 1.0f, 0.0f,
		-0.25f, -1.0f, 0.0f,

		-0.20f, 1.0f, 0.0f,
		-0.20f, -1.0f, 0.0f,

		-0.15f, 1.0f, 0.0f,
		-0.15f, -1.0f, 0.0f,

		-0.10f, 1.0f, 0.0f,
		-0.10f, -1.0f, 0.0f,

		-0.05f, 1.0f, 0.0f,
		-0.05f, -1.0f, 0.0f,

		0.95f, 1.0f, 0.0f,
		0.95f, -1.0f, 0.0f,

		0.90f, 1.0f, 0.0f,
		0.90f, -1.0f, 0.0f,

		0.85f, 1.0f, 0.0f,
		0.85f, -1.0f, 0.0f,

		0.80f, 1.0f, 0.0f,
		0.80f, -1.0f, 0.0f,

		0.75f, 1.0f, 0.0f,
		0.75f, -1.0f, 0.0f,

		0.70f, 1.0f, 0.0f,
		0.70f, -1.0f, 0.0f,

		0.65f, 1.0f, 0.0f,
		0.65f, -1.0f, 0.0f,

		0.60f, 1.0f, 0.0f,
		0.60f, -1.0f, 0.0f,

		0.55f, 1.0f, 0.0f,
		0.55f, -1.0f, 0.0f,

		0.50f, 1.0f, 0.0f,
		0.50f, -1.0f, 0.0f,

		0.45f, 1.0f, 0.0f,
		0.45f, -1.0f, 0.0f,

		0.40f, 1.0f, 0.0f,
		0.40f, -1.0f, 0.0f,

		0.35f, 1.0f, 0.0f,
		0.35f, -1.0f, 0.0f,

		0.30f, 1.0f, 0.0f,
		0.30f, -1.0f, 0.0f,

		0.25f, 1.0f, 0.0f,
		0.25f, -1.0f, 0.0f,

		0.20f, 1.0f, 0.0f,
		0.20f, -1.0f, 0.0f,

		0.15f, 1.0f, 0.0f,
		0.15f, -1.0f, 0.0f,

		0.10f, 1.0f, 0.0f,
		0.10f, -1.0f, 0.0f,

		0.05f, 1.0f, 0.0f,
		0.05f, -1.0f, 0.0f,

		1.0f, -0.95f, 0.0f,
		-1.0f, -0.95, 0.0f,

		1.0f, -0.90f, 0.0f,
		-1.0f, -0.90f, 0.0f,

		1.0f, -0.85f, 0.0f,
		-1.0f, -0.85f, 0.0f,

		1.0f, -0.80f, 0.0f,
		-1.0f, -0.80f, 0.0f,

		1.0f, -0.75f, 0.0f,
		-1.0f, -0.75f, 0.0f,

		1.0f, -0.70f, 0.0f,
		-1.0f, -0.70f, 0.0f,

		1.0f, -0.65f, 0.0f,
		-1.0f, -0.65f, 0.0f,

		1.0f, -0.60f, 0.0f,
		-1.0f, -0.60f, 0.0f,

		1.0f, -0.55f, 0.0f,
		-1.0f, -0.55f, 0.0f,

		1.0f, -0.50f, 0.0f,
		-1.0f, -0.50f, 0.0f,

		1.0f, -0.45f, 0.0f,
		-1.0f, -0.45f, 0.0f,

		1.0f, -0.40f, 0.0f,
		-1.0f, -0.40f, 0.0f,

		1.0f, -0.35f, 0.0f,
		-1.0f, -0.35f, 0.0f,

		1.0f, -0.30f, 0.0f,
		-1.0f, -0.30f, 0.0f,

		1.0f, -0.25f, 0.0f,
		-1.0f, -0.25f, 0.0f,

		1.0f, -0.20f, 0.0f,
		-1.0f, -0.20f, 0.0f,

		1.0f, -0.15f, 0.0f,
		-1.0f, -0.15f, 0.0f,

		1.0f, -0.10f, 0.0f,
		-1.0f, -0.10f, 0.0f,

		1.0f, -0.05f, 0.0f,
		-1.0f, -0.05f, 0.0f,

		1.0f, 0.95f, 0.0f,
		-1.0f, 0.95f, 0.0f,

		1.0f, 0.90f, 0.0f,
		-1.0f, 0.90f, 0.0f,

		1.0f, 0.85f, 0.0f,
		-1.0f, 0.85f, 0.0f,

		1.0f, 0.80f, 0.0f,
		-1.0f, 0.80f, 0.0f,

		1.0f, 0.75f, 0.0f,
		-1.0f, 0.75f, 0.0f,

		1.0f, 0.70f, 0.0f,
		-1.0f, 0.70f, 0.0f,

		1.0f, 0.65f, 0.0f,
		-1.0f, 0.65f, 0.0f,

		1.0f, 0.60f, 0.0f,
		-1.0f, 0.60f, 0.0f,

		1.0f, 0.55f, 0.0f,
		-1.0f, 0.55f, 0.0f,

		1.0f, 0.50f, 0.0f,
		-1.0f, 0.50f, 0.0f,

		1.0f, 0.45f, 0.0f,
		-1.0f, 0.45f, 0.0f,

		1.0f, 0.40f, 0.0f,
		-1.0f, 0.40f, 0.0f,

		1.0f, 0.35f, 0.0f,
		-1.0f, 0.35f, 0.0f,

		1.0f, 0.30f, 0.0f,
		-1.0f, 0.30f, 0.0f,

		1.0f, 0.25f, 0.0f,
		-1.0f, 0.25f, 0.0f,

		1.0f, 0.20f, 0.0f,
		-1.0f, 0.20f, 0.0f,

		1.0f, 0.15f, 0.0f,
		-1.0f, 0.15f, 0.0f,

		1.0f, 0.10f, 0.0f,
		-1.0f, 0.10f, 0.0f,

		1.0f, 0.05f, 0.0f,
		-1.0f, 0.05f, 0.0f
	};

	const GLfloat redLine[] =
	{
		1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f,
	};

	const GLfloat greenLine[] =
	{
		0.0f, 1.0f, 0.0f,
		0.0f, -1.0f, 0.0f
	};

	//color buffers
	const GLfloat redColor[] =
	{
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f
	};
	const GLfloat greenColor[] =
	{
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f
	};

	//triangle of circumscribed circle
	const GLfloat triangleVertices[] =
	{
		0.0f, 1.2f, 0.0f,
		-1.0f, -0.6f, 0.0f,

		-1.0f, -0.6f, 0.0f,
		1.0f, -0.6f, 0.0f,

		1.0f, -0.6f, 0.0f,
		0.0f, 1.2f, 0.0f
	};

	const GLfloat circumscribedCircleColor[] =
	{
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};

	//circumscribed rectangle
	const GLfloat rectangleVertices[] =
	{
		1.0f, 0.58f, 0.0f,
		-1.0f, 0.58f, 0.0f,

		-1.0f, -0.6f, 0.0f,
		1.0f, -0.6f, 0.0f,

		1.0f, -0.6f, 0.0f,
		1.0f, 0.58f, 0.0f,

		-1.0f, 0.58f, 0.0f,
		-1.0f, -0.6f, 0.0f
	};

	const GLfloat circumscribedRectangleColor[] =
	{
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f

	};

	//circumscribed circle
	//triangle
	glGenVertexArrays(1, &vao_circumscribed_circle_triangle);
	glBindVertexArray(vao_circumscribed_circle_triangle);

	glGenBuffers(1, &vbo_circumscribed_circle_position_triangle);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_position_triangle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_circumscribed_circle_color_triangle);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_color_triangle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(circumscribedCircleColor), circumscribedCircleColor, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//circle
	glGenVertexArrays(1, &vao_circumscribed_circle);
	glBindVertexArray(vao_circumscribed_circle);
	glGenBuffers(1, &vbo_circumscribed_circle_position_circle);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_position_circle);
	glBufferData(GL_ARRAY_BUFFER, 1 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_circumscribed_circle_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(circumscribedCircleColor), circumscribedCircleColor, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//circumscribed rectangle
	//rectangle
	glGenVertexArrays(1, &vao_circumscribed_rectangle);
	glBindVertexArray(vao_circumscribed_rectangle);

	glGenBuffers(1, &vbo_circumscribed_rectangle_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(rectangleVertices), rectangleVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_circumscribed_rectangle_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_color);

	glBufferData(GL_ARRAY_BUFFER, sizeof(circumscribedRectangleColor), circumscribedRectangleColor, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//circle
	glGenVertexArrays(1, &vao_circumscribed_rectangle_circle);
	glBindVertexArray(vao_circumscribed_rectangle_circle);

	glGenBuffers(1, &vbo_circumscribed_rectangle_position_circle);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_position_circle);

	glBufferData(GL_ARRAY_BUFFER, 1 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_circumscribed_rectangle_color_circle);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_color_circle);

	glBufferData(GL_ARRAY_BUFFER, sizeof(circumscribedRectangleColor), circumscribedRectangleColor, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	//create vao and vbo
	glGenVertexArrays(1, &vao_green);
	glBindVertexArray(vao_green);

	//green
	glGenBuffers(1, &vbo_green_line_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_green_line_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(greenLine), greenLine, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_green_line_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_green_line_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(greenColor), greenColor, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//red
	glGenVertexArrays(1, &vao_red);
	glBindVertexArray(vao_red);

	glGenBuffers(1, &vbo_red_line_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_red_line_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(redLine), redLine, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_red_line_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_red_line_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(redColor), redColor, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//blue
	glGenVertexArrays(1, &vao_blue);
	glBindVertexArray(vao_blue);

	glGenBuffers(1, &vbo_blue_line_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_blue_line_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(blueLines), blueLines, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 1.0f);

	glBindVertexArray(0);

	
	//clear the window
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

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
	void oglGenCircleInsideTriangle(void);
	void oglGenCircumscribedRectangle(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderProgramObject);

	//declaration of metrices
	vmath::mat4 modelViewMatrix;
	vmath::mat4 modelViewProjectionMatrix;
	vmath::mat4 translationMatrix;

	//circumscribed circle

	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();

	//do necessary transformations here
	translationMatrix = vmath::translate(0.0f, 0.0f, -3.9f);

	//do necessary matrix multiplication
	modelViewMatrix *= translationMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	oglGenCircleInsideTriangle();

	oglGenCircumscribedRectangle();
		
	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();

	//do necessary transformations here
	modelViewMatrix = vmath::translate(0.0f, 0.0f, -1.2f);

	//do necessary matrix multiplication
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	//bind with vao red
	glBindVertexArray(vao_red);
	//draw scene
	glDrawArrays(GL_LINES, 0, 2);
	//unbind vao red
	glBindVertexArray(0);

	//bind with vao green
	glBindVertexArray(vao_green);
	//draw scene
	glDrawArrays(GL_LINES, 0, 2);
	//unbind vao green
	glBindVertexArray(0);

	//bind with vao blue
	glBindVertexArray(vao_blue);

	//draw scene
	glDrawArrays(GL_LINES, 0, 2);
	glDrawArrays(GL_LINES, 2, 2);
	glDrawArrays(GL_LINES, 4, 2);
	glDrawArrays(GL_LINES, 6, 2);
	glDrawArrays(GL_LINES, 8, 2);
	glDrawArrays(GL_LINES, 10, 2);
	glDrawArrays(GL_LINES, 12, 2);
	glDrawArrays(GL_LINES, 14, 2);
	glDrawArrays(GL_LINES, 16, 2);
	glDrawArrays(GL_LINES, 18, 2);
	glDrawArrays(GL_LINES, 20, 2);

	glDrawArrays(GL_LINES, 22, 2);
	glDrawArrays(GL_LINES, 24, 2);
	glDrawArrays(GL_LINES, 26, 2);
	glDrawArrays(GL_LINES, 28, 2);
	glDrawArrays(GL_LINES, 30, 2);
	glDrawArrays(GL_LINES, 32, 2);
	glDrawArrays(GL_LINES, 34, 2);
	glDrawArrays(GL_LINES, 36, 2);
	glDrawArrays(GL_LINES, 38, 2);
	glDrawArrays(GL_LINES, 40, 2);
	glDrawArrays(GL_LINES, 42, 2);

	glDrawArrays(GL_LINES, 44, 2);
	glDrawArrays(GL_LINES, 46, 2);
	glDrawArrays(GL_LINES, 48, 2);
	glDrawArrays(GL_LINES, 50, 2);
	glDrawArrays(GL_LINES, 52, 2);
	glDrawArrays(GL_LINES, 54, 2);
	glDrawArrays(GL_LINES, 56, 2);
	glDrawArrays(GL_LINES, 58, 2);
	glDrawArrays(GL_LINES, 60, 2);
	glDrawArrays(GL_LINES, 62, 2);
	glDrawArrays(GL_LINES, 64, 2);

	glDrawArrays(GL_LINES, 66, 2);
	glDrawArrays(GL_LINES, 68, 2);
	glDrawArrays(GL_LINES, 70, 2);
	glDrawArrays(GL_LINES, 72, 2);
	glDrawArrays(GL_LINES, 74, 2);
	glDrawArrays(GL_LINES, 76, 2);
	glDrawArrays(GL_LINES, 78, 2);
	glDrawArrays(GL_LINES, 80, 2);
	glDrawArrays(GL_LINES, 82, 2);
	glDrawArrays(GL_LINES, 84, 2);
	glDrawArrays(GL_LINES, 86, 2);

	glDrawArrays(GL_LINES, 88, 2);
	glDrawArrays(GL_LINES, 90, 2);
	glDrawArrays(GL_LINES, 92, 2);
	glDrawArrays(GL_LINES, 94, 2);
	glDrawArrays(GL_LINES, 96, 2);
	glDrawArrays(GL_LINES, 98, 2);
	glDrawArrays(GL_LINES, 100, 2);
	glDrawArrays(GL_LINES, 102, 2);
	glDrawArrays(GL_LINES, 104, 2);
	glDrawArrays(GL_LINES, 106, 2);
	glDrawArrays(GL_LINES, 108, 2);

	glDrawArrays(GL_LINES, 110, 2);
	glDrawArrays(GL_LINES, 112, 2);
	glDrawArrays(GL_LINES, 114, 2);
	glDrawArrays(GL_LINES, 116, 2);
	glDrawArrays(GL_LINES, 118, 2);
	glDrawArrays(GL_LINES, 120, 2);
	glDrawArrays(GL_LINES, 122, 2);
	glDrawArrays(GL_LINES, 124, 2);
	glDrawArrays(GL_LINES, 126, 2);
	glDrawArrays(GL_LINES, 128, 2);
	glDrawArrays(GL_LINES, 130, 2);

	glDrawArrays(GL_LINES, 132, 2);
	glDrawArrays(GL_LINES, 134, 2);
	glDrawArrays(GL_LINES, 136, 2);
	glDrawArrays(GL_LINES, 138, 2);
	glDrawArrays(GL_LINES, 140, 2);
	glDrawArrays(GL_LINES, 142, 2);
	glDrawArrays(GL_LINES, 144, 2);
	glDrawArrays(GL_LINES, 146, 2);
	glDrawArrays(GL_LINES, 148, 2);
	glDrawArrays(GL_LINES, 150, 2);
	glDrawArrays(GL_LINES, 152, 2);

	glDrawArrays(GL_LINES, 154, 2);
	glDrawArrays(GL_LINES, 156, 2);
	glDrawArrays(GL_LINES, 158, 2);
	glDrawArrays(GL_LINES, 160, 2);
	glDrawArrays(GL_LINES, 162, 2);
	glDrawArrays(GL_LINES, 164, 2);

	//unbind vao blue
	glBindVertexArray(0);

	//unuse program
	glUseProgram(0);
	SwapBuffers(ghdc);
}
void oglUpdate(void)
{
	//code
}


void oglGenCircleInsideTriangle(void)
{
	//OGLCircleInsideTriangle Variables
	GLfloat radius, a, b, c;
	GLfloat Perimeter;
	GLfloat Area_Of_Triangle, x_center, y_center;
	GLfloat circleVertices[3];

	//function declaration
	GLfloat findDistance(GLfloat, GLfloat, GLfloat, GLfloat);
	GLfloat findPerimeter(GLfloat, GLfloat, GLfloat);
	GLfloat findXCenter(GLfloat a, GLfloat b, GLfloat c);
	GLfloat findYCenter(GLfloat a, GLfloat b, GLfloat c);
	GLfloat findAreaOfTriangle(GLfloat Perimeter, GLfloat a, GLfloat b, GLfloat c);
	GLfloat findRadius(GLfloat AreaOfTrianlge, GLfloat Perimeter);

	//code
	//Distance Between Vertices Of The Triangle
	a = sqrtf((powf((-1.0f - 0.0f), 2) + powf((-0.6f - 1.2f), 2)));
	b = sqrtf((powf((1.0f - (-1.0f)), 2) + powf((-0.6f - (-0.6f)), 2)));
	c = sqrtf((powf((0.0f - 1.0f), 2) + powf((1.2f - (-0.6f)), 2)));
	
	//Semi Perimeter
	Perimeter = (a + b + c) / 2;

	//Area Of Trianle Using Heron's Formula
	Area_Of_Triangle = sqrtf(Perimeter * (Perimeter - a) * (Perimeter - b) * (Perimeter - c));

	//Radius Of Circle
	radius = Area_Of_Triangle / Perimeter;
	
	//Center Of The Circle
	x_center = ((a * 1.0f) + (b * (0.0f)) + (c * (-1.0f))) / (a + b + c);
	y_center = ((a * (-0.6f)) + (b * (1.2f)) + (c * (-0.6f))) / (a + b + c);

	//bind with vao
	glBindVertexArray(vao_circumscribed_circle_triangle);

	glDrawArrays(GL_LINES, 0, 2);
	glDrawArrays(GL_LINES, 2, 2);
	glDrawArrays(GL_LINES, 4, 2);

	//unbind vao
	glBindVertexArray(0);

	//bind with vao
	glBindVertexArray(vao_circumscribed_circle);
	for (GLfloat angle = 0.0f; angle < (2.0f * M_PI); angle = angle + 0.01f)
	{
		circleVertices[0] = ((cosf(angle) * radius) + x_center);
		circleVertices[1] = ((sinf(angle) * radius) + y_center);
		circleVertices[2] = 0.0f;

		//vertices
		glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_position_circle);
		glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		//draw scene
		glPointSize(1.5f);
		glDrawArrays(GL_POINTS, 0, 1);
		//glDrawArrays(GL_LINE_LOOP, 0, 10);
	}
	
	//unbind vao
	glBindVertexArray(0);
}

void oglGenCircumscribedRectangle(void)
{
	//variable declaration
	GLfloat Number_Of_Sides = 4.0f;
	GLfloat radius;
	GLfloat Top, Bottom, Left, Right, Area_of_Rectangle;
	GLfloat circleVertices[3];

	//code
	
	//Rectangle's Sides
	Top = sqrtf((powf((-1.0f - 1.0f), 2) + powf((0.58f - 0.58f), 2)));
	fprintf_s(gpFile, "\nTop = %f\n", Top);
	Bottom = sqrtf((powf((1.0f - (-1.0f)), 2) + powf((-0.6f - (-0.6f)), 2)));
	fprintf_s(gpFile, "Bottom = %f\n", Bottom);
	Left = sqrtf((powf((1.0f - 1.0f), 2) + powf((-0.58f - 0.6f), 2)));
	fprintf_s(gpFile, "Left = %f\n", Left);
	Right = sqrtf((powf((-1.0f - (-1.0f)), 2) + powf((-0.6f - 0.58f), 2)));
	fprintf_s(gpFile, "Right = %f\n\n", Right);

	//Area Of Rectangle
	Area_of_Rectangle = (Top + Bottom + Left + Right) / 2;
	fprintf_s(gpFile, "Area_of_Rectangle = %f\n\n", Area_of_Rectangle);

	//Radius
	radius = (sqrtf((pow(Bottom, 2)) + (pow(Right, 2))) / 2);
	fprintf_s(gpFile, "Radius = %f\n\n", radius);

	//bind with vao
	glBindVertexArray(vao_circumscribed_rectangle);

	glDrawArrays(GL_LINES, 0, 2);
	glDrawArrays(GL_LINES, 2, 2);
	glDrawArrays(GL_LINES, 4, 2);
	glDrawArrays(GL_LINES, 6, 2);
	
	//unbind vao
	glBindVertexArray(0);

	//bind with vao
	glBindVertexArray(vao_circumscribed_rectangle_circle);
	for (GLfloat angle = 0.0f; angle < (2.0f * M_PI); angle = angle + 0.01f)
	{
		circleVertices[0] = ((cosf(angle) * radius));
		circleVertices[1] = ((sinf(angle) * radius));
		circleVertices[2] = 0.0f;

		//vertices
		glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_position_circle);
		glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		//draw scene
		glPointSize(2.0f);
		glDrawArrays(GL_POINTS, 0, 1);
	}

	//unbind vao
	glBindVertexArray(0);

}

void oglUninitialise(void)
{
	fprintf_s(gpFile, "\nIn OGLUninitialise\n");

	//code
	if (vbo_circumscribed_circle_color_triangle)
	{
		glDeleteBuffers(1, &vbo_circumscribed_circle_color_triangle);
		vbo_circumscribed_circle_color_triangle = 0;
	}
	if (vbo_circumscribed_circle_color)
	{
		glDeleteBuffers(1, &vbo_circumscribed_circle_color);
		vbo_circumscribed_circle_color = 0;
	}

	if (vbo_red_line_position)
	{
		glDeleteBuffers(1, &vbo_red_line_position);
		vbo_red_line_position = 0;
	}
	if (vbo_red_line_color)
	{
		glDeleteBuffers(1, &vbo_red_line_color);
		vbo_red_line_color = 0;
	}

	if (vbo_green_line_position)
	{
		glDeleteBuffers(1, &vbo_green_line_position);
		vbo_green_line_position = 0;
	}
	if (vbo_green_line_color)
	{
		glDeleteBuffers(1, &vbo_green_line_color);
		vbo_green_line_color = 0;
	}

	if (vbo_blue_line_position)
	{
		glDeleteBuffers(1, &vbo_blue_line_position);
		vbo_blue_line_position = 0;
	}
	

	if (vao_red)
	{
		glDeleteVertexArrays(1, &vao_red);
		vao_red = 0;
	}
	if (vao_green)
	{
		glDeleteVertexArrays(1, &vao_green);
		vao_green = 0;
	}
	if (vao_blue)
	{
		glDeleteVertexArrays(1, &vao_blue);
		vao_blue = 0;
	}


	if (vao_circumscribed_circle)
	{
		glDeleteVertexArrays(1, &vao_circumscribed_circle);
		vao_circumscribed_circle = 0;
	}
	if (vao_circumscribed_circle_triangle)
	{
		glDeleteVertexArrays(1, &vao_circumscribed_circle_triangle);
		vao_circumscribed_circle_triangle = 0;
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
