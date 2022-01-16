//Standard Header Files
#include <Windows.h>
#include <stdio.h>

//OpenGL Related Header Files
#include <C:\Libs\glew-2.1.0\include\GL\glew.h>
#include <GL/GL.h>
#include "vmath.h"

//Library Function
#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "C:\\Libs\\glew-2.1.0\\lib\\Release\\Win32\\glew32.lib")
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

struct buffers
{
	GLuint vao;
	GLuint vbo_position;
	GLuint vbo_color;
}one, two, three, four, five, six;

GLuint mvpUniform;
vmath::mat4 perspectiveProjectionMatrix;

//declaration of metrices
vmath::mat4 modelViewMatrix;
vmath::mat4 modelViewProjectionMatrix;

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
		TEXT("Perspective Yellow Triangl"),
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
		"in vec4 vCOlor;" \
		"uniform mat4 u_mvp_matrix;" \
		"out vec4 out_color;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_color = vCOlor;" \
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
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_COLOR, "vCOlor");

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

	/* vertices */
	const GLfloat firstDesign_vertices[] = 
	{
		//First Row
		-1.7f, 0.9f, 0.0f, 
		-1.5f, 0.9f, 0.0f, 
		-1.3f, 0.9f, 0.0f, 
		-1.1f, 0.9f, 0.0f, 

		//Second Row
		-1.7f, 0.7f, 0.0f, 
		-1.5f, 0.7f, 0.0f, 
		-1.3f, 0.7f, 0.0f, 
		-1.1f, 0.7f, 0.0f, 

		//Third Row
		-1.7f, 0.5f, 0.0f, 
		-1.5f, 0.5f, 0.0f, 
		-1.3f, 0.5f, 0.0f, 
		-1.1f, 0.5f, 0.0f, 

		//Fourth Row
		-1.7f, 0.3f, 0.0f, 
		-1.5f, 0.3f, 0.0f, 
		-1.3f, 0.3f, 0.0f, 
		-1.1f, 0.3f, 0.0f
	};

	const GLfloat secondDesign_vertice[] = 
	{
		//1st Vertical Line
		-0.6f, 0.9f, 0.0f, 
		-0.6f, 0.3f, 0.0f, 
		//2nd Vertical Line
		-0.4f, 0.9f, 0.0f, 
		-0.4f, 0.3f, 0.0f, 
		//3rd Vertical Line
		-0.2f, 0.9f, 0.0f, 
		-0.2f, 0.3f, 0.0f, 
		
		//1st Horizontal Line
		-0.6f, 0.9f, 0.0f, 
		-0.0f, 0.9f, 0.0f, 

		//2nd Horizontal Line
		-0.6f, 0.7f, 0.0f, 
		-0.0f, 0.7f, 0.0f, 
		
		//3rd Horizontal Line
		-0.6f, 0.5f, 0.0f, 
		-0.0f, 0.5f, 0.0f, 

		//1st Olique Line
		-0.6f, 0.7f, 0.0f, 
		-0.4f, 0.9, 0.0f, 

		//2nd Olique Line
		-0.6f, 0.5f, 0.0f, 
		-0.2f, 0.9, 0.0f, 

		//3rd Olique Line
		-0.6f, 0.3f, 0.0f, 
		-0.0f, 0.9f, 0.0f, 

		//4th Olique Line
		-0.4f, 0.3f, 0.0f, 
		-0.0f, 0.7f, 0.0f, 

		-0.2f, 0.3f, 0.0f, 
		-0.0f, 0.5f, 0.0f
	};

	const GLfloat thirdDesign_vertices[] =
	{
		//1st Vertical Line
		0.3f, 0.9f, 0.0f,
		0.3f, 0.3f, 0.0f,
		//2nd Vertical Line
		0.5f, 0.9f, 0.0f,
		0.5f, 0.3f, 0.0f,
		//3rd Vertical Line
		0.7f, 0.9f, 0.0f,
		0.7f, 0.3f, 0.0f,
		//4th Vertical Line
		0.9f, 0.9f, 0.0f,
		0.9f, 0.3f, 0.0f,
		//1st Horizontal Line
		0.3f, 0.9f, 0.0f,
		0.9f, 0.9f, 0.0f,
		//2nd Horizontal Line
		0.3f, 0.7f, 0.0f,
		0.9f, 0.7f, 0.0f,
		//3rd Horizontal Line
		0.3f, 0.5f, 0.0f,
		0.9f, 0.5f, 0.0f,
		//4th Horizontal Line
		0.3f, 0.3f, 0.0f,
		0.9f, 0.3f, 0.0f
	};

	const GLfloat fourthDesign_vertices[] = 
	{
		//4th Row
		-1.7f, -0.9f, 0.0f, 
		-1.1f, -0.9f, 0.0f, 
		//3rd Row
		-1.7f, -0.7f, 0.0f, 
		-1.1f, -0.7f, 0.0f, 
		//2nd Row
		-1.7f, -0.5f, 0.0f, 
		-1.1f, -0.5f, 0.0f, 
		//1st Row
		-1.7f, -0.3f, 0.0f, 
		-1.1f, -0.3f, 0.0f, 

		//4th column
		-1.7f, -0.9f, 0.0f, 
		-1.7f, -0.3f, 0.0f, 
		//3rd Column
		-1.5f, -0.9f, 0.0f, 
		-1.5f, -0.3f, 0.0f, 
		//2nd Column
		-1.3f, -0.9f, 0.0f, 
		-1.3f, -0.3f, 0.0f, 
		//1st Column
		-1.1f, -0.9f, 0.0f, 
		-1.1f, -0.3f, 0.0f, 

		//1st Olique Line
		-1.7f, -0.5f, 0.0f, 
		-1.5f, -0.3f, 0.0f, 
		//2nd Olique Line
		-1.7f, -0.7f, 0.0f, 
		-1.3f, -0.3f, 0.0f, 
		//3rd Olique Line
		-1.7f, -0.9f, 0.0f, 
		-1.1f, -0.3f, 0.0f, 
		//4th Olique Line
		-1.5f, -0.9f, 0.0f, 
		-1.1f, -0.5f, 0.0f, 
		//5th Olique Line
		-1.3f, -0.9f, 0.0f, 
		-1.1f, -0.7f, 0.0f
	};

	const GLfloat fifthDesign_vertices[] = 
	{
		//4th Row
		-0.6f, -0.9f, 0.0f,
		-0.0f, -0.9f, 0.0f,
		//1st Row
		-0.6f, -0.3f, 0.0f,
		-0.0f, -0.3f, 0.0f,

		//4th column
		-0.6f, -0.9f, 0.0f,
		-0.6f, -0.3f, 0.0f,
		//1st Column
		0.0f, -0.9f, 0.0f,
		0.0f, -0.3f, 0.0f,

		//Ray
		-0.6f, -0.3f, 0.0f,
		0.0f, -0.5f, 0.0f,

		-0.6f, -0.3f, 0.0f,
		0.0f, -0.7f, 0.0f,

		-0.6f, -0.3f, 0.0f,
		0.0f, -0.9f, 0.0f,

		-0.6f, -0.3f, 0.0f,
		-0.4f, -0.9f, 0.0f,

		-0.6f, -0.3f, 0.0f,
		-0.2f, -0.9f, 0.0f
	};

	const GLfloat sixthDesign_vertices[] = 
	{
		//first quad
		0.5f, -0.3f, 0.0f, 
		0.3f, -0.3f, 0.0f, 
		0.3f, -0.9f, 0.0f, 
		0.5f, -0.9f, 0.0f, 

		//second quad
		0.7f, -0.3f, 0.0f, 
		0.5f, -0.3f, 0.0f, 
		0.5f, -0.9f, 0.0f, 
		0.7f, -0.9f, 0.0f, 

		//third quad
		0.9f, -0.3f, 0.0f,
		0.7f, -0.3f, 0.0f,
		0.7f, -0.9f, 0.0f,
		0.9f, -0.9f, 0.0f,

		//vertical line 1
		0.5f, -0.3f, 0.0f,
		0.5f, -0.9f, 0.0f,

		//vertical line 2
		0.7f, -0.3f, 0.0f,
		0.7f, -0.9f, 0.0f,

		//Horizontal Line 1
		0.3f, -0.5f, 0.0f,
		0.9f, -0.5f, 0.0f,

		//Horizontal Line 1
		0.3f, -0.7f, 0.0f,
		0.9f, -0.7f, 0.0f
	};	

	const GLfloat sixthDesign_color[] = 
	{
		//first quad
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,

		//second quad
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,

		//third quad
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,

		//vertical line 1
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		//vertical line 2
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		//Horizontal Line 1
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		//Horizontal Line 1
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f
	};


	/* First Design */

	//generate and bind vao
	glGenVertexArrays(1, &one.vao);
	glBindVertexArray(one.vao);

	//generate and bind vbo
	glGenBuffers(1, &one.vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, one.vbo_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(firstDesign_vertices), firstDesign_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	//unbind vbo_position
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

	//unbind vao
	glBindVertexArray(0);

	/* Second Design */

	//generate and bind vao
	glGenVertexArrays(1, &two.vao);
	glBindVertexArray(two.vao);

	//generate and bind vbo_position
	glGenBuffers(1, &two.vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, two.vbo_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(secondDesign_vertice), secondDesign_vertice, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	//unbind vbo_position
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

	//unbind vao
	glBindVertexArray(0);

	/* Third Design */

	//generate and bind vao
	glGenVertexArrays(1, &three.vao);
	glBindVertexArray(three.vao);

	//generate and bind vbo_position
	glGenBuffers(1, &three.vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, three.vbo_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(thirdDesign_vertices), thirdDesign_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	//unbind vbo_position
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

	//unbind vao
	glBindVertexArray(0);

	/* Fourth Design */

	//generate and bind vao
	glGenVertexArrays(1, &four.vao);
	glBindVertexArray(four.vao);

	//generate and bind vbo_position
	glGenBuffers(1, &four.vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, four.vbo_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(fourthDesign_vertices), fourthDesign_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	//unbind vbo_position
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

	//unbind vao
	glBindVertexArray(0);

	/* Fifth Design */

	//generate and bind vao
	glGenVertexArrays(1, &five.vao);
	glBindVertexArray(five.vao);

	//generate and bind vbo_position
	glGenBuffers(1, &five.vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, five.vbo_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(fifthDesign_vertices), fifthDesign_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	//unbind vbo_position
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

	//unbind vao
	glBindVertexArray(0);

	/* Sixth Design */

	//generate and bind vao
	glGenVertexArrays(1, &six.vao);
	glBindVertexArray(six.vao);

	//generate and bind vbo_position
	glGenBuffers(1, &six.vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, six.vbo_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(sixthDesign_vertices), sixthDesign_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	//unbind vbo_position
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//generate and bind vbo_color
	glGenBuffers(1, &six.vbo_color);
	glBindBuffer(GL_ARRAY_BUFFER, six.vbo_color);

	glBufferData(GL_ARRAY_BUFFER, sizeof(sixthDesign_color), sixthDesign_color, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);

	//unbind vbo_color
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//unbind vao
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
	//Function declaration
	void DottedSquare(void);
	void Design_two(void);
	void Square(void);
	void SquareAndObliqueLine(void);
	void SquareAndRay(void);
	void RGB_Quads(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderProgramObject);
	
	//declaration of metrices
	vmath::mat4 modelViewMatrix;
	vmath::mat4 modelViewProjectionMatrix;
	vmath::mat4 translationMatrix;

	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	
	//do necessary transformations here
	modelViewMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

	//do necessary matrix multiplication
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	DottedSquare();
	SquareAndObliqueLine();

	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	
	//do necessary transformations here
	modelViewMatrix = vmath::translate(0.2f, 0.0f, -3.0f);

	//do necessary matrix multiplication
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	Design_two();
	SquareAndRay();

	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	
	//do necessary transformations here
	modelViewMatrix = vmath::translate(0.6f, 0.0f, -3.0f);

	//do necessary matrix multiplication
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	Square();
	RGB_Quads();

	//unuse program
	glUseProgram(0);
	SwapBuffers(ghdc);
}
void oglUpdate(void)
{
	//code
}
void oglUninitialise(void)
{
	fprintf_s(gpFile, "\nIn OGLUninitialise\n");

	//code
	if (one.vbo_position)
	{
		glDeleteBuffers(1, &one.vbo_position);
		one.vbo_position = 0;
	}
	if (one.vao)
	{
		glDeleteVertexArrays(1, &one.vao);
		one.vao = 0;
	}

	if (two.vbo_position)
	{
		glDeleteBuffers(1, &two.vbo_position);
		two.vbo_position = 0;
	}
	if (two.vao)
	{
		glDeleteVertexArrays(1, &two.vao);
		two.vao = 0;
	}

	if (three.vbo_position)
	{
		glDeleteBuffers(1, &three.vbo_position);
		three.vbo_position = 0;
	}
	if (three.vao)
	{
		glDeleteVertexArrays(1, &three.vao);
		three.vao = 0;
	}

	if (four.vbo_position)
	{
		glDeleteBuffers(1, &four.vbo_position);
		four.vbo_position = 0;
	}
	if (four.vao)
	{
		glDeleteVertexArrays(1, &four.vao);
		four.vao = 0;
	}

	if (five.vbo_position)
	{
		glDeleteBuffers(1, &five.vbo_position);
		five.vbo_position = 0;
	}
	if (five.vao)
	{
		glDeleteVertexArrays(1, &five.vao);
		five.vao = 0;
	}

	if (six.vbo_position)
	{
		glDeleteBuffers(1, &six.vbo_position);
		six.vbo_position = 0;
	}
	if (six.vbo_color)
	{
		glDeleteBuffers(1, &six.vbo_color);
		six.vbo_color = 0;
	}
	if (six.vao)
	{
		glDeleteVertexArrays(1, &six.vao);
		six.vao = 0;
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

void DottedSquare(void)
{
	glPointSize(2.0f);
	glBindVertexArray(one.vao);

	//First Row
	glDrawArrays(GL_POINTS, 0, 1);
	glDrawArrays(GL_POINTS, 1, 1);
	glDrawArrays(GL_POINTS, 2, 1);
	glDrawArrays(GL_POINTS, 3, 1);

	//Second Row
	glDrawArrays(GL_POINTS, 4, 1);
	glDrawArrays(GL_POINTS, 5, 1);
	glDrawArrays(GL_POINTS, 6, 1);
	glDrawArrays(GL_POINTS, 7, 1);

	//Third Row
	glDrawArrays(GL_POINTS, 8, 1);
	glDrawArrays(GL_POINTS, 9, 1);
	glDrawArrays(GL_POINTS, 10, 1);
	glDrawArrays(GL_POINTS, 11, 1);

	//Fourth Row
	glDrawArrays(GL_POINTS, 12, 1);
	glDrawArrays(GL_POINTS, 13, 1);
	glDrawArrays(GL_POINTS, 14, 1);
	glDrawArrays(GL_POINTS, 15, 1);

	glBindVertexArray(0);
}

void Design_two(void)
{

	glBindVertexArray(two.vao);
	
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
	
	

	glBindVertexArray(0);

}

void Square(void)
{


	glBindVertexArray(three.vao);

	//1st Vertical Line
	glDrawArrays(GL_LINES, 0, 2);
	//2nd Vertical Line
	glDrawArrays(GL_LINES, 2, 2);
	//3rd Vertical Line
	glDrawArrays(GL_LINES, 4, 2);
	//4th Vertical Line
	glDrawArrays(GL_LINES, 6, 2);

	//1st Horizontal Line
	glDrawArrays(GL_LINES, 8, 2);
	//2nd Horizontal Line
	glDrawArrays(GL_LINES, 10, 2);
	//3rd Horizontal Line
	glDrawArrays(GL_LINES, 12, 2);
	//4th Horizontal Line
	glDrawArrays(GL_LINES, 14, 2);

	glBindVertexArray(0);
}

void SquareAndObliqueLine(void)
{

	glBindVertexArray(four.vao);

	glDrawArrays(GL_LINES, 0, 2);//4th Row
	glDrawArrays(GL_LINES, 2, 2);//3rd Row
	glDrawArrays(GL_LINES, 4, 2);//2nd Row
	glDrawArrays(GL_LINES, 6, 2);//1st Row
	
	glDrawArrays(GL_LINES, 8, 2);//4th column
	glDrawArrays(GL_LINES, 10, 2);//3rd column
	glDrawArrays(GL_LINES, 12, 2);//2nd column
	glDrawArrays(GL_LINES, 14, 2);//1st column

	glDrawArrays(GL_LINES, 16, 2);//1st OliqueLine
	glDrawArrays(GL_LINES, 18, 2);//2nd OliqueLine
	glDrawArrays(GL_LINES, 20, 2);//3rd OliqueLine
	glDrawArrays(GL_LINES, 22, 2);//4th OliqueLine
	glDrawArrays(GL_LINES, 24, 2);//5th OliqueLine

	glBindVertexArray(0);
}

void SquareAndRay(void)
{
	glBindVertexArray(five.vao);

	glDrawArrays(GL_LINES, 0, 2);//4th Row
	glDrawArrays(GL_LINES, 2, 2);//1st Row
	glDrawArrays(GL_LINES, 4, 2);//4th column
	glDrawArrays(GL_LINES, 6, 2);//1st Column
	
	//ray
	glDrawArrays(GL_LINES, 8, 2);
	glDrawArrays(GL_LINES, 10, 2);
	glDrawArrays(GL_LINES, 12, 2);
	glDrawArrays(GL_LINES, 14, 2);
	glDrawArrays(GL_LINES, 16, 2);

	glBindVertexArray(0);
}

void RGB_Quads(void)
{
	glLineWidth(3.0f);
	glBindVertexArray(six.vao);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);

	glDrawArrays(GL_LINES, 12, 2);//vertical line 1
	glDrawArrays(GL_LINES, 14, 2);//vertical line 2
	glDrawArrays(GL_LINES, 16, 2);//Horizontal Line 1
	glDrawArrays(GL_LINES, 18, 2);//Horizontal Line 2

	glBindVertexArray(0);
}
