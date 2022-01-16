//Standard Header Files
#include <Windows.h>
#include <stdio.h>

//OpenGL Related Header Files
#include <C:\Libs\glew-2.1.0\include\GL\glew.h>
#include <gl/GL.h>
#include "vmath.h"
//cuda headers
#include <C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include\cuda_gl_interop.h>
#include <C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include\cuda.h>
#include <C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include\cuda_runtime.h>

#include "sineWave.h"

//Library Function
#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "C:\\Libs\\glew-2.1.0\\lib\\Release\\x64\\glew32.lib")
#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "kernel32.lib")

//libraries of cuda
#pragma comment (lib, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64\\cudart.lib")

//Macros
#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define X (GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH) / 2
#define Y (GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT) / 2

#define MESH_WIDTH	1024
#define MESH_HEIGHT	1024

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

GLuint vao;
GLuint vao_gpu;
GLuint vbo;
GLuint vbo_gpu;
GLuint mvpUniform;
vmath::mat4 perspectiveProjectionMatrix;

//cuda related global variables, macros
const int gMeshWidth = 512;
const int gMeshHeight = 512;

float pos[gMeshWidth][gMeshHeight][4];
float fAnimationTime = 0.0f;
cudaGraphicsResource *graphicResource = NULL;

bool bOnGPU = false;
cudaError_t cuError;

#define ARRAY_SIZE	gMeshWidth * gMeshHeight * 4

//void launchCudaKernel(float4 *, unsigned int, unsigned int, float);

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
		TEXT("CUDA OpenGL InterOp"),
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

		case 'C':
		case 'c':
			if (bOnGPU == true);
			{
				bOnGPU = false;
			}
			break;

		case 'G':
		case 'g':
			if (bOnGPU == false)
			{
				bOnGPU = true;
			}
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
	//variable declaration
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum result;
	static HWND hwnd;

	//cuda local variables
	int iCuDevCount = 0;

	//variables related to error checking
	GLint iShaderCompileStatus = 0;
	GLint iProgramLinkStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	//function declaration
	void oglResize(int, int);
	void oglUninitialise(void);

	//code

	//check whether machine has atleat single cuda device or not
	cuError = cudaGetDeviceCount(&iCuDevCount);
	fprintf_s(gpFile, "Device Count = %d\n", iCuDevCount);
	if (cuError != cudaSuccess)
	{
		fprintf_s(gpFile, "Error During Extraction Of Device Count.\nExitting...\n");
		oglUninitialise();
		exit(0);
	}
	else if (iCuDevCount == 0)
	{
		fprintf_s(gpFile, "There Is No Cuda Device.\nExitting...\n");
		oglUninitialise();
		exit(0);
	}
	else
	{
		//set 0th device
		cudaSetDevice(0);
	}

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
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
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
		"void main(void)" \
		"{" \
		"FragColor = vec4(1.0, 1.0, 0.0, 1.0);" \
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

	//zero out position array
	for (int i = 0; i < gMeshWidth; i++)
	{
		for (int j = 0; j < gMeshHeight; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				pos[i][j][k] = 0;
			}
		}
	}

	//create and bind vao 
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	//create and bind vbo
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(	GL_ARRAY_BUFFER, 
					ARRAY_SIZE * sizeof(float), 
					NULL, 
					GL_DYNAMIC_DRAW);
	glVertexAttribPointer(	AMC_ATTRIBUTE_POSITION, 
							4, 
							GL_FLOAT, 
							GL_FALSE, 
							0, 
							NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	//unbind vbo and vao
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	/* GPU */	
	//create and bind vao 
	glGenVertexArrays(1, &vao_gpu);
	glBindVertexArray(vao_gpu);

	//create and bind vbo
	glGenBuffers(1, &vbo_gpu);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu);

	glBufferData(	GL_ARRAY_BUFFER, 
					ARRAY_SIZE * sizeof(float), 
					NULL,
					GL_DYNAMIC_DRAW);
	glVertexAttribPointer(	AMC_ATTRIBUTE_POSITION, 
							4, 
							GL_FLOAT, 
							GL_FALSE, 
							0, 
							NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	cuError = cudaGraphicsGLRegisterBuffer(&graphicResource, vbo_gpu, cudaGraphicsMapFlagsWriteDiscard);
	
	if (cuError != cudaSuccess)
	{
		fprintf_s(gpFile, "Error During Regitration of OpenGL Buffer.\nExitting...\n");
		oglUninitialise();
		exit(0);
	}


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
	//variable declaration
	float4 *pPos = NULL;
	size_t byteCount = 0;

	//function declaration
	void launchCPUKernel(unsigned int, unsigned int, float);
	void oglUninitialise(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderProgramObject);

	//declaration of metrices
	vmath::mat4 modelViewMatrix;
	vmath::mat4 modelViewProjectionMatrix;

	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();

	//do necessary transformations here
	modelViewMatrix = vmath::translate(0.0f, 0.0f, -2.0f);

	//do necessary matrix multiplication
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	if (bOnGPU == true)
	{
		//map with resource
		cuError = cudaGraphicsMapResources(1, &graphicResource, 0);
		if (cuError != cudaSuccess)
		{
			fprintf_s(gpFile, "Display :: cudaGraphicsMapResources() Failed.\nExitting...\n");
			oglUninitialise();
			exit(0);
		}

		//map our array with graphics resource.
		cuError = cudaGraphicsResourceGetMappedPointer((void **)&pPos, &byteCount, graphicResource);
		if (cuError != cudaSuccess)
		{
			fprintf_s(gpFile, "Display :: cudaGraphicsResourceGetMappedPointer() Failed.\nExitting...\n");
			oglUninitialise();
			exit(0);
		}

		//lunch cuda kernel
		launchCudaKernel(pPos, gMeshWidth, gMeshHeight, fAnimationTime);

		//unmap resource
		cuError = cudaGraphicsUnmapResources(1, &graphicResource, 0);
		if (cuError != cudaSuccess)
		{
			fprintf_s(gpFile, "Display :: cudaGraphicsResourceGetMappedPointer() Failed.\nExitting...\n");
			oglUninitialise();
			exit(0);
		}

		glBindVertexArray(vao_gpu);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu);
		//draw scene
		glDrawArrays(GL_POINTS, 0, MESH_WIDTH * MESH_HEIGHT);
		glBindVertexArray(0);
	}

	else
	{
		launchCPUKernel(gMeshWidth, gMeshHeight, fAnimationTime);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(pos), pos, GL_DYNAMIC_DRAW);
		glBindVertexArray(vao);
		//draw scene
		glDrawArrays(GL_POINTS, 0, MESH_WIDTH * MESH_HEIGHT);
		glBindVertexArray(0);
	}
	
	//unuse program
	glUseProgram(0);
	SwapBuffers(ghdc);
	fAnimationTime = fAnimationTime + 0.01f;
}
void oglUpdate(void)
{
	//code
}

void launchCPUKernel(unsigned int width, unsigned int height, float fAnimTime)
{
	//variable declaration
	float u, v, frequency, w;

	//code
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				u = (GLfloat)i / (GLfloat)width;
				v = (GLfloat)j / (GLfloat)height;

				u = (u * 2.0f) - 1.0f;
				v = (v * 2.0f) - 1.0f;

				frequency = 4.0f;
				w = sinf((frequency * u + fAnimTime)) * cosf((frequency * v + fAnimTime)) * 0.5f;
				
				if (k == 0)
				{
					pos[i][j][k] = u;
				}
				if (k == 1)
				{
					pos[i][j][k] = w;
				}
				if (k == 2)
				{
					pos[i][j][k] = v;
				}
				if (k == 3)
				{
					pos[i][j][k] = 1.0f;
				}
			}
		}
	}
}

void oglUninitialise(void)
{
	fprintf_s(gpFile, "\nIn OGLUninitialise\n");

	//code
	if (vbo)
	{
		glDeleteBuffers(1, &vbo);
		vbo = 0;
	}
	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

	if (vbo_gpu)
	{
		glDeleteBuffers(1, &vbo_gpu);
		vbo_gpu = 0;
	}
	if (vao_gpu)
	{
		glDeleteVertexArrays(1, &vao_gpu);
		vao_gpu = 0;
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

	//release cuda graphics resource
	if (graphicResource != NULL)
	{
		cudaGraphicsUnregisterResource(graphicResource);
		graphicResource = NULL;
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
