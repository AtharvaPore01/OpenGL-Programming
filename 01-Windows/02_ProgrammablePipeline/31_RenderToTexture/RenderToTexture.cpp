//Standard Header Files
#include <Windows.h>
#include <stdio.h>

//OpenGL Related Header Files
#include <C:\Libs\glew-2.1.0\include\GL\glew.h>
#include <GL/GL.h>
#include "vmath.h"
#include "Resource.h"	

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

GLuint vao;
GLuint vbo_vertices;
GLuint vbo_normal;
GLuint vbo_texcoord;
GLuint element_buffer_vertices;

GLuint vao_cube;
GLuint vbo_cube_position;
GLuint vbo_cube_texture;

GLuint texture_marble;
GLuint target_texture;
GLuint target_texture_width = 256;
GLuint target_texture_height = 256;

GLuint fbo;		//frame buffer object
GLuint rbo;		//render buffer object

GLuint samplerUniform;
GLuint model_uniform;
GLuint view_uniform;
GLuint projection_uniform;

vmath::mat4 perspectiveProjectionMatrix;

//model loading variables
struct vec_int
{
	int *p;
	int size;
};

struct vec_float
{
	float *pf;
	int size;
};

#define BUFFER_SIZE 1024
char buffer[BUFFER_SIZE];

FILE *gpMeshFile = NULL;

struct vec_float *gpVertices = NULL;
struct vec_float *gpTexture = NULL;
struct vec_float *gpNormal = NULL;

struct vec_float *gp_sorted_vertices = NULL;
struct vec_float *gp_sorted_texture = NULL;
struct vec_float *gp_sorted_normal = NULL;

struct vec_int *gp_indices_vertices = NULL;
struct vec_int *gp_indices_texture = NULL;
struct vec_int *gp_indices_normal = NULL;

GLuint windowWidth;
GLuint windowHeight;

//Rotation variables
GLfloat rotation_angle = 0.0f;
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
		TEXT("White Cube : Template"),
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
			oglToggleFullScreen();
			break;
		}
		break;


	case WM_SIZE:
		oglResize(LOWORD(lParam), HIWORD(lParam));
		windowWidth = LOWORD(lParam);
		windowHeight = HIWORD(lParam);
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

bool oglLoadTexture(GLuint *texture, TCHAR imageResourceID[])
{
	//variable declaration
	HBITMAP hBitmap;
	BITMAP bmp;
	bool bStatus = false;

	//code
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL),
		imageResourceID,
		IMAGE_BITMAP,
		0,
		0,
		LR_CREATEDIBSECTION);

	if (hBitmap)
	{
		bStatus = true;

		//fill the bitmap structure 
		GetObject(hBitmap, sizeof(BITMAP), &bmp);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		//generate texture
		glGenTextures(1, texture);

		//bind texture
		glBindTexture(GL_TEXTURE_2D, *texture);

		//set parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glTexImage2D(GL_TEXTURE_2D,
			0,
			GL_RGB,
			bmp.bmWidth,
			bmp.bmHeight,
			0,
			GL_BGR, //GL_BGR_EXT,
			GL_UNSIGNED_BYTE,
			bmp.bmBits);

		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);
		DeleteObject(hBitmap);
	}
	return(bStatus);
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
	void oglLoadMesh(void);

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
		"in vec2 vTexCoord;" \
		"out vec2 out_texcoord;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"out_texcoord = vTexCoord;" \
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
		"in vec2 out_texcoord;" \
		"out vec4 FragColor;" \
		"uniform sampler2D u_sampler;" \
		"void main(void)" \
		"{" \
		"FragColor = texture(u_sampler, out_texcoord);" \
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
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_TEXCOODR_0, "vTexCoord");

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

	//load mesh
	oglLoadMesh();

	//post linking retriving uniform location
	model_uniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
	view_uniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
	projection_uniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
	samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");

	GLfloat cubeVertices[] =
	{
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,

		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, -1.0f
	};
	const GLfloat cubeTexcoord[] = 
	{
		0.0f, 1.0f,
		0.0f, 0.0f, 
		1.0f, 0.0f,
		1.0f, 1.0f,

		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f, 
		0.0f, 1.0f,

		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,

		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f, 
		0.0f, 0.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f
	};

	//RECTANGLE
	for (int i = 0; i < 72; i++)
	{
		if (cubeVertices[i] == -1.0f)
		{
			cubeVertices[i] = cubeVertices[i] + 0.25f;
		}
		else if (cubeVertices[i] == 1.0f)
		{
			cubeVertices[i] = cubeVertices[i] - 0.25f;
		}
	}
	
	glGenVertexArrays(1, &vao_cube);
	glBindVertexArray(vao_cube);
	//position
	glGenBuffers(1, &vbo_cube_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//texture
	glGenBuffers(1, &vbo_cube_texture);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_texture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeTexcoord), cubeTexcoord, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOODR_0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOODR_0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//create vao and vbo
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	//position
	glGenBuffers(1, &vbo_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);

	glBufferData(	GL_ARRAY_BUFFER, 
					(gp_sorted_vertices->size) * sizeof(float), 
					gp_sorted_vertices->pf, 
					GL_STATIC_DRAW);

	glVertexAttribPointer(	AMC_ATTRIBUTE_POSITION, 
							3, 
							GL_FLOAT, 
							GL_FALSE, 
							0, 
							NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//normal
	glGenBuffers(1, &vbo_normal);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);

	glBufferData(	GL_ARRAY_BUFFER, 
					(gp_sorted_normal->size) * sizeof(float), 
					gp_sorted_normal->pf, 
					GL_STATIC_DRAW);

	glVertexAttribPointer(	AMC_ATTRIBUTE_NORMAL, 
							3, 
							GL_FLOAT, 
							GL_FALSE, 
							0, 
							NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//texcoord
	glGenBuffers(1, &vbo_texcoord);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoord);

	glBufferData(	GL_ARRAY_BUFFER, 
					(gp_sorted_texture->size) * sizeof(float), 
					gp_sorted_texture->pf, 
					GL_STATIC_DRAW);

	glVertexAttribPointer(	AMC_ATTRIBUTE_TEXCOODR_0, 
							2, 
							GL_FLOAT, 
							GL_FALSE, 
							0, 
							NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOODR_0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//element buffer
	
	//vertices
	glGenBuffers(1, &element_buffer_vertices);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_vertices);
	glBufferData(	GL_ELEMENT_ARRAY_BUFFER,
					gp_indices_vertices->size * sizeof(int),
					gp_indices_vertices->p,
					GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Create a texture to render to
	target_texture_width = 256;
	target_texture_height = 256;
	glGenTextures(1, &target_texture);
	glBindTexture(GL_TEXTURE_2D, target_texture);
	glTexImage2D(	GL_TEXTURE_2D, 
					0, 
					GL_RGBA, 
					target_texture_width, 
					target_texture_height, 
					0, 
					GL_RGBA, 
					GL_UNSIGNED_BYTE, 
					NULL);
	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // we clamp to the edge as the blur filter would otherwise sample repeated texture values!
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    //Create and bind Frame buffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(	GL_FRAMEBUFFER, 
    						GL_COLOR_ATTACHMENT0, 
    						GL_TEXTURE_2D, 
    						target_texture, 
    						0);
   	
   	glGenRenderbuffers(1, &rbo);

   	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	
	glRenderbufferStorage(	GL_RENDERBUFFER,
							GL_DEPTH_COMPONENT16,
							target_texture_width,
							target_texture_height);

	glFramebufferRenderbuffer(	GL_FRAMEBUFFER,
								GL_DEPTH_ATTACHMENT,
								GL_RENDERBUFFER,
								rbo);


    glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//clear the window
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	
	//disable culling
	glDisable(GL_CULL_FACE);
	
	//depth
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	//texture
	glEnable(GL_TEXTURE_2D);
	oglLoadTexture(&texture_marble, MAKEINTRESOURCE(IDMITMAP_MARBLE));

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
	
}
void oglDisplay(void)
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glViewport(0, 0, (GLsizei)target_texture_width, (GLsizei)target_texture_height);

	glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderProgramObject);

	//declaration of metrices
	vmath::mat4 modelMatrix;
	vmath::mat4 viewMatrix;
	vmath::mat4 projectionMatrix;
	vmath::mat4 translationMatrix;
	vmath::mat4 rotationMatrix;

	/* teapot code */ 
	//init above metrices to identity
	modelMatrix = vmath::mat4::identity();
	viewMatrix = vmath::mat4::identity();
	projectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();

	//do necessary transformations here
	translationMatrix = vmath::translate(0.0f, -1.5f, -10.0f);
	rotationMatrix = vmath::rotate(rotation_angle, 0.0f, 1.0f, 0.0f);
	perspectiveProjectionMatrix = vmath::perspective(	45.0f, 
														((GLfloat)target_texture_width / (GLfloat)target_texture_height), 
														0.1f, 
														100.0f);
	//do necessary matrix multiplication
	modelMatrix = modelMatrix * translationMatrix;
	modelMatrix = modelMatrix * rotationMatrix;
	projectionMatrix = perspectiveProjectionMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(model_uniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(view_uniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(projection_uniform, 1, GL_FALSE, projectionMatrix);

	//active texture
	glActiveTexture(GL_TEXTURE0);

	//bind with texture 
	glBindTexture(GL_TEXTURE_2D, texture_marble);

	//push in fragment shader
	glUniform1i(samplerUniform, 0);

	glBindVertexArray(vao);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_vertices);
	glDrawElements(	GL_TRIANGLES,
					(gp_indices_vertices->size),
					GL_UNSIGNED_INT,
					NULL);
	
	glBindVertexArray(0);
	//bind with texture 
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/* Cube Code */
	glViewport(0, 0, (GLsizei)windowWidth, (GLsizei)windowHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderProgramObject);
	//init above metrices to identity
	modelMatrix = vmath::mat4::identity();
	viewMatrix = vmath::mat4::identity();
	projectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();

	//do necessary transformations here
	translationMatrix = vmath::translate(0.0f, 0.0f, -4.0f);
	rotationMatrix = vmath::rotate(rotation_angle, rotation_angle, rotation_angle);

	perspectiveProjectionMatrix = vmath::perspective(45.0f, ((GLfloat)windowWidth / (GLfloat)windowHeight), 0.1f, 100.0f);
	
	//do necessary matrix multiplication
	modelMatrix = modelMatrix * translationMatrix;
	modelMatrix = modelMatrix * rotationMatrix;
	projectionMatrix = perspectiveProjectionMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(model_uniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(view_uniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(projection_uniform, 1, GL_FALSE, projectionMatrix);

	//acitve texture
	glActiveTexture(GL_TEXTURE0);

	//bind texture
	glBindTexture(GL_TEXTURE_2D, target_texture);

	//push in fragment
	glUniform1i(samplerUniform, 0);

	//bind with vao
	glBindVertexArray(vao_cube);

	//draw scene
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	//unbind vao
	glBindVertexArray(0);

	glBindTexture(GL_TEXTURE_2D, 0);

	//unuse program
	glUseProgram(0);
	SwapBuffers(ghdc);
}
void oglUpdate(void)
{
	//code
	rotation_angle = rotation_angle + 1.0f;
	if (rotation_angle >= 360.0f)
	{
		rotation_angle = 0.0f;
	}
}
void oglUninitialise(void)
{
	//function declaration
	int destroy_vec_int(struct vec_int *p_vec_int); 
	int destroy_vec_float(struct vec_float *p_vec_float); 
	//code

	if (vbo_cube_position)
	{
		glDeleteBuffers(1, &vbo_cube_position);
		vbo_cube_position = 0;
	}
	if (vbo_cube_texture)
	{
		glDeleteBuffers(1, &vbo_cube_texture);
		vbo_cube_texture = 0;
	}
	if (vao_cube)
	{
		glDeleteVertexArrays(1, &vao_cube);
		vao_cube = 0;
	}

	if (vbo_vertices)
	{
		glDeleteBuffers(1, &vbo_vertices);
		vbo_vertices = 0;
	}
	if (vbo_texcoord)
	{
		glDeleteBuffers(1, &vbo_texcoord);
		vbo_texcoord = 0;
	}
	if (vbo_normal)
	{
		glDeleteBuffers(1, &vbo_normal);
		vbo_normal = 0;
	}
	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

	if(fbo)
	{
		glDeleteFramebuffers(1, &fbo);
		fbo = 0;
	}

	if(rbo)
	{
		glDeleteRenderbuffers(1, &rbo);
		rbo = 0;
	}

	if(element_buffer_vertices)
	{
		glDeleteBuffers(1, &element_buffer_vertices); 
		element_buffer_vertices = 0; 
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

	if(gpVertices)
	{
		destroy_vec_float(gpVertices);
		gpVertices = NULL;
	}

	if(gpTexture)
	{
		destroy_vec_float(gpTexture);
		gpTexture = NULL;
	}

	if(gpNormal)
	{
		destroy_vec_float(gpNormal);
		gpNormal = NULL;
	}

	if(gp_sorted_vertices)
	{
		destroy_vec_float(gp_sorted_vertices);
		gp_sorted_vertices = NULL;
	}

	if(gp_sorted_texture)
	{
		destroy_vec_float(gp_sorted_texture);
		gp_sorted_texture = NULL;
	}

	if(gp_sorted_normal)
	{
		destroy_vec_float(gp_sorted_normal);
		gp_sorted_normal = NULL;
	}

	if(gp_indices_vertices)
	{
		destroy_vec_int(gp_indices_vertices);
		gp_indices_vertices = NULL;
	}

	if(gp_indices_texture)
	{
		destroy_vec_int(gp_indices_texture);
		gp_indices_texture = NULL;
	}

	if(gp_indices_normal)
	{
		destroy_vec_int(gp_indices_normal);
		gp_indices_normal = NULL;
	}


	if (gpFile)
	{
		fprintf_s(gpFile, "Log File Is Closed Successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}


void oglLoadMesh(void)
{
	//function declarations
	struct vec_int *create_vec_int(void);
	struct vec_float *create_vec_float(void);
	int push_back_vec_int(struct vec_int *p_vec_int, int data);
	int push_back_vec_float(struct vec_float *p_vec_float, float data);
	void show_vec_float(struct vec_float *p_vec_float);
	void show_vec_int(struct vec_int *p_vec_int);
	int destroy_vec_float(struct vec_float *p_vec_float);

	//variable declaration
	char *space = " ";
	char *slash = "/";
	char *first_token = NULL;
	char *token = NULL;
	char *f_enteries[3] = { NULL, NULL, NULL };
	
	int nr_vert_cords = 0;
	int nr_tex_cords = 0;
	int nr_norm_cords = 0;
	int nr_faces = 0;

	int i, vi;

	gpMeshFile = fopen("teapot.obj", "r");
	if(gpMeshFile == NULL)
	{
		fprintf(stderr, "error : unable to open obj file\n");
		exit(EXIT_FAILURE);
	}

	gpVertices 	= 	create_vec_float();
	gpTexture 	= 	create_vec_float();
	gpNormal 	=	create_vec_float();

	gp_indices_vertices 	=	create_vec_int();
	gp_indices_texture		=	create_vec_int();
	gp_indices_normal	 	=	create_vec_int();

	while(fgets(buffer, BUFFER_SIZE, gpMeshFile) != NULL)
	{
		first_token = strtok(buffer, space);
		if(strcmp(first_token, "v") == 0)
		{
			nr_vert_cords++;
			while((token = strtok(NULL, space)) != NULL)
			{
				push_back_vec_float(gpVertices, atof(token));
			}

		}

		else if(strcmp(first_token, "vt") == 0)
		{
			nr_tex_cords++;
			while((token = strtok(NULL, space)) != NULL)
			{
				push_back_vec_float(gpTexture, atof(token));
			}
			
		}

		else if(strcmp(first_token, "vn") == 0)
		{
			nr_norm_cords++;
			while((token = strtok(NULL, space)) != NULL)
			{
				push_back_vec_float(gpNormal, atof(token));
			}
			
		}

		else if(strcmp(first_token, "f") == 0)
		{
			nr_faces++;
			for(i = 0; i < 3; i++)
			{
				f_enteries[i] = strtok(NULL, space);
			}

			for(i = 0; i < 3; i++)
			{
				token = strtok(f_enteries[i], slash);
				push_back_vec_int(gp_indices_vertices, atoi(token) - 1);

				token = strtok(NULL, slash);
				push_back_vec_int(gp_indices_texture, atoi(token) - 1);

				token = strtok(NULL, slash);
				push_back_vec_int(gp_indices_normal, atoi(token) - 1);
			}			
		}
	}

	gp_sorted_vertices = create_vec_float();
	for(int i = 0; i < gp_indices_vertices->size; i++)
	{
		push_back_vec_float(gp_sorted_vertices, gpVertices->pf[i]);
	}

	gp_sorted_texture = create_vec_float();
	for(int i = 0; i < gp_indices_texture->size; i++)
	{
		push_back_vec_float(gp_sorted_texture, gpTexture->pf[i]);
	}

	gp_sorted_normal = create_vec_float();
	for(int i = 0; i < gp_indices_normal->size; i++)
	{
		push_back_vec_float(gp_sorted_normal, gpNormal->pf[i]);
	}

	fclose(gpMeshFile);
	gpMeshFile = NULL;
}

struct vec_int *create_vec_int(void)
{
	//code
	struct vec_int *p = (struct vec_int *)malloc(sizeof(struct vec_int));
	if(p != NULL)
	{
		memset(p, 0, sizeof(struct vec_int));
		return (p);
	}
	return(NULL);
}

struct vec_float *create_vec_float(void)
{
	//code
	struct vec_float *p = (struct vec_float *)malloc(sizeof(struct vec_float));
	if(p != NULL)
	{
		memset(p, 0, sizeof(struct vec_float));
		return (p);
	}
	return(NULL);
}
int push_back_vec_int(struct vec_int *p_vec_int, int data)
{
	//code
	p_vec_int->p = (int *)realloc(p_vec_int->p, (p_vec_int->size + 1) * sizeof(int));
	p_vec_int->size = p_vec_int->size + 1;
	p_vec_int->p[p_vec_int->size-1] = data;
	return (0);
}
int push_back_vec_float(struct vec_float *p_vec_float, float data)
{
	//code
	p_vec_float->pf = (float *)realloc(p_vec_float->pf, (p_vec_float->size + 1) * sizeof(float));
	p_vec_float->size = p_vec_float->size + 1;
	p_vec_float->pf[p_vec_float->size-1] = data;
	return (0);
}
void show_vec_float(struct vec_float *p_vec_float)
{
	//code
	int i = 0;
	for(i = 0; i < p_vec_float->size; i++)
	{
		fprintf(gpFile, "%f\n", p_vec_float->pf[i]);
	}
}
void show_vec_int(struct vec_int *p_vec_int)
{
	//code
	int i = 0;
	for(i = 0; i < p_vec_int->size; i++)
	{
		fprintf(gpFile, "%d\n", p_vec_int->p[i]);
	}
}
int destroy_vec_float(struct vec_float *p_vec_float)
{
	//code
	free(p_vec_float->pf);
	free(p_vec_float);
	p_vec_float = NULL;
	return (0);
}

int destroy_vec_int(struct vec_int *p_vec_int)
{
	//code
	free(p_vec_int->p);
	free(p_vec_int);
	p_vec_int = NULL;
	return (0);
}


