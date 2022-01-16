//Standard Header Files
#include <Windows.h>
#include <stdio.h>

//OpenGL Related Header Files
#include <gl/glew.h>
#include <GL/GL.h>
#include "vmath.h"
#include "Sphere.h"


//Library Function
#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "glew32.lib")
#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "kernel32.lib")
#pragma comment (lib, "Sphere.lib")

//Macros
#define WIN_WIDTH 1366
#define WIN_HEIGHT 768
#define X (GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH) / 2
#define Y (GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT) / 2
#define RADIUS 100.0f;

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

GLuint vao_sphere;
GLuint vbo_sphere_position;
GLuint vbo_sphere_normal;
GLuint vbo_sphere_element;

GLuint model_uniform;
GLuint view_uniform;
GLuint projection_uniform;

GLuint La_uniform;
GLuint Ld_uniform;
GLuint Ls_uniform;
GLuint lightPosition_uniform;

GLuint Ka_uniform;
GLuint Kd_uniform;
GLuint Ks_uniform;
GLuint shininess_uniform;
GLuint LKeyPressed_Uniform;

vmath::mat4 perspectiveProjectionMatrix;

//Rotation variables
GLfloat rotation_angle_triangle = 0.0f;
GLfloat rotation_angle_rectangle = 0.0f;

//spher related variables
float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_texture[764];
unsigned short sphere_elements[2280];
unsigned int gNumVertices;
unsigned int gNumElements;

//light values
float LightAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float LightDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightPosition[4] = { 0.0f, 0.0f, 0.0f, 1.0f };			//{ 1.0f, 1.0f, 1.0f, 1.0f };

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 0.0f;							//{128.0f};

struct material_array
{
	float MaterialAmbient[4];
	float MaterialDiffuse[4];
	float MaterialSpecular[4];
	float MaterialShininess;
};

material_array mat_arr[24];

//flags 
BOOL bLight = FALSE;
int iCount = 0;

float lightAngle = 0.0f;

int giWindowWidth = 1366;
int giWindowHeight = 768;

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
		TEXT("Per Fragment Light On Sphere"),
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
/*
		case 'F':
		case 'f':
			fprintf_s(gpFile, "2.1 Going To ToggleFullScreen()\n\n");
			oglToggleFullScreen();
			break;
*/
		case 'L':
		case 'l':
			if (bLight == FALSE)
			{
				bLight = TRUE;
			}
			else
			{
				bLight = FALSE;
			}
			break;

		case 'X':
		case 'x':
			iCount = 1;
			break;

		case 'Y':
		case 'y':
			iCount = 2;
			break;

		case 'Z':
		case 'z':
			iCount = 3;
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
	void oglInitMaterial(void);
	void oglToggleFullScreen(void);

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
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform mat4 u_mvp_matrix;" \
		"uniform int u_LKeyPressed;" \
		"uniform vec4 u_light_position;" \
		"out vec3 t_norm;" \
		"out vec3 light_direction;" \
		"out vec3 viewer_vector;" \
		"void main(void)" \
		"{" \
		"if (u_LKeyPressed == 1)" \
		"{" \
		"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" \
		"mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" \
		"t_norm = normal_matrix * vNormal;" \
		"light_direction = vec3(u_light_position - eye_coordinates);" \
		"viewer_vector = vec3(-eye_coordinates);" \
		"}" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
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
		"in vec3 t_norm;" \
		"in vec3 light_direction;" \
		"in vec3 viewer_vector;" \
		"uniform int u_LKeyPressed;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec4 u_light_position;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float shininess;" \
		"out vec3 phong_ads_light;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"if(u_LKeyPressed == 1)" \
		"{" \
		"vec3 normalised_transformed_normal = normalize(t_norm);" \
		"vec3 normalised_light_direction = normalize(light_direction);" \
		"vec3 normalised_viewer_vector = normalize(viewer_vector);" \
		"vec3 reflection_vector = reflect(-normalised_light_direction, normalised_transformed_normal);" \
		"float tn_dot_LightDirection = max(dot(normalised_light_direction, normalised_transformed_normal), 0.0);" \
		"vec3 ambient = u_La * u_Ka;" \
		"vec3 diffuse = u_Ld * u_Kd * tn_dot_LightDirection;" \
		"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalised_viewer_vector), 0.0), shininess);" \
		"phong_ads_light = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{" \
		"phong_ads_light = vec3(1.0, 1.0, 1.0);" \
		"}" \
		"FragColor = vec4(phong_ads_light, 0.0);" \
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
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_NORMAL, "vNormal");

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
	model_uniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
	view_uniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
	projection_uniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
	LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject, "u_LKeyPressed");
	La_uniform = glGetUniformLocation(gShaderProgramObject, "u_La");
	Ld_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
	Ls_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ls");
	Ka_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	Kd_uniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	Ks_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
	shininess_uniform = glGetUniformLocation(gShaderProgramObject, "shininess");
	lightPosition_uniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");


	//sphere vertices
	getSphereVertexData(sphere_vertices, sphere_normals, sphere_texture, sphere_elements);
	gNumVertices = getNumberOfSphereVertices();
	gNumElements = getNumberOfSphereElements();
	for (int i = 0; i < 1146; i++)
	{
		fprintf_s(gpFile, "sphere_vertices [%d] : %f, ", i, sphere_vertices[i]);
	}
	fprintf_s(gpFile, "gNumVertices : %d\n gNumElements : %d\n", gNumVertices, gNumElements);

	glGenVertexArrays(1, &vao_sphere);
	glBindVertexArray(vao_sphere);
	//position
	glGenBuffers(1, &vbo_sphere_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_vertices), sphere_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//normal
	glGenBuffers(1, &vbo_sphere_normal);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_normals), sphere_normals, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//elements
	glGenBuffers(1, &vbo_sphere_element);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements), sphere_elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//clear the window
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

	//disable culling
	glDisable(GL_CULL_FACE);
	
	//init material
	oglInitMaterial();

	//depth
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	//make orthograhic projection matrix a identity matrix
	perspectiveProjectionMatrix = vmath::mat4::identity();

	//toggle
	oglToggleFullScreen();

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
	void oglDraw24Spheres(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderProgramObject);

	oglDraw24Spheres();

	//unuse program
	glUseProgram(0);
	SwapBuffers(ghdc);
}

void oglUpdate(void)
{
	//code
	lightAngle = lightAngle + 0.005f;
	if (lightAngle >= 360)
	{
		lightAngle = 0.0f;
	}
}

void oglUninitialise(void)
{
	fprintf_s(gpFile, "\nIn OGLUninitialise\n");

	//code

	if (vbo_sphere_element)
	{
		glDeleteBuffers(1, &vbo_sphere_element);
		vbo_sphere_element = 0;
	}
	if (vbo_sphere_normal)
	{
		glDeleteBuffers(1, &vbo_sphere_normal);
		vbo_sphere_normal = 0;
	}
	if (vbo_sphere_position)
	{
		glDeleteBuffers(1, &vbo_sphere_position);
		vbo_sphere_position = 0;
	}
	if (vao_sphere)
	{
		glDeleteVertexArrays(1, &vao_sphere);
		vao_sphere = 0;
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

void oglDraw24Spheres(void)
{
	//variable declaration
	int i = 0;
	float _x = -7.0f;
	float _y = 3.0f;

	//declaration of metrices
	vmath::mat4 modelMatrix;
	vmath::mat4 viewMatrix;
	vmath::mat4 projectionMatrix;
	vmath::mat4 translationMatrix;

	for (i = 0; i < 24; i++)
	{
		glViewport((i % 6) * giWindowWidth / 6, giWindowHeight - (i / 6 + 1) * giWindowHeight / 4, (GLsizei)giWindowWidth / 6, (GLsizei)giWindowHeight / 4);

		perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)(giWindowWidth / 6) / (GLfloat)(giWindowHeight / 4), 0.1f, 100.0f);
		//init above metrices to identity
		modelMatrix = vmath::mat4::identity();
		viewMatrix = vmath::mat4::identity();
		projectionMatrix = vmath::mat4::identity();
		translationMatrix = vmath::mat4::identity();

		//do necessary transformations here
		if (_x <= 5.0f)
		{
			_x = _x + 2.0f;
		}
		if (_x > 5.0f)
		{
			_x = -5.0f;
			_y = _y - 2.0f;
		}
		translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

		//do necessary matrix multiplication
		modelMatrix = modelMatrix * translationMatrix;
		projectionMatrix *= perspectiveProjectionMatrix;

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(model_uniform, 1, GL_FALSE, modelMatrix);
		glUniformMatrix4fv(view_uniform, 1, GL_FALSE, viewMatrix);
		glUniformMatrix4fv(projection_uniform, 1, GL_FALSE, projectionMatrix);

		//if light is enabled
		if (bLight)
		{
			//notify shader that we pressed the "L" key
			glUniform1i(LKeyPressed_Uniform, 1);
			//send light intensityx
			glUniform3fv(La_uniform, 1, LightAmbient);
			glUniform3fv(Ld_uniform, 1, LightDiffuse);
			glUniform3fv(Ls_uniform, 1, LightSpecular);
			//send coeff. of material's reflectivity
			glUniform3fv(Ka_uniform, 1, mat_arr[i].MaterialAmbient);
			glUniform3fv(Kd_uniform, 1, mat_arr[i].MaterialDiffuse);
			glUniform3fv(Ks_uniform, 1, mat_arr[i].MaterialSpecular);
			//shininess
			glUniform1f(shininess_uniform, mat_arr[i].MaterialShininess);
			//send light position
			if (iCount == 1)
			{
				LightPosition[0] = 0.0f;
				LightPosition[1] = cosf(lightAngle) * RADIUS;
				LightPosition[2] = sinf(lightAngle) * RADIUS;
				LightPosition[3] = 1.0f;
			}

			if (iCount == 2)
			{
				LightPosition[0] = cosf(lightAngle) * RADIUS;
				LightPosition[1] = 0.0f;
				LightPosition[2] = sinf(lightAngle) * RADIUS;
				LightPosition[3] = 1.0f;
			}

			if (iCount == 3)
			{
				LightPosition[0] = cosf(lightAngle) * RADIUS;
				LightPosition[1] = sinf(lightAngle) * RADIUS;
				LightPosition[2] = 0.0f;
				LightPosition[3] = 1.0f;
			}

			glUniform4fv(lightPosition_uniform, 1, LightPosition);
		}
		else
		{
			//notify shader that we aren't pressed the "L" key
			glUniform1i(LKeyPressed_Uniform, 0);
		}

		//bind with vao
		glBindVertexArray(vao_sphere);

		//draw scene
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element);
		glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);

		//unbind vao
		glBindVertexArray(0);
	}
}

void oglInitMaterial(void)
{
	//code
	//emrald
	mat_arr[0].MaterialAmbient[0] = 0.0215f;
	mat_arr[0].MaterialAmbient[1] = 0.1745f;
	mat_arr[0].MaterialAmbient[2] = 0.0215f;
	mat_arr[0].MaterialAmbient[3] = 1.0f;
	mat_arr[0].MaterialDiffuse[0] = 0.07568f;
	mat_arr[0].MaterialDiffuse[1] = 0.61424f;
	mat_arr[0].MaterialDiffuse[2] = 0.07568f;
	mat_arr[0].MaterialDiffuse[3] = 1.0f;
	mat_arr[0].MaterialSpecular[0] = 0.633f;
	mat_arr[0].MaterialSpecular[1] = 0.727811f;
	mat_arr[0].MaterialSpecular[2] = 0.633f;
	mat_arr[0].MaterialSpecular[3] = 1.0f;
	mat_arr[0].MaterialShininess = 0.6f * 128.0f;

	//jade
	mat_arr[1].MaterialAmbient[0] = 0.135f;
	mat_arr[1].MaterialAmbient[1] = 0.2225f;
	mat_arr[1].MaterialAmbient[2] = 0.1575f;
	mat_arr[1].MaterialAmbient[3] = 1.0f;
	mat_arr[1].MaterialDiffuse[0] = 0.54f;
	mat_arr[1].MaterialDiffuse[1] = 0.89f;
	mat_arr[1].MaterialDiffuse[2] = 0.63f;
	mat_arr[1].MaterialDiffuse[3] = 1.0f;
	mat_arr[1].MaterialSpecular[0] = 0.316228f;
	mat_arr[1].MaterialSpecular[1] = 0.316228f;
	mat_arr[1].MaterialSpecular[2] = 0.316228f;
	mat_arr[1].MaterialSpecular[3] = 1.0f;
	mat_arr[1].MaterialShininess = 0.1f * 128.0f;

	//obsidian
	mat_arr[2].MaterialAmbient[0] = 0.05375f;
	mat_arr[2].MaterialAmbient[1] = 0.05f;
	mat_arr[2].MaterialAmbient[2] = 0.06625f;
	mat_arr[2].MaterialAmbient[3] = 1.0f;
	mat_arr[2].MaterialDiffuse[0] = 0.18275f;
	mat_arr[2].MaterialDiffuse[1] = 0.17f;
	mat_arr[2].MaterialDiffuse[2] = 0.22525f;
	mat_arr[2].MaterialDiffuse[3] = 1.0f;
	mat_arr[2].MaterialSpecular[0] = 0.332741f;
	mat_arr[2].MaterialSpecular[1] = 0.328634f;
	mat_arr[2].MaterialSpecular[2] = 0.346435f;
	mat_arr[2].MaterialSpecular[3] = 1.0f;
	mat_arr[2].MaterialShininess = 0.3f * 128.0f;

	//pearl
	mat_arr[3].MaterialAmbient[0] = 0.25f;
	mat_arr[3].MaterialAmbient[1] = 0.20725f;
	mat_arr[3].MaterialAmbient[2] = 0.20725f;
	mat_arr[3].MaterialAmbient[3] = 1.0f;
	mat_arr[3].MaterialDiffuse[0] = 1.0f;
	mat_arr[3].MaterialDiffuse[1] = 0.829f;
	mat_arr[3].MaterialDiffuse[2] = 0.829f;
	mat_arr[3].MaterialDiffuse[3] = 1.0f;
	mat_arr[3].MaterialSpecular[0] = 0.296648f;
	mat_arr[3].MaterialSpecular[1] = 0.296648f;
	mat_arr[3].MaterialSpecular[2] = 0.296648f;
	mat_arr[3].MaterialSpecular[3] = 1.0f;
	mat_arr[3].MaterialShininess = 0.088f * 128.0f;

	//ruby
	mat_arr[4].MaterialAmbient[0] = 0.1745f;
	mat_arr[4].MaterialAmbient[1] = 0.01175f;
	mat_arr[4].MaterialAmbient[2] = 0.01175f;
	mat_arr[4].MaterialAmbient[3] = 1.0f;
	mat_arr[4].MaterialDiffuse[0] = 0.61424f;
	mat_arr[4].MaterialDiffuse[1] = 0.04136f;
	mat_arr[4].MaterialDiffuse[2] = 0.04136f;
	mat_arr[4].MaterialDiffuse[3] = 1.0f;
	mat_arr[4].MaterialSpecular[0] = 0.727811f;
	mat_arr[4].MaterialSpecular[1] = 0.626959f;
	mat_arr[4].MaterialSpecular[2] = 0.626959f;
	mat_arr[4].MaterialSpecular[3] = 1.0f;
	mat_arr[4].MaterialShininess = 0.6f * 128.0f;

	//Turquoise
	mat_arr[5].MaterialAmbient[0] = 0.1f;
	mat_arr[5].MaterialAmbient[1] = 0.18725f;
	mat_arr[5].MaterialAmbient[2] = 0.1745f;
	mat_arr[5].MaterialAmbient[3] = 1.0f;
	mat_arr[5].MaterialDiffuse[0] = 0.396f;
	mat_arr[5].MaterialDiffuse[1] = 0.74151f;
	mat_arr[5].MaterialDiffuse[2] = 0.69102f;
	mat_arr[5].MaterialDiffuse[3] = 1.0f;
	mat_arr[5].MaterialSpecular[0] = 0.297254f;
	mat_arr[5].MaterialSpecular[1] = 0.30829f;
	mat_arr[5].MaterialSpecular[2] = 0.306678f;
	mat_arr[5].MaterialSpecular[3] = 1.0f;
	mat_arr[5].MaterialShininess = 0.1f * 128.0f;

	//brass
	mat_arr[6].MaterialAmbient[0] = 0.329412f;
	mat_arr[6].MaterialAmbient[1] = 0.223529f;
	mat_arr[6].MaterialAmbient[2] = 0.027451f;
	mat_arr[6].MaterialAmbient[3] = 1.0f;
	mat_arr[6].MaterialDiffuse[0] = 0.782392f;
	mat_arr[6].MaterialDiffuse[1] = 0.568627f;
	mat_arr[6].MaterialDiffuse[2] = 0.113725f;
	mat_arr[6].MaterialDiffuse[3] = 1.0f;
	mat_arr[6].MaterialSpecular[0] = 0.992157f;
	mat_arr[6].MaterialSpecular[1] = 0.941176f;
	mat_arr[6].MaterialSpecular[2] = 0.807843f;
	mat_arr[6].MaterialSpecular[3] = 1.0f;
	mat_arr[6].MaterialShininess = 0.21794872f * 128.0f;

	//bronze
	mat_arr[7].MaterialAmbient[0] = 0.2125f;
	mat_arr[7].MaterialAmbient[1] = 0.1275f;
	mat_arr[7].MaterialAmbient[2] = 0.054f;
	mat_arr[7].MaterialAmbient[3] = 1.0f;
	mat_arr[7].MaterialDiffuse[0] = 0.714f;
	mat_arr[7].MaterialDiffuse[1] = 0.4284f;
	mat_arr[7].MaterialDiffuse[2] = 0.18144f;
	mat_arr[7].MaterialDiffuse[3] = 1.0f;
	mat_arr[7].MaterialSpecular[0] = 0.393548f;
	mat_arr[7].MaterialSpecular[1] = 0.271906f;
	mat_arr[7].MaterialSpecular[2] = 0.166721f;
	mat_arr[7].MaterialSpecular[3] = 1.0f;
	mat_arr[7].MaterialShininess = 0.2f * 128.0f;

	//chrome
	mat_arr[8].MaterialAmbient[0] = 0.25f;
	mat_arr[8].MaterialAmbient[1] = 0.25f;
	mat_arr[8].MaterialAmbient[2] = 0.25f;
	mat_arr[8].MaterialAmbient[3] = 1.0f;
	mat_arr[8].MaterialDiffuse[0] = 0.4f;
	mat_arr[8].MaterialDiffuse[1] = 0.4f;
	mat_arr[8].MaterialDiffuse[2] = 0.4f;
	mat_arr[8].MaterialDiffuse[3] = 1.0f;
	mat_arr[8].MaterialSpecular[0] = 0.774597f;
	mat_arr[8].MaterialSpecular[1] = 0.774597f;
	mat_arr[8].MaterialSpecular[2] = 0.774597f;
	mat_arr[8].MaterialSpecular[3] = 1.0f;
	mat_arr[8].MaterialShininess = 0.6f * 128.0f;

	//copper
	mat_arr[9].MaterialAmbient[0] = 0.19125f;
	mat_arr[9].MaterialAmbient[1] = 0.0735f;
	mat_arr[9].MaterialAmbient[2] = 0.0225f;
	mat_arr[9].MaterialAmbient[3] = 1.0f;
	mat_arr[9].MaterialDiffuse[0] = 0.7038f;
	mat_arr[9].MaterialDiffuse[1] = 0.27048f;
	mat_arr[9].MaterialDiffuse[2] = 0.0828f;
	mat_arr[9].MaterialDiffuse[3] = 1.0f;
	mat_arr[9].MaterialSpecular[0] = 0.256777f;
	mat_arr[9].MaterialSpecular[1] = 0.137622f;
	mat_arr[9].MaterialSpecular[2] = 0.086014f;
	mat_arr[9].MaterialSpecular[3] = 1.0f;
	mat_arr[9].MaterialShininess = 0.1f * 128.0f;

	//gold
	mat_arr[10].MaterialAmbient[0] = 0.24725f;
	mat_arr[10].MaterialAmbient[1] = 0.1995f;
	mat_arr[10].MaterialAmbient[2] = 0.0745f;
	mat_arr[10].MaterialAmbient[3] = 1.0f;
	mat_arr[10].MaterialDiffuse[0] = 0.75164f;
	mat_arr[10].MaterialDiffuse[1] = 0.60648f;
	mat_arr[10].MaterialDiffuse[2] = 0.22648f;
	mat_arr[10].MaterialDiffuse[3] = 1.0f;
	mat_arr[10].MaterialSpecular[0] = 0.628281f;
	mat_arr[10].MaterialSpecular[1] = 0.555802f;
	mat_arr[10].MaterialSpecular[2] = 0.366065f;
	mat_arr[10].MaterialSpecular[3] = 1.0f;
	mat_arr[10].MaterialShininess = 0.4f * 128.0f;

	//silver
	mat_arr[11].MaterialAmbient[0] = 0.19225f;
	mat_arr[11].MaterialAmbient[1] = 0.19225f;
	mat_arr[11].MaterialAmbient[2] = 0.19225f;
	mat_arr[11].MaterialAmbient[3] = 1.0f;
	mat_arr[11].MaterialDiffuse[0] = 0.50754f;
	mat_arr[11].MaterialDiffuse[1] = 0.50754f;
	mat_arr[11].MaterialDiffuse[2] = 0.50754f;
	mat_arr[11].MaterialDiffuse[3] = 1.0f;
	mat_arr[11].MaterialSpecular[0] = 0.508273f;
	mat_arr[11].MaterialSpecular[1] = 0.508273f;
	mat_arr[11].MaterialSpecular[2] = 0.508273f;
	mat_arr[11].MaterialSpecular[3] = 1.0f;
	mat_arr[11].MaterialShininess = 0.4f * 128.0f;

	//Black Plastic
	mat_arr[12].MaterialAmbient[0] = 0.0f;
	mat_arr[12].MaterialAmbient[1] = 0.0f;
	mat_arr[12].MaterialAmbient[2] = 0.0f;
	mat_arr[12].MaterialAmbient[3] = 1.0f;
	mat_arr[12].MaterialDiffuse[0] = 0.01f;
	mat_arr[12].MaterialDiffuse[1] = 0.01f;
	mat_arr[12].MaterialDiffuse[2] = 0.01f;
	mat_arr[12].MaterialDiffuse[3] = 1.0f;
	mat_arr[12].MaterialSpecular[0] = 0.50f;
	mat_arr[12].MaterialSpecular[1] = 0.50f;
	mat_arr[12].MaterialSpecular[2] = 0.50f;
	mat_arr[12].MaterialSpecular[3] = 1.0f;
	mat_arr[12].MaterialShininess = 0.25f * 128.0f;
	//Cyan Plastic
	mat_arr[13].MaterialAmbient[0] = 0.0f;
	mat_arr[13].MaterialAmbient[1] = 0.1f;
	mat_arr[13].MaterialAmbient[2] = 0.06f;
	mat_arr[13].MaterialAmbient[3] = 1.0f;
	mat_arr[13].MaterialDiffuse[0] = 0.01f;
	mat_arr[13].MaterialDiffuse[1] = 0.50980392f;
	mat_arr[13].MaterialDiffuse[2] = 0.50980392f;
	mat_arr[13].MaterialDiffuse[3] = 1.0f;
	mat_arr[13].MaterialSpecular[0] = 0.50196078f;
	mat_arr[13].MaterialSpecular[1] = 0.50196078f;
	mat_arr[13].MaterialSpecular[2] = 0.50196078f;
	mat_arr[13].MaterialSpecular[3] = 1.0f;
	mat_arr[13].MaterialShininess = 0.25f * 128.0f;
	//Green Plastic
	mat_arr[14].MaterialAmbient[0] = 0.0f;
	mat_arr[14].MaterialAmbient[1] = 0.0f;
	mat_arr[14].MaterialAmbient[2] = 0.0f;
	mat_arr[14].MaterialAmbient[3] = 1.0f;
	mat_arr[14].MaterialDiffuse[0] = 0.1f;
	mat_arr[14].MaterialDiffuse[1] = 0.35f;
	mat_arr[14].MaterialDiffuse[2] = 0.1f;
	mat_arr[14].MaterialDiffuse[3] = 1.0f;
	mat_arr[14].MaterialSpecular[0] = 0.45f;
	mat_arr[14].MaterialSpecular[1] = 0.55f;
	mat_arr[14].MaterialSpecular[2] = 0.45f;
	mat_arr[14].MaterialSpecular[3] = 1.0f;
	mat_arr[14].MaterialShininess = 0.25f * 128.0f;	
	//Red Plastic
	mat_arr[15].MaterialAmbient[0] = 0.0f;
	mat_arr[15].MaterialAmbient[1] = 0.0f;
	mat_arr[15].MaterialAmbient[2] = 0.0f;
	mat_arr[15].MaterialAmbient[3] = 1.0f;
	mat_arr[15].MaterialDiffuse[0] = 0.5f;
	mat_arr[15].MaterialDiffuse[1] = 0.0f;
	mat_arr[15].MaterialDiffuse[2] = 0.0f;
	mat_arr[15].MaterialDiffuse[3] = 1.0f;
	mat_arr[15].MaterialSpecular[0] = 0.7f;
	mat_arr[15].MaterialSpecular[1] = 0.6f;
	mat_arr[15].MaterialSpecular[2] = 0.6f;
	mat_arr[15].MaterialSpecular[3] = 1.0f;
	mat_arr[15].MaterialShininess = 0.25f * 128.0f;
	//White Plastic
	mat_arr[16].MaterialAmbient[0] = 0.0f;
	mat_arr[16].MaterialAmbient[1] = 0.0f;
	mat_arr[16].MaterialAmbient[2] = 0.0f;
	mat_arr[16].MaterialAmbient[3] = 1.0f;
	mat_arr[16].MaterialDiffuse[0] = 0.55f;
	mat_arr[16].MaterialDiffuse[1] = 0.55f;
	mat_arr[16].MaterialDiffuse[2] = 0.55f;
	mat_arr[16].MaterialDiffuse[3] = 1.0f;
	mat_arr[16].MaterialSpecular[0] = 0.70f;
	mat_arr[16].MaterialSpecular[1] = 0.70f;
	mat_arr[16].MaterialSpecular[2] = 0.70f;
	mat_arr[16].MaterialSpecular[3] = 1.0f;
	mat_arr[16].MaterialShininess = 0.25f * 128.0f;
	//yellow Plastic
	mat_arr[17].MaterialAmbient[0] = 0.0f;
	mat_arr[17].MaterialAmbient[1] = 0.0f;
	mat_arr[17].MaterialAmbient[2] = 0.0f;
	mat_arr[17].MaterialAmbient[3] = 1.0f;
	mat_arr[17].MaterialDiffuse[0] = 0.5f;
	mat_arr[17].MaterialDiffuse[1] = 0.5f;
	mat_arr[17].MaterialDiffuse[2] = 0.0f;
	mat_arr[17].MaterialDiffuse[3] = 1.0f;
	mat_arr[17].MaterialSpecular[0] = 0.60f;
	mat_arr[17].MaterialSpecular[1] = 0.60f;
	mat_arr[17].MaterialSpecular[2] = 0.50f;
	mat_arr[17].MaterialSpecular[3] = 1.0f;
	mat_arr[17].MaterialShininess = 0.25f * 128.0f;	

	//Black Rubber
	mat_arr[18].MaterialAmbient[0] = 0.02f;
	mat_arr[18].MaterialAmbient[1] = 0.02f;
	mat_arr[18].MaterialAmbient[2] = 0.02f;
	mat_arr[18].MaterialAmbient[3] = 1.0f;
	mat_arr[18].MaterialDiffuse[0] = 0.01f;
	mat_arr[18].MaterialDiffuse[1] = 0.01f;
	mat_arr[18].MaterialDiffuse[2] = 0.01f;
	mat_arr[18].MaterialDiffuse[3] = 1.0f;
	mat_arr[18].MaterialSpecular[0] = 0.4f;
	mat_arr[18].MaterialSpecular[1] = 0.4f;
	mat_arr[18].MaterialSpecular[2] = 0.4f;
	mat_arr[18].MaterialSpecular[3] = 1.0f;
	mat_arr[18].MaterialShininess = 0.078125f * 128.0f;
	
	//Cyan Rubber
	mat_arr[19].MaterialAmbient[0] = 0.0f;
	mat_arr[19].MaterialAmbient[1] = 0.05f;
	mat_arr[19].MaterialAmbient[2] = 0.05f;
	mat_arr[19].MaterialAmbient[3] = 1.0f;
	mat_arr[19].MaterialDiffuse[0] = 0.4f;
	mat_arr[19].MaterialDiffuse[1] = 0.5f;
	mat_arr[19].MaterialDiffuse[2] = 0.5f;
	mat_arr[19].MaterialDiffuse[3] = 1.0f;
	mat_arr[19].MaterialSpecular[0] = 0.04f;
	mat_arr[19].MaterialSpecular[1] = 0.7f;
	mat_arr[19].MaterialSpecular[2] = 0.7f;
	mat_arr[19].MaterialSpecular[3] = 1.0f;
	mat_arr[19].MaterialShininess = 0.078125f * 128.0f;
	
	//Green Rubber
	mat_arr[20].MaterialAmbient[0] = 0.0f;
	mat_arr[20].MaterialAmbient[1] = 0.05f;
	mat_arr[20].MaterialAmbient[2] = 0.0f;
	mat_arr[20].MaterialAmbient[3] = 1.0f;
	mat_arr[20].MaterialDiffuse[0] = 0.4f;
	mat_arr[20].MaterialDiffuse[1] = 0.5f;
	mat_arr[20].MaterialDiffuse[2] = 0.4f;
	mat_arr[20].MaterialDiffuse[3] = 1.0f;
	mat_arr[20].MaterialSpecular[0] = 0.04f;
	mat_arr[20].MaterialSpecular[1] = 0.7f;
	mat_arr[20].MaterialSpecular[2] = 0.04f;
	mat_arr[20].MaterialSpecular[3] = 1.0f;
	mat_arr[20].MaterialShininess = 0.078125f * 128.0f;
	
	//Red Rubber
	mat_arr[21].MaterialAmbient[0] = 0.05f;
	mat_arr[21].MaterialAmbient[1] = 0.0f;
	mat_arr[21].MaterialAmbient[2] = 0.0f;
	mat_arr[21].MaterialAmbient[3] = 1.0f;
	mat_arr[21].MaterialDiffuse[0] = 0.5f;
	mat_arr[21].MaterialDiffuse[1] = 0.4f;
	mat_arr[21].MaterialDiffuse[2] = 0.4f;
	mat_arr[21].MaterialDiffuse[3] = 1.0f;
	mat_arr[21].MaterialSpecular[0] = 0.7f;
	mat_arr[21].MaterialSpecular[1] = 0.04f;
	mat_arr[21].MaterialSpecular[2] = 0.04f;
	mat_arr[21].MaterialSpecular[3] = 1.0f;
	mat_arr[21].MaterialShininess = 0.078125f * 128.0f;
	
	//White Rubber
	mat_arr[22].MaterialAmbient[0] = 0.05f;
	mat_arr[22].MaterialAmbient[1] = 0.05f;
	mat_arr[22].MaterialAmbient[2] = 0.05f;
	mat_arr[22].MaterialAmbient[3] = 1.0f;
	mat_arr[22].MaterialDiffuse[0] = 0.5f;
	mat_arr[22].MaterialDiffuse[1] = 0.5f;
	mat_arr[22].MaterialDiffuse[2] = 0.5f;
	mat_arr[22].MaterialDiffuse[3] = 1.0f;
	mat_arr[22].MaterialSpecular[0] = 0.7f;
	mat_arr[22].MaterialSpecular[1] = 0.7f;
	mat_arr[22].MaterialSpecular[2] = 0.7f;
	mat_arr[22].MaterialSpecular[3] = 1.0f;
	mat_arr[22].MaterialShininess = 0.078125f * 128.0f;
	
	//Yellow Rubber
	mat_arr[23].MaterialAmbient[0] = 0.05f;
	mat_arr[23].MaterialAmbient[1] = 0.05f;
	mat_arr[23].MaterialAmbient[2] = 0.0f;
	mat_arr[23].MaterialAmbient[3] = 1.0f;
	mat_arr[23].MaterialDiffuse[0] = 0.5f;
	mat_arr[23].MaterialDiffuse[1] = 0.5f;
	mat_arr[23].MaterialDiffuse[2] = 0.4f;
	mat_arr[23].MaterialDiffuse[3] = 1.0f;
	mat_arr[23].MaterialSpecular[0] = 0.7f;
	mat_arr[23].MaterialSpecular[1] = 0.7f;
	mat_arr[23].MaterialSpecular[2] = 0.04f;
	mat_arr[23].MaterialSpecular[3] = 1.0f;
	mat_arr[23].MaterialShininess = 0.078125f * 128.0f;
}
