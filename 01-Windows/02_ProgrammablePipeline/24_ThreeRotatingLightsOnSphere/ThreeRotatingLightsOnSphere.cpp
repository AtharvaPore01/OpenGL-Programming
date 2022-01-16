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
#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define X (GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH) / 2
#define Y (GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT) / 2
#define RADIUS	100.0f

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
GLuint gVertexShaderObject_perVertex;
GLuint gVertexShaderObject_perFragment;
GLuint gFragmentShaderObject_perVertex;
GLuint gFragmentShaderObject_perFragment;
GLuint gShaderProgramObject_perVertex;
GLuint gShaderProgramObject_perFragment;

GLuint vao_sphere;
GLuint vbo_sphere_position;
GLuint vbo_sphere_normal;
GLuint vbo_sphere_element;

struct
{
	GLuint model_uniform;
	GLuint view_uniform;
	GLuint projection_uniform;

	GLuint La_uniform_red;
	GLuint La_uniform_blue;
	GLuint La_uniform_green;
	GLuint Ld_uniform_red;
	GLuint Ld_uniform_green;
	GLuint Ld_uniform_blue;
	GLuint Ls_uniform_red;
	GLuint Ls_uniform_green;
	GLuint Ls_uniform_blue;
	GLuint lightPosition_uniform_red;
	GLuint lightPosition_uniform_green;
	GLuint lightPosition_uniform_blue;

	GLuint Ka_uniform;
	GLuint Kd_uniform;
	GLuint Ks_uniform;
	GLuint shininess_uniform;

	GLuint LKeyPressed_Uniform;
}vertex, fragment;

vmath::mat4 perspectiveProjectionMatrix;

//spher related variables
float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_texture[764];
unsigned short sphere_elements[2280];
unsigned int gNumVertices;
unsigned int gNumElements;

//flags 
BOOL bLight = FALSE;
BOOL bPerVertex = TRUE;
BOOL bPerFragment = FALSE;

//for rotation
float x_red = 0.0f;
float y_red = 0.0f;
float z_red = 0.0f;

float x_green = 0.0f;
float y_green = 0.0f;
float z_green = 0.0f;

float x_blue = 0.0f;
float y_blue = 0.0f;
float z_blue = 0.0f;

//light values
//Red
float LightAmbient_red[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
float LightSpecular_red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
float LightPosition_red[4] = { x_red, y_red, z_red, 1.0f };
//float LightPosition_red[4];
//green
float LightAmbient_green[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
float LightSpecular_green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
float LightPosition_green[4] = { x_green, y_green, z_green, 1.0f };
//float LightPosition_green[4];
//blue
float LightAmbient_blue[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_blue[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
float LightSpecular_blue[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
float LightPosition_blue[4] = { x_blue, y_blue, z_blue, 1.0f };
//float LightPosition_blue[4];

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 128.0f;						

float LightAngle_red = 0.0f;
float LightAngle_green = 0.0f;
float LightAngle_blue = 0.0f;

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
		TEXT("Light on Sphere"),
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
			fprintf_s(gpFile, "2.1 Going To ToggleFullScreen()\n\n");
			oglToggleFullScreen();
			break;
		}
		break;

	case WM_CHAR:
		switch (wParam)
		{
		case 'Q':
		case 'q':
			DestroyWindow(hwnd);
			break;

		case 'F':
		case 'f':
			if (bPerFragment == FALSE)
			{
				//MessageBox(NULL, TEXT("bPerFragment Flag True"), TEXT("bPerFragment Flag Message."), MB_OK);
				bPerFragment = TRUE;
				if (bPerVertex == TRUE)
				{
					//MessageBox(NULL, TEXT("bPerVertex Flag False"), TEXT("bPerFragment Flag Message."), MB_OK);
					bPerVertex = FALSE;
				}
			}
			else
			{
				//MessageBox(NULL, TEXT("bPerFragment Flag False"), TEXT("bPerFragment Flag Message."), MB_OK);
				bPerFragment = FALSE;
			}
			break;

		case 'L':
		case 'l':
			if (bLight == FALSE)
			{
				//MessageBox(NULL, TEXT("Light Flag True"), TEXT("bLight Flag Message."), MB_OK);
				bLight = TRUE;

			}
			else
			{
				//MessageBox(NULL, TEXT("Light Flag False"), TEXT("bLight Flag Message."), MB_OK);
				bLight = FALSE;
			}
			break;

		case 'V':
		case 'v':
			if (bPerVertex == FALSE)
			{
			//	MessageBox(NULL, TEXT("bPerVertex Flag True"), TEXT("bPerVertex Flag Message."), MB_OK);
				bPerVertex = TRUE;
				if (bPerFragment == TRUE)
				{
					//MessageBox(NULL, TEXT("bPerFragment Flag False"), TEXT("bPerVertex Flag Message."), MB_OK);
					bPerFragment = FALSE;
				}
			}
			else
			{
				//MessageBox(NULL, TEXT("bPerVertex Flag False"), TEXT("bPerVertex Flag Message."), MB_OK);
				bPerVertex = FALSE;
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

	/**** Per Vertex ****/

	/* Vertex Shader Code */

	//define vertex shader object
	gVertexShaderObject_perVertex = glCreateShader(GL_VERTEX_SHADER);

	//write vertex shader code
	const GLchar *vertexShaderSourceCode_perVertex =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_LKeyPressed;" \
		
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_La_green;" \
		"uniform vec3 u_La_blue;" \
		
		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ld_green;" \
		"uniform vec3 u_Ld_blue;" \
		
		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_Ls_green;" \
		"uniform vec3 u_Ls_blue;" \

		"uniform vec4 u_light_position_red;" \
		"uniform vec4 u_light_position_green;" \
		"uniform vec4 u_light_position_blue;" \

		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		
		"uniform float shininess;" \
		"out vec3 phong_ads_light;" \
		"void main(void)" \
		"{" \
			"if (u_LKeyPressed == 1)" \
			"{" \
				"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" \
				"mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" \
				"vec3 transformed_normal = normalize(normal_matrix * vNormal);" \

				"vec3 light_direction_red = normalize(vec3(u_light_position_red - eye_coordinates));" \
				"vec3 light_direction_green = normalize(vec3(u_light_position_green - eye_coordinates));" \
				"vec3 light_direction_blue = normalize(vec3(u_light_position_blue - eye_coordinates));" \

				"float tn_dot_LightDirection_red = max(dot(light_direction_red, transformed_normal), 0.0);" \
				"float tn_dot_LightDirection_green = max(dot(light_direction_green, transformed_normal), 0.0);" \
				"float tn_dot_LightDirection_blue = max(dot(light_direction_blue, transformed_normal), 0.0);" \

				"vec3 reflection_vector_red = reflect(-light_direction_red, transformed_normal);" \
				"vec3 reflection_vector_green = reflect(-light_direction_green, transformed_normal);" \
				"vec3 reflection_vector_blue = reflect(-light_direction_blue, transformed_normal);" \

				"vec3 viewer_vector = normalize(vec3(-eye_coordinates.xyz));" \

				"vec3 ambient = (u_La_red * u_Ka) + (u_La_green * u_Ka) + (u_La_blue * u_Ka);" \
				"vec3 diffuse = (u_Ld_red * u_Kd * tn_dot_LightDirection_red) + (u_Ld_green * u_Kd * tn_dot_LightDirection_green) + (u_Ld_blue * u_Kd * tn_dot_LightDirection_blue);" \
				"vec3 specular = (u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red, viewer_vector), 0.0), shininess)) + (u_Ls_green * u_Ks * pow(max(dot(reflection_vector_green, viewer_vector), 0.0), shininess)) + (u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue, viewer_vector), 0.0), shininess));" \

				"phong_ads_light = ambient + diffuse + specular;" \
			"}" \
			"else" \
			"{" \
				"phong_ads_light = vec3(1.0, 1.0, 1.0);" \
			"}" \
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";

	//specify above source code to vertex shader object
	glShaderSource(gVertexShaderObject_perVertex, 1, (const GLchar **)&vertexShaderSourceCode_perVertex, NULL);

	//compile the vertex shader
	glCompileShader(gVertexShaderObject_perVertex);

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
	glGetShaderiv(gVertexShaderObject_perVertex, GL_COMPILE_STATUS, &iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject_perVertex, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetShaderInfoLog(gVertexShaderObject_perVertex,
					iInfoLogLength,
					&Written,
					szInfoLog);

				fprintf_s(gpFile, "Per Vertex : Vertex Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	/* Fragment Shader Code */

	//define fragment shader object
	gFragmentShaderObject_perVertex = glCreateShader(GL_FRAGMENT_SHADER);

	//write shader code
	const GLchar *fragmentShaderSourceCode_perVertex =
		"#version 450 core" \
		"\n" \
		"in vec3 phong_ads_light;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(phong_ads_light, 0.0);" \
		"}";
	//specify above shader code to fragment shader object
	glShaderSource(gFragmentShaderObject_perVertex, 1, (const GLchar **)&fragmentShaderSourceCode_perVertex, NULL);

	//compile the shader
	glCompileShader(gFragmentShaderObject_perVertex);

	//error checking
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(gFragmentShaderObject_perVertex, GL_COMPILE_STATUS, &iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject_perVertex, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{

			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetShaderInfoLog(gFragmentShaderObject_perVertex,
					iInfoLogLength,
					&Written,
					szInfoLog);
				fprintf_s(gpFile, "Per Vertex : Fragment Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	//create shader program object
	gShaderProgramObject_perVertex = glCreateProgram();

	//Attach Vertex Shader
	glAttachShader(gShaderProgramObject_perVertex, gVertexShaderObject_perVertex);

	//Attach Fragment Shader
	glAttachShader(gShaderProgramObject_perVertex, gFragmentShaderObject_perVertex);

	//pre linking bonding to vertex attributes
	glBindAttribLocation(gShaderProgramObject_perVertex, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_perVertex, AMC_ATTRIBUTE_NORMAL, "vNormal");

	//link the shader porgram
	glLinkProgram(gShaderProgramObject_perVertex);

	//error checking

	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject_perVertex, GL_LINK_STATUS, &iProgramLinkStatus);

	if (iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject_perVertex, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetProgramInfoLog(gShaderProgramObject_perVertex, iInfoLogLength, &Written, szInfoLog);
				fprintf_s(gpFile, "Per Vertex : Program Link Error : \n %s\n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	//post linking retriving uniform location
	vertex.LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_LKeyPressed");

	vertex.model_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_model_matrix");
	vertex.view_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_view_matrix");
	vertex.projection_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_projection_matrix");

	/*
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_La_green;" \
		"uniform vec3 u_La_blue;" \
		
		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ld_green;" \
		"uniform vec3 u_Ld_blue;" \
		
		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_Ls_green;" \
		"uniform vec3 u_Ls_blue;" \

		"uniform vec4 u_light_position_red;" \
		"uniform vec4 u_light_position_green;" \
		"uniform vec4 u_light_position_blue;" \
	*/

	vertex.La_uniform_red = glGetUniformLocation(gShaderProgramObject_perVertex, "u_La_red");
	vertex.La_uniform_green = glGetUniformLocation(gShaderProgramObject_perVertex, "u_La_green");
	vertex.La_uniform_blue = glGetUniformLocation(gShaderProgramObject_perVertex, "u_La_blue");

	vertex.Ld_uniform_red = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ld_red");
	vertex.Ld_uniform_green = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ld_green");
	vertex.Ld_uniform_blue = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ld_blue");

	vertex.Ls_uniform_red = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ls_red");
	vertex.Ls_uniform_green = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ls_green");
	vertex.Ls_uniform_blue = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ls_blue");

	vertex.Ka_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ka");
	vertex.Kd_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Kd");
	vertex.Ks_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ks");
	
	vertex.shininess_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "shininess");
	
	vertex.lightPosition_uniform_red = glGetUniformLocation(gShaderProgramObject_perVertex, "u_light_position_red");
	vertex.lightPosition_uniform_green= glGetUniformLocation(gShaderProgramObject_perVertex, "u_light_position_green");
	vertex.lightPosition_uniform_blue = glGetUniformLocation(gShaderProgramObject_perVertex, "u_light_position_blue");


	/**** Per Fragment ****/

	/* Vertex Shader Code */

	//define vertex shader object
	gVertexShaderObject_perFragment = glCreateShader(GL_VERTEX_SHADER);

	//write vertex shader code
	const GLchar *vertexShaderSourceCode_perFragment =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_LKeyPressed;" \

		"uniform vec4 u_light_position_red;" \
		"uniform vec4 u_light_position_green;" \
		"uniform vec4 u_light_position_blue;" \
		
		"out vec3 t_norm;" \
		
		"out vec3 light_direction_red;" \
		"out vec3 light_direction_green;" \
		"out vec3 light_direction_blue;" \

		"out vec3 viewer_vector;" \
		"void main(void)" \
		"{" \
			"if (u_LKeyPressed == 1)" \
			"{" \
				"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" \
				"mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" \
				"t_norm = normal_matrix * vNormal;" \

				"light_direction_red = vec3(u_light_position_red - eye_coordinates);" \
				"light_direction_green = vec3(u_light_position_green - eye_coordinates);" \
				"light_direction_blue = vec3(u_light_position_blue - eye_coordinates);" \

				"viewer_vector = vec3(-eye_coordinates);" \
			"}" \
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";

	//specify above source code to vertex shader object
	glShaderSource(gVertexShaderObject_perFragment, 1, (const GLchar **)&vertexShaderSourceCode_perFragment, NULL);

	//compile the vertex shader
	glCompileShader(gVertexShaderObject_perFragment);

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
	glGetShaderiv(gVertexShaderObject_perFragment, GL_COMPILE_STATUS, &iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject_perFragment, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetShaderInfoLog(gVertexShaderObject_perFragment,
					iInfoLogLength,
					&Written,
					szInfoLog);

				fprintf_s(gpFile, "Per Fragment : Vertex Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	/* Fragment Shader Code */

	//define fragment shader object
	gFragmentShaderObject_perFragment = glCreateShader(GL_FRAGMENT_SHADER);

	//write shader code
	const GLchar *fragmentShaderSourceCode_perFragment =
		"#version 450 core" \
		"\n" \
		"in vec3 t_norm;" \

		"in vec3 light_direction_red;" \
		"in vec3 light_direction_green;" \
		"in vec3 light_direction_blue;" \

		"in vec3 viewer_vector;" \

		"uniform int u_LKeyPressed;" \

		"uniform vec3 u_La_red;" \
		"uniform vec3 u_La_green;" \
		"uniform vec3 u_La_blue;" \

		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ld_green;" \
		"uniform vec3 u_Ld_blue;" \

		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_Ls_green;" \
		"uniform vec3 u_Ls_blue;" \

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

				"vec3 normalised_light_direction_red = normalize(light_direction_red);" \
				"vec3 normalised_light_direction_green = normalize(light_direction_green);" \
				"vec3 normalised_light_direction_blue = normalize(light_direction_blue);" \

				"vec3 normalised_viewer_vector = normalize(viewer_vector);" \

				"vec3 reflection_vector_red = reflect(-normalised_light_direction_red, normalised_transformed_normal);" \
				"vec3 reflection_vector_green = reflect(-normalised_light_direction_green, normalised_transformed_normal);" \
				"vec3 reflection_vector_blue = reflect(-normalised_light_direction_blue, normalised_transformed_normal);" \

				"float tn_dot_LightDirection_red = max(dot(normalised_light_direction_red, normalised_transformed_normal), 0.0);" \
				"float tn_dot_LightDirection_green = max(dot(normalised_light_direction_green, normalised_transformed_normal), 0.0);" \
				"float tn_dot_LightDirection_blue = max(dot(normalised_light_direction_blue, normalised_transformed_normal), 0.0);" \

				"vec3 ambient = (u_La_red * u_Ka) + (u_La_green * u_Ka) + (u_La_blue * u_Ka);" \
				"vec3 diffuse = (u_Ld_red * u_Kd * tn_dot_LightDirection_red) + (u_Ld_green * u_Kd * tn_dot_LightDirection_green) + (u_Ld_blue * u_Kd * tn_dot_LightDirection_blue);" \
				"vec3 specular = (u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red, normalised_viewer_vector), 0.0), shininess)) + (u_Ls_green * u_Ks * pow(max(dot(reflection_vector_green, normalised_viewer_vector), 0.0), shininess)) + (u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue, normalised_viewer_vector), 0.0), shininess));" \
				
				"phong_ads_light = ambient + diffuse + specular;" \
			"}" \
			"else" \
			"{" \
				"phong_ads_light = vec3(1.0, 1.0, 1.0);" \
			"}" \
			"FragColor = vec4(phong_ads_light, 0.0);" \
		"}";
	//specify above shader code to fragment shader object
	glShaderSource(gFragmentShaderObject_perFragment, 1, (const GLchar **)&fragmentShaderSourceCode_perFragment, NULL);

	//compile the shader
	glCompileShader(gFragmentShaderObject_perFragment);

	//error checking
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(gFragmentShaderObject_perFragment, GL_COMPILE_STATUS, &iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject_perFragment, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{

			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetShaderInfoLog(gFragmentShaderObject_perFragment,
					iInfoLogLength,
					&Written,
					szInfoLog);
				fprintf_s(gpFile, "Per Fragment : Fragment Shader Error  : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	//create shader program object
	gShaderProgramObject_perFragment = glCreateProgram();

	//Attach Vertex Shader
	glAttachShader(gShaderProgramObject_perFragment, gVertexShaderObject_perFragment);

	//Attach Fragment Shader
	glAttachShader(gShaderProgramObject_perFragment, gFragmentShaderObject_perFragment);

	//pre linking bonding to vertex attributes
	glBindAttribLocation(gShaderProgramObject_perFragment, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_perFragment, AMC_ATTRIBUTE_NORMAL, "vNormal");

	//link the shader porgram
	glLinkProgram(gShaderProgramObject_perFragment);

	//error checking

	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject_perFragment, GL_LINK_STATUS, &iProgramLinkStatus);

	if (iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject_perFragment, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei Written;
				glGetProgramInfoLog(gShaderProgramObject_perFragment, iInfoLogLength, &Written, szInfoLog);
				fprintf_s(gpFile, "Per Fragment : Program Link Error : \n %s\n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	//post linking retriving uniform location
	fragment.LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_LKeyPressed");

	fragment.model_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_model_matrix");
	fragment.view_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_view_matrix");
	fragment.projection_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_projection_matrix");

	/*
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_La_green;" \
		"uniform vec3 u_La_blue;" \

		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ld_green;" \
		"uniform vec3 u_Ld_blue;" \

		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_Ls_green;" \
		"uniform vec3 u_Ls_blue;" \

		"uniform vec4 u_light_position_red;" \
		"uniform vec4 u_light_position_green;" \
		"uniform vec4 u_light_position_blue;" \
	*/

	fragment.La_uniform_red = glGetUniformLocation(gShaderProgramObject_perFragment, "u_La_red");
	fragment.La_uniform_green = glGetUniformLocation(gShaderProgramObject_perFragment, "u_La_green");
	fragment.La_uniform_blue = glGetUniformLocation(gShaderProgramObject_perFragment, "u_La_blue");

	fragment.Ld_uniform_red = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ld_red");
	fragment.Ld_uniform_green = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ld_green");
	fragment.Ld_uniform_blue = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ld_blue");

	fragment.Ls_uniform_red = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ls_red");
	fragment.Ls_uniform_green = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ls_green");
	fragment.Ls_uniform_blue = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ls_blue");

	fragment.Ka_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ka");
	fragment.Kd_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Kd");
	fragment.Ks_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ks");

	fragment.shininess_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "shininess");

	fragment.lightPosition_uniform_red = glGetUniformLocation(gShaderProgramObject_perFragment, "u_light_position_red");
	fragment.lightPosition_uniform_green = glGetUniformLocation(gShaderProgramObject_perFragment, "u_light_position_green");
	fragment.lightPosition_uniform_blue = glGetUniformLocation(gShaderProgramObject_perFragment, "u_light_position_blue");

	//sphere vertices
	getSphereVertexData(sphere_vertices, sphere_normals, sphere_texture, sphere_elements);
	gNumVertices = getNumberOfSphereVertices();
	gNumElements = getNumberOfSphereElements();

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
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//disable culling
	glDisable(GL_CULL_FACE);

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
	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//declaration of metrices
	vmath::mat4 modelMatrix_perVertex;
	vmath::mat4 modelMatrix_perFragment;
	vmath::mat4 viewMatrix_perVertex;
	vmath::mat4 viewMatrix_perFragment;
	vmath::mat4 projectionMatrix_perVertex;
	vmath::mat4 projectionMatrix_perFragment;
	vmath::mat4 translationMatrix_perVertex;

	vmath::mat4 rotationMatrixRed_perVertex;
	vmath::mat4 rotationMatrixGreen_perVertex;
	vmath::mat4 rotationMatrixBlue_perVertex;

	vmath::mat4 translationMatrix_perFragment;

	vmath::mat4 rotationMatrixRed_perFragment;
	vmath::mat4 rotationMatrixGreen_perFragment;
	vmath::mat4 rotationMatrixBlue_perFragment;

	//init above metrices to identity
	modelMatrix_perVertex = vmath::mat4::identity();
	viewMatrix_perVertex = vmath::mat4::identity();
	projectionMatrix_perVertex = vmath::mat4::identity();
	translationMatrix_perVertex = vmath::mat4::identity();
	
	//do necessary transformations here
	translationMatrix_perVertex = vmath::translate(0.0f, 0.0f, -3.0f);

	//do necessary matrix multiplication
	modelMatrix_perVertex = modelMatrix_perVertex * translationMatrix_perVertex;
	projectionMatrix_perVertex *= perspectiveProjectionMatrix;

	//init above metrices to identity
	modelMatrix_perFragment = vmath::mat4::identity();
	viewMatrix_perFragment = vmath::mat4::identity();
	projectionMatrix_perFragment = vmath::mat4::identity();
	translationMatrix_perFragment = vmath::mat4::identity();
	
	//do necessary transformations here
	translationMatrix_perFragment = vmath::translate(0.0f, 0.0f, -3.0f);
	
	//do necessary matrix multiplication
	modelMatrix_perFragment = modelMatrix_perFragment * translationMatrix_perFragment;
	projectionMatrix_perFragment *= perspectiveProjectionMatrix;

	if (bPerVertex)
	{
		glUseProgram(gShaderProgramObject_perVertex);

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(vertex.model_uniform, 1, GL_FALSE, modelMatrix_perVertex);
		glUniformMatrix4fv(vertex.view_uniform, 1, GL_FALSE, viewMatrix_perVertex);
		glUniformMatrix4fv(vertex.projection_uniform, 1, GL_FALSE, projectionMatrix_perVertex);

		//if light is enabled
		if (bLight)
		{
			//notify shader that we pressed the "L" key
			glUniform1i(vertex.LKeyPressed_Uniform, 1);
			//send light intensity
			glUniform3fv(vertex.La_uniform_red, 1, LightAmbient_red);
			glUniform3fv(vertex.La_uniform_green, 1, LightAmbient_green);
			glUniform3fv(vertex.La_uniform_blue, 1, LightAmbient_blue);

			glUniform3fv(vertex.Ld_uniform_red, 1, LightDiffuse_red);
			glUniform3fv(vertex.Ld_uniform_green, 1, LightDiffuse_green);
			glUniform3fv(vertex.Ld_uniform_blue, 1, LightDiffuse_blue);

			glUniform3fv(vertex.Ls_uniform_red, 1, LightSpecular_red);
			glUniform3fv(vertex.Ls_uniform_green, 1, LightSpecular_green);
			glUniform3fv(vertex.Ls_uniform_blue, 1, LightSpecular_blue);
			
			//send coeff. of material's reflectivity
			glUniform3fv(vertex.Ka_uniform, 1, MaterialAmbient);
			glUniform3fv(vertex.Kd_uniform, 1, MaterialDiffuse);
			glUniform3fv(vertex.Ks_uniform, 1, MaterialSpecular);
			//shininess
			glUniform1f(vertex.shininess_uniform, MaterialShininess);
			
			//send light position
			LightPosition_red[0] = 0.0f;
			LightPosition_red[1] = RADIUS * cosf(LightAngle_red);
			LightPosition_red[2] = RADIUS * sinf(LightAngle_red);
			LightPosition_red[3] = 1.0f;
			glUniform4fv(vertex.lightPosition_uniform_red, 1, LightPosition_red);

			LightPosition_green[0] = RADIUS * cosf(LightAngle_green);
			LightPosition_green[1] = 0.0f;
			LightPosition_green[2] = RADIUS * sinf(LightAngle_green);
			LightPosition_green[3] = 1.0f;
			glUniform4fv(vertex.lightPosition_uniform_green, 1, LightPosition_green);

			LightPosition_blue[0] = RADIUS * cosf(LightAngle_blue);
			LightPosition_blue[1] = RADIUS * sinf(LightAngle_blue);
			LightPosition_blue[2] = 0.0f;
			LightPosition_blue[3] = 1.0f;
			glUniform4fv(vertex.lightPosition_uniform_blue, 1, LightPosition_blue);
		}
		else
		{
			//notify shader that we aren't pressed the "L" key
			glUniform1i(vertex.LKeyPressed_Uniform, 0);
		}
	}

	if (bPerFragment)
	{
		glUseProgram(gShaderProgramObject_perFragment);

		//send necessary matrics to shaders in respective uniforms
		glUniformMatrix4fv(fragment.model_uniform, 1, GL_FALSE, modelMatrix_perFragment);
		glUniformMatrix4fv(fragment.view_uniform, 1, GL_FALSE, viewMatrix_perFragment);
		glUniformMatrix4fv(fragment.projection_uniform, 1, GL_FALSE, projectionMatrix_perFragment);

		//if light is enabled
		if (bLight)
		{
			//notify shader that we pressed the "L" key
			glUniform1i(fragment.LKeyPressed_Uniform, 1);
			//send light intensityx
			glUniform3fv(fragment.La_uniform_red, 1, LightAmbient_red);
			glUniform3fv(fragment.La_uniform_green, 1, LightAmbient_green);
			glUniform3fv(fragment.La_uniform_blue, 1, LightAmbient_blue);

			glUniform3fv(fragment.Ld_uniform_red, 1, LightDiffuse_red);
			glUniform3fv(fragment.Ld_uniform_green, 1, LightDiffuse_green);
			glUniform3fv(fragment.Ld_uniform_blue, 1, LightDiffuse_blue);

			glUniform3fv(fragment.Ls_uniform_red, 1, LightSpecular_red);
			glUniform3fv(fragment.Ls_uniform_green, 1, LightSpecular_green);
			glUniform3fv(fragment.Ls_uniform_blue, 1, LightSpecular_blue);

			//send coeff. of material's reflectivity
			glUniform3fv(fragment.Ka_uniform, 1, MaterialAmbient);
			glUniform3fv(fragment.Kd_uniform, 1, MaterialDiffuse);
			glUniform3fv(fragment.Ks_uniform, 1, MaterialSpecular);
			//shininess
			glUniform1f(fragment.shininess_uniform, MaterialShininess);
			//send light position

			LightPosition_red[0] = 0.0f;
			LightPosition_red[1] = RADIUS * cosf(LightAngle_red);
			LightPosition_red[2] = RADIUS * sinf(LightAngle_red);
			LightPosition_red[3] = 1.0f;
			glUniform4fv(fragment.lightPosition_uniform_red, 1, LightPosition_red);

			LightPosition_green[0] = RADIUS * cosf(LightAngle_green);
			LightPosition_green[1] = 0.0f;
			LightPosition_green[2] = RADIUS * sinf(LightAngle_green);
			LightPosition_green[3] = 1.0f;
			glUniform4fv(fragment.lightPosition_uniform_green, 1, LightPosition_green);

			LightPosition_blue[0] = RADIUS * cosf(LightAngle_blue);
			LightPosition_blue[1] = RADIUS * sinf(LightAngle_blue);
			LightPosition_blue[2] = 0.0f;
			LightPosition_blue[3] = 1.0f;
			glUniform4fv(fragment.lightPosition_uniform_blue, 1, LightPosition_blue);
		}

		else
		{
			//notify shader that we aren't pressed the "L" key
			glUniform1i(fragment.LKeyPressed_Uniform, 0);
		}
	}

	if (bPerVertex == TRUE || bPerFragment == TRUE)
	{
		//bind with vao
		glBindVertexArray(vao_sphere);

		//draw scene
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element);
		glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);

		//unbind vao
		glBindVertexArray(0);
	}

	//unuse program
	glUseProgram(0);

	SwapBuffers(ghdc);
}

void oglUpdate(void)
{
	//code
	LightAngle_red = LightAngle_red + 0.02f;
	if (LightAngle_red >= 360)
	{
		LightAngle_red = 0.0f;
	}

	LightAngle_green = LightAngle_green + 0.02f;
	if (LightAngle_green >= 360)
	{
		LightAngle_green = 0.0f;
	}

	LightAngle_blue = LightAngle_blue + 0.02f;
	if (LightAngle_blue >= 360)
	{
		LightAngle_blue = 0.0f;
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

	if (gShaderProgramObject_perVertex)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;

		glUseProgram(gShaderProgramObject_perVertex);

		//ask program how many shaders are attached
		glGetProgramiv(gShaderProgramObject_perVertex, GL_ATTACHED_SHADERS, &shaderCount);

		GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

		if (pShaders)
		{
			glGetAttachedShaders(gShaderProgramObject_perVertex, shaderCount, &shaderCount, pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				//detach shader
				glDetachShader(gShaderProgramObject_perVertex, pShaders[shaderNumber]);
				//delete shader
				glDeleteShader(pShaders[shaderNumber]);
				pShaders[shaderNumber] = 0;
			}
			free(pShaders);
		}
		glDeleteProgram(gShaderProgramObject_perVertex);
		gShaderProgramObject_perVertex = 0;
		glUseProgram(0);
	}

	if (gShaderProgramObject_perFragment)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;

		glUseProgram(gShaderProgramObject_perFragment);

		//ask program how many shaders are attached
		glGetProgramiv(gShaderProgramObject_perFragment, GL_ATTACHED_SHADERS, &shaderCount);

		GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

		if (pShaders)
		{
			glGetAttachedShaders(gShaderProgramObject_perFragment, shaderCount, &shaderCount, pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				//detach shader
				glDetachShader(gShaderProgramObject_perFragment, pShaders[shaderNumber]);
				//delete shader
				glDeleteShader(pShaders[shaderNumber]);
				pShaders[shaderNumber] = 0;
			}
			free(pShaders);
		}
		glDeleteProgram(gShaderProgramObject_perFragment);
		gShaderProgramObject_perFragment = 0;
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
