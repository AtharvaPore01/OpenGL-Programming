//Standard Header Files
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

//X11 Related Header Files
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>

//opengl Related Header files
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>

//mathematics related header files
#include "vmath.h"

// our header file for sphere
#include "Sphere.h"

//namespaces
using namespace std;

//macros
#define RADIUS	100.0f

//enum
enum
{
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOODR_0
};

//global variable declaration
bool			bFullScreen 	=	false;
Display			*gpDisplay		= 	NULL;
XVisualInfo		*gpXVisualInfo	=	NULL;
Colormap		gColormap;
Window			gWindow;
int 			giWindowWidth	=	800;
int 			giWindowHeight	=	600;

//opengl Related global variable
static GLXContext gGLXContext;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
GLXFBConfig gGLXFBConfig;

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

//sphere related variables
float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_texture[764];
unsigned short sphere_elements[2280];
unsigned int gNumVertices;
unsigned int gNumElements;

//flags 
bool bLight = false;
bool bPerVertex = true;
bool bPerFragment = false;

//light values
//Red
float LightAmbient_red[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
float LightSpecular_red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
float LightPosition_red[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

//green
float LightAmbient_green[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
float LightSpecular_green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
float LightPosition_green[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

//blue
float LightAmbient_blue[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_blue[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
float LightSpecular_blue[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
float LightPosition_blue[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 128.0f;						

float LightAngle_red = 0.0f;
float LightAngle_green = 0.0f;
float LightAngle_blue = 0.0f;

//entry-point function
int main(void)
{
	//function prototype
	void oglCreateWindow(void);
	void oglToggleFullscreen(void);
	void oglInitialise(void);
	void oglResize(int, int);
	void oglDisplay(void);
	void oglUpdate(void);
	void oglUninitialise(void);

	//variable declaration
	bool	bDone 		=	false;
	int 	winWidth 	=	giWindowWidth;
	int 	winHeight 	=	giWindowHeight;
	char 	keys[26];

	//code
	oglCreateWindow();

	//initialise
	oglInitialise();

	//Messageloop
	XEvent event;
	KeySym keysym;

	while(bDone == false)
	{
		while(XPending(gpDisplay))
		{
			XNextEvent(gpDisplay, &event);
			switch(event.type)
			{
				case MapNotify:
				break;

				case KeyPress:
					keysym = XkbKeycodeToKeysym(gpDisplay, event.xkey.keycode, 0, 0);
					switch(keysym)
					{
						case XK_Escape:
							if(bFullScreen == false)
							{
								oglToggleFullscreen();
								bFullScreen = true;
							}
							else
							{
								oglToggleFullscreen();
								bFullScreen = false;
							}
							
						break;

						default:
						break;
					}
					XLookupString(&event.xkey, keys, sizeof(keys), NULL, NULL);
					switch(keys[0])
					{
						
						case'F':
						case'f':
							if (bPerFragment == false)
							{
								bPerFragment = true;
								if (bPerVertex == true)
								{
									bPerVertex = false;
								}
							}
							else
							{
								bPerFragment = false;
							}
						break;

						case'V':
						case'v':
							if (bPerVertex == false)
							{
								//MessageBox(NULL, TEXT("bPerVertex Flag True"), TEXT("bPerVertex Flag Message."), MB_OK);
								bPerVertex = true;
								if (bPerFragment == true)
								{
									//MessageBox(NULL, TEXT("bPerFragment Flag False"), TEXT("bPerVertex Flag Message."), MB_OK);
									bPerFragment = false;
								}
							}
							else
							{
								//MessageBox(NULL, TEXT("bPerVertex Flag False"), TEXT("bPerVertex Flag Message."), MB_OK);
								bPerVertex = false;
							}
						break;

						case 'L':
						case 'l':

							if (bLight == false)
							{
								bLight = true;
							}
							else
							{
								bLight = false;
							}

						break;

						case 'Q':
						case 'q':
							bDone = true;
						break;

						default:
						break;

					}
				break;

				case ButtonPress:
					switch(event.xbutton.button)
					{
						case 1:
						break;

						case 2:
						break;

						case 3:
						break;

						default:
						break;

					}
				break;

				case MotionNotify:
				break;

				case ConfigureNotify:
					winWidth 	=	event.xconfigure.width;
					winHeight 	=	event.xconfigure.height;

					oglResize(winWidth, winHeight);
				break;

				case Expose:
				break;

				case DestroyNotify:
				break;

				case 33:
					bDone = true;
				break;

				default:
				break;
			}
		}
		//Here Call Update And Display
		oglUpdate();	
		oglDisplay();
	}

	oglUninitialise();
	return(0);
}

void oglCreateWindow(void)
{
	//function prototype
	void oglUninitialise(void);

	//variable declaration
	XSetWindowAttributes	winAttribs;
	int 					defaultScreen;
	int 					defaultDepth;
	int 					styleMask;

	//window centering related variables
	Screen 					*screen;
	int 					screen_width;
	int 					screen_height;
	int 					screen_count;
	int 					x, y;

	//fbconfig related variables
	GLXFBConfig *pGLXFBConfig = NULL;
	GLXFBConfig bestGLXFBConfig;
	XVisualInfo *pTempXVisualInfo = NULL;
	int iNumberOfFBConfig = 0; 

	int bestFrameBufferConfig = -1;
	int bestNumberOfSamples = -1;
	int worstFrameBufferConfig = -1;
	int worstNumberOfSamples = 999;

	//Code
	//	1.	Initialise the frame buffer attributes
	static int frameBufferAttributes[]	=	
	{	
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		GLX_STENCIL_SIZE, 8,
		GLX_DOUBLEBUFFER, True,
		None
	};

	gpDisplay	=	XOpenDisplay(NULL);
	if(gpDisplay == NULL)
	{
		printf("ERROR : Unable To Open X Display.\n Exitting Now....\n");
		oglUninitialise();
		exit(1);
	}

	defaultScreen	=	XDefaultScreen(gpDisplay);
	
	//	2.	retrive the all fbconfig drivers
	pGLXFBConfig = glXChooseFBConfig(	gpDisplay,
										defaultScreen,
										frameBufferAttributes,
										&iNumberOfFBConfig);

	for(int i = 0; i < iNumberOfFBConfig; i++)
	{
		//	3.	For Each Obtained fbconfig get pTempXVisualInfo
		//		used to check the capability of following 2 calls.

		pTempXVisualInfo = glXGetVisualFromFBConfig(	gpDisplay,
														pGLXFBConfig[i]);

		if(pTempXVisualInfo)
		{
			//	4.	get number of sample buffers from respective fbconfig
			int sampleBuffers, samples;
			glXGetFBConfigAttrib(	gpDisplay,
									pGLXFBConfig[i],
									GLX_SAMPLE_BUFFERS,
									&sampleBuffers);
			//	5.	get number of samples from respective fbconfig
			glXGetFBConfigAttrib(	gpDisplay,
									pGLXFBConfig[i],
									GLX_SAMPLES,
									&samples);

			//	6.	more the number of samples and sampleBuffers more the eligible fbconfig is
			//		so compare.
			

			if(bestFrameBufferConfig < 0 || sampleBuffers && samples > bestNumberOfSamples)
			{
				bestFrameBufferConfig = i;
				bestNumberOfSamples = samples;
			}

			if(worstFrameBufferConfig < 0 || !sampleBuffers || samples < worstNumberOfSamples)
			{
				worstFrameBufferConfig = i;
				worstNumberOfSamples = samples;
			}
		}

		XFree(pTempXVisualInfo);
	}

	//	7.	assign the found best one
	bestGLXFBConfig = pGLXFBConfig[bestFrameBufferConfig];

	//	8.	assign the same best to global one
	gGLXFBConfig = bestGLXFBConfig;

	//	9.	Free the obtained GLXFBConfig array
	XFree(pGLXFBConfig);

	//	10.	Accordingly get best visual
	gpXVisualInfo = glXGetVisualFromFBConfig(	gpDisplay,
												bestGLXFBConfig);

	if(gpXVisualInfo == NULL)
	{
		printf("glXChooseVisual Failed.\nExitting Now...\n");
		oglUninitialise();
		exit(1);
	}

	winAttribs.border_pixel			=	0;
	winAttribs.border_pixmap		=	0;
	winAttribs.background_pixmap	=	0;
	winAttribs.background_pixel		=	BlackPixel(gpDisplay, defaultScreen);
	winAttribs.colormap 			=	XCreateColormap(	gpDisplay,
															RootWindow(gpDisplay, gpXVisualInfo->screen),
															gpXVisualInfo->visual,
															AllocNone);
	gColormap 						=	winAttribs.colormap;

	winAttribs.event_mask 			= 	ExposureMask | VisibilityChangeMask | ButtonPressMask | KeyPressMask | PointerMotionMask | StructureNotifyMask;

	styleMask						=	CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;

	gWindow 						=	XCreateWindow(	gpDisplay,
														RootWindow(gpDisplay, gpXVisualInfo->screen),
														0,
														0,
														giWindowWidth,
														giWindowHeight,
														0,
														gpXVisualInfo->depth,
														InputOutput,
														gpXVisualInfo->visual,
														styleMask,
														&winAttribs);
	if(!gWindow)
	{
		printf("ERROR : Failed To Create Main Window.\n Exitting Now...\n");
		oglUninitialise();
		exit(1);
	}

	XStoreName(gpDisplay, gWindow, "Programmable Pipeline : White Sphere Template ");
	Atom windowManagerDelete 	=	XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay, gWindow, &windowManagerDelete, 1);

	screen_count = ScreenCount(gpDisplay);

	for(int i = 0; i < screen_count; i++)
	{
		screen = ScreenOfDisplay(gpDisplay, i);
	}

	screen_width 	=	screen->width;
	screen_height	=	screen->height;

	x = ((screen_width / 2) - (giWindowWidth / 2));
	y = ((screen_height / 2) - (giWindowHeight / 2));	

	XMapWindow(gpDisplay, gWindow);	
	//This Function will be used to move the window
	XMoveWindow(	gpDisplay,
					gWindow,
					x,
					y);			
}

void oglToggleFullscreen(void)
{
	
	//variable declaration
	Atom wm_state;
	Atom fullscreen;
	XEvent xev = {0};

	//code
	wm_state = XInternAtom(gpDisplay, "_NET_WM_STATE", False);
	memset(&xev, 0, sizeof(xev));

	xev.type 					=	ClientMessage;
	xev.xclient.window 			=	gWindow;
	xev.xclient.message_type	=	wm_state;
	xev.xclient.format			=	32;
	xev.xclient.data.l[0]		=	bFullScreen ? 0 : 1;

	fullscreen 					=	XInternAtom(gpDisplay, "_NET_WM_STATE_FULLSCREEN", False);
	xev.xclient.data.l[1]		=	fullscreen;

	XSendEvent(	gpDisplay,
				RootWindow(gpDisplay, gpXVisualInfo->screen),
				False,
				StructureNotifyMask,
				&xev);
}

void oglInitialise(void)
{
	//variable declaration
	GLenum result;
	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLint iProgramLinkStatus = 0;
	GLchar *szInfoLog = NULL;

	//function declaration
	void oglUninitialise(void);
	void oglResize(int, int);

	//code
	glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte *)"glXCreateContextAttribsARB");

	if(glXCreateContextAttribsARB == NULL)
	{
		printf("ERROR : glXCreateContextAttribsARB Not Found\n");
		oglUninitialise();
	}

	//context attrib array
	const int Attribs[] = 
	{
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};

	//get the context
	gGLXContext = glXCreateContextAttribsARB(	gpDisplay,
												gGLXFBConfig,
												0,
												True,
												Attribs);

	if(!gGLXContext)
	{
			//context attrib array
		const int Attribs[] = 
		{
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		//get the context
		gGLXContext = glXCreateContextAttribsARB(	gpDisplay,
													gGLXFBConfig,
													0,
													True,
													Attribs);
	}

	if(!glXIsDirect(gpDisplay, gGLXContext))
	{
		printf("The Obtained Context Is Not Hardware rendering Context\n\n");
	}
	
	else
	{
		printf("The Obtained Context Is 'Hardware rendering' Context\n\n");
	}

	glXMakeCurrent(	gpDisplay, gWindow, gGLXContext);

	result = glewInit();
	if(result != GLEW_OK)
	{
		oglUninitialise();
		exit(1);
	}

	/** Per Vertex Code **/

	/* Vertex Shader code */

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


	//specify above source code to vertex shader
	glShaderSource(	gVertexShaderObject_perVertex, 
					1,
					(const GLchar **)&vertexShaderSourceCode_perVertex,
					NULL);

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

	glGetShaderiv(	gVertexShaderObject_perVertex,
					GL_COMPILE_STATUS,
					&iShaderCompileStatus);
	if(iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(	gVertexShaderObject_perVertex,
						GL_INFO_LOG_LENGTH,
						&iInfoLogLength);
		if(iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(	gVertexShaderObject_perVertex,
									iInfoLogLength,
									&written,
									szInfoLog);
				printf("Per Vertex Error :\nVertex Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
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

	//Specify The code to the fragment shader object
	glShaderSource(	gFragmentShaderObject_perVertex,
					1,
					(const GLchar **)&fragmentShaderSourceCode_perVertex,
					NULL);

	//compile the shader
	glCompileShader(gFragmentShaderObject_perVertex);

	//error checking
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(	gFragmentShaderObject_perVertex,
					GL_COMPILE_STATUS,
					&iShaderCompileStatus);

	if(iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(	gFragmentShaderObject_perVertex,
						GL_INFO_LOG_LENGTH,
						&iInfoLogLength);

		if(iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(	gFragmentShaderObject_perVertex,
									iInfoLogLength,
									&written,
									szInfoLog);
				printf("Per Vertex Error :\nFragment Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				exit(0);
			}
		}
	}

	//create shader program object
	gShaderProgramObject_perVertex = glCreateProgram();

	//attach vertex shader
	glAttachShader(	gShaderProgramObject_perVertex, gVertexShaderObject_perVertex);

	//attach fragment shader
	glAttachShader(	gShaderProgramObject_perVertex, gFragmentShaderObject_perVertex );

	//pre linking bonding to vertex attributes
	glBindAttribLocation(	gShaderProgramObject_perVertex,
							AMC_ATTRIBUTE_POSITION,
							"vPosition");
	glBindAttribLocation(	gShaderProgramObject_perVertex,
							AMC_ATTRIBUTE_NORMAL,
							"vNormal");
	
	//link shader program
	glLinkProgram(gShaderProgramObject_perVertex);

	//error checking
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(	gShaderProgramObject_perVertex,
					GL_LINK_STATUS,
					&iProgramLinkStatus);

	if(iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(	gShaderProgramObject_perVertex,
						GL_INFO_LOG_LENGTH,
						&iInfoLogLength);

		if(iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(	gShaderProgramObject_perVertex,
										iInfoLogLength,
										&written,
										szInfoLog);

				printf("Per Vertex Error:\nprogram Link Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				exit(0);
			}
		}
	}

	//post linking retriving uniform location
	vertex.LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_LKeyPressed");

	vertex.model_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_model_matrix");
	vertex.view_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_view_matrix");
	vertex.projection_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_projection_matrix");

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

				printf("Per Fragment Error :\nVertex Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
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
				printf("Per Fragment Error :\nFragment Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
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
	glBindAttribLocation(	gShaderProgramObject_perFragment, 
							AMC_ATTRIBUTE_POSITION, 
							"vPosition");
	glBindAttribLocation(	gShaderProgramObject_perFragment, 
							AMC_ATTRIBUTE_NORMAL, 
							"vNormal");

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
				printf("Per Fragment Error:\nprogram Link Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				exit(0);
			}
		}
	}

	//post linking retriving uniform location
	fragment.LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_LKeyPressed");

	fragment.model_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_model_matrix");
	fragment.view_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_view_matrix");
	fragment.projection_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_projection_matrix");

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
	printf("gNumVertices : %d\ngNumElements : %d\n", gNumVertices, gNumElements);
	

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

	//clear
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	
	//depth
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	//make orthograhic projection matrix a identity matrix
	perspectiveProjectionMatrix = vmath::mat4::identity();

	//warm-up resize call
	oglResize(giWindowWidth, giWindowHeight);
}	

void oglResize(int iWidth, int iHeight)
{
	//code
	if(iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	perspectiveProjectionMatrix = vmath::perspective(	45.0f, 
														((GLfloat)iWidth / (GLfloat)iHeight), 
														0.1f, 
														100.0f);
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
	vmath::mat4 translationMatrix_perFragment;
/*	
	vmath::mat4 rotationMatrixRed_perVertex;
	vmath::mat4 rotationMatrixGreen_perVertex;
	vmath::mat4 rotationMatrixBlue_perVertex;

	vmath::mat4 rotationMatrixRed_perFragment;
	vmath::mat4 rotationMatrixGreen_perFragment;
	vmath::mat4 rotationMatrixBlue_perFragment;
*/
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

	if (bPerVertex == true || bPerFragment == true)
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

	glXSwapBuffers(gpDisplay, gWindow);
}

void oglUpdate(void)
{
	//code
	LightAngle_red = LightAngle_red + 0.002f;
	if (LightAngle_red >= 360)
	{
		LightAngle_red = 0.0f;
	}

	LightAngle_green = LightAngle_green + 0.002f;
	if (LightAngle_green >= 360)
	{
		LightAngle_green = 0.0f;
	}

	LightAngle_blue = LightAngle_blue + 0.002f;
	if (LightAngle_blue >= 360)
	{
		LightAngle_blue = 0.0f;
	}
}

void oglUninitialise(void)
{
	//variable declaration
	GLXContext currentGLXContext;

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

	currentGLXContext = glXGetCurrentContext();

	if(currentGLXContext != NULL && currentGLXContext == gGLXContext)
	{
		glXMakeCurrent(gpDisplay, 0, 0);
	}

	if(gGLXContext)
	{
		glXDestroyContext(gpDisplay, gGLXContext);
	}

	if(gWindow)
	{
		XDestroyWindow(gpDisplay, gWindow);
		gWindow = 0;
	}

	if(gColormap)
	{
		XFreeColormap(gpDisplay, gColormap);
		gColormap = 0;
	}
	
	if(gpXVisualInfo)
	{
		free(gpXVisualInfo);
		gpXVisualInfo = NULL;
	}

	if(gpDisplay)
	{
		XCloseDisplay(gpDisplay);
		gpDisplay = NULL;
	}
}
