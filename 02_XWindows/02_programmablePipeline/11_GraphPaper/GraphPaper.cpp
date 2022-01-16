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

//our header files
#include "vmath.h"

//namespaces
using namespace std;

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
GLXFBConfig gGLXFBconfig;


//global variables related to shaders
GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

GLuint vao_red;
GLuint vao_green;
GLuint vao_blue;
GLuint vbo_red_line_position;
GLuint vbo_red_line_color;
GLuint vbo_green_line_position;
GLuint vbo_green_line_color;
GLuint vbo_blue_line_position;
GLuint vbo_blue_line_color;
GLuint mvpUniform;
vmath::mat4 perspectiveProjectionMatrix;

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
							bDone = true;
						break;

						default:
						break;
					}
					XLookupString(&event.xkey, keys, sizeof(keys), NULL, NULL);
					switch(keys[0])
					{
						case'F':
						case'f':
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

	//Screen centering parameters
	Screen 					*screen;
	int 					screen_width;
	int 					screen_height;
	int 					screen_count;
	int 					x, y;

	//FBConfig Related parameters
	GLXFBConfig *pGLXFBConfig = NULL;
	GLXFBConfig bestGLXFBConfig;
	XVisualInfo *pTempXVisualInfo = NULL;
	int iNumberOfFBConfigs = 0;

	int bestFrameBufferConfig = -1;
	int bestNumberOfSamples = -1;
	int worstFrameBufferConfig = -1;
	int worstNumberOfSamples = 999;

	//code
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
										&iNumberOfFBConfigs);

	for(int i = 0; i < iNumberOfFBConfigs; i++)
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

			if(bestFrameBufferConfig < 0 || (sampleBuffers && samples > bestNumberOfSamples))
			{
				bestFrameBufferConfig = i;
				bestNumberOfSamples = samples;
			}

			if(worstFrameBufferConfig < 0 || !sampleBuffers || samples < worstNumberOfSamples)
			{
				worstFrameBufferConfig = i;
				worstNumberOfSamples = samples;
			}
		}//pTempXVisualInfo
		XFree(pTempXVisualInfo);
	}

	//	7.	assign the found best one
	bestGLXFBConfig = pGLXFBConfig[bestFrameBufferConfig];

	//	8.	assign the same best to global one
	gGLXFBconfig = bestGLXFBConfig;

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

	XStoreName(gpDisplay, gWindow, "Programmable Pipeline : Graph Paper");
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
	//function prototype
	//void oglResize(int, int);

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

	//context attrib array declaration
	const int Attribs[] = 
	{
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};

	//now get the context
	gGLXContext = glXCreateContextAttribsARB(	gpDisplay,
												gGLXFBconfig,
												0,
												True,
												Attribs);

	if(!gGLXContext)
	{
		const int Attribs[] = 
		{
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext = glXCreateContextAttribsARB(	gpDisplay,
													gGLXFBconfig,
													0,
													True,
													Attribs);
	}

	//check whether obtained context is hardwae rendering context or not
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
				printf("Vertex Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
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
				printf("Fragment Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
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
				printf("program Link Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
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
	glUseProgram(gShaderProgramObject);

	//declaration of metrices
	vmath::mat4 modelViewMatrix;
	vmath::mat4 modelViewProjectionMatrix;

	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();

	//do necessary transformations here
	modelViewMatrix = vmath::translate(0.0f, 0.0f, -1.2f);

	//do necessary matrix multiplication
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	//bind with vao
	glBindVertexArray(vao_red);

	//draw scene
	glDrawArrays(GL_LINES, 0, 2);

	//unbind vao
	glBindVertexArray(0);

	//bind with vao
	glBindVertexArray(vao_green);

	//draw scene
	glDrawArrays(GL_LINES, 0, 2);

	//unbind vao
	glBindVertexArray(0);

	//bind with vao
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
	
	//unbind vao
	glBindVertexArray(0);

	//unuse program
	glUseProgram(0);

	glXSwapBuffers(gpDisplay, gWindow);
}

void oglUpdate(void)
{
	//todo
}

void oglUninitialise(void)
{
	//variable declaration
	GLXContext currentGLXContext;

	//code
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
	if (vbo_blue_line_color)
	{
		glDeleteBuffers(1, &vbo_blue_line_color);
		vbo_blue_line_color = 0;
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
