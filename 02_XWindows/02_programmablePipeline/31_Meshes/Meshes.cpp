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

struct buffers
{
	GLuint vao;
	GLuint vbo_position;
	GLuint vbo_color;
}one, two, three, four, five, six;
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

	XStoreName(gpDisplay, gWindow, "Programmable Pipeline : Meshes");
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

	/* Vertex Shader code */

	//define the vaertex shader object
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

	//specify above source code to vertex shader
	glShaderSource(	gVertexShaderObject, 
					1,
					(const GLchar **)&vertexShaderSourceCode,
					NULL);

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

	glGetShaderiv(	gVertexShaderObject,
					GL_COMPILE_STATUS,
					&iShaderCompileStatus);
	if(iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(	gVertexShaderObject,
						GL_INFO_LOG_LENGTH,
						&iInfoLogLength);
		if(iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(	gVertexShaderObject,
									iInfoLogLength,
									&written,
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

	//write code of fragment shader
	const GLchar *fragementShaderSourceCode = 
	"#version 450 core" \
	"\n" \
	"in vec4 out_color;" \
	"out vec4 FragColor;" \
	"void main(void)" \
	"{" \
		"FragColor = out_color;" \
	"}";

	//Specify The code to the fragment shader object
	glShaderSource(	gFragmentShaderObject,
					1,
					(const GLchar **)&fragementShaderSourceCode,
					NULL);

	//compile the shader
	glCompileShader(gFragmentShaderObject);

	//error checking
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(	gFragmentShaderObject,
					GL_COMPILE_STATUS,
					&iShaderCompileStatus);

	if(iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(	gFragmentShaderObject,
						GL_INFO_LOG_LENGTH,
						&iInfoLogLength);

		if(iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(	gFragmentShaderObject,
									iInfoLogLength,
									&written,
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

	//attach vertex shader
	glAttachShader(	gShaderProgramObject, gVertexShaderObject );

	//attach fragment shader
	glAttachShader(	gShaderProgramObject, gFragmentShaderObject );

	//pre linking bonding to vertex attributes
	glBindAttribLocation(	gShaderProgramObject,
							AMC_ATTRIBUTE_POSITION,
							"vPosition");
	glBindAttribLocation(	gShaderProgramObject,
							AMC_ATTRIBUTE_COLOR,
							"vCOlor");

	//link shader program
	glLinkProgram(gShaderProgramObject);

	//error checking
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(	gShaderProgramObject,
					GL_LINK_STATUS,
					&iProgramLinkStatus);

	if(iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(	gShaderProgramObject,
						GL_INFO_LOG_LENGTH,
						&iInfoLogLength);

		if(iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(	gShaderProgramObject,
										iInfoLogLength,
										&written,
										szInfoLog);

				printf("program Link Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				exit(0);
			}
		}
	}

	//post linking retiving uniform location
	mvpUniform = glGetUniformLocation(	gShaderProgramObject, 
										"u_mvp_matrix");

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

	//clear window 
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

