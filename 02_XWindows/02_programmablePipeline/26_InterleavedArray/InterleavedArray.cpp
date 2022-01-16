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

//texture related header files
#include <SOIL/SOIL.h>

//mathematics related header files
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
GLXFBConfig gGLXFBConfig;

//global variables related to shaders
GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

GLuint vao_cube;
GLuint vbo_cube;

GLuint mvpUniform;
vmath::mat4 perspectiveProjectionMatrix;

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
GLuint samplerUniform;

GLfloat rotation_angle_rectangle = 0.0f;

//light values
float LightAmbient[4] = { 0.25f, 0.25f, 0.25f, 0.25f };
float LightDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightPosition[4] = { 100.0f, 100.0f, 100.0f, 1.0f };			//{ 1.0f, 1.0f, 1.0f, 1.0f };

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 128.0f;							//{128.0f};

//flags 
bool bLight = false;

bool bAnimate = false;

//texture variable
GLuint marble_texture;

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

						case 'A':
						case 'a':

							if(bAnimate == false)
							{
								bAnimate = true;
							}
							else
							{
								bAnimate = false;
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
		if(bAnimate == true)
		{
			oglUpdate();	
		}
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

	XStoreName(gpDisplay, gWindow, "Programmable Pipeline : White Cube Template ");
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

	bool oglLoadTexture(GLuint *texture, const char *path)
	{
		//variable declaration
		bool bResult = false;
		int width = 0;
		int height = 0;
		unsigned char *pImageData = NULL;

		//code
		pImageData = SOIL_load_image(	path,			//path of an image
										&width,			//width will be given in this parameter
										&height,		//height will be given in this parameter
										0,				//i g=have nothing to give your (No masking)
										SOIL_LOAD_RGB);

		if(pImageData == NULL)
		{
			printf("oglLoadTexture() : ERROR : pImageData is NULL.\n");
			bResult = false;
			return(bResult);
		}
		else
		{
			bResult = true;
		}

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		//generate the memory on the device
		glGenTextures(1, texture);

		//bind with appropriate structure
		glBindTexture(GL_TEXTURE_2D, *texture);

		//set parameters
		glTexParameteri(	GL_TEXTURE_2D,
							GL_TEXTURE_MAG_FILTER,
							GL_LINEAR);

		glTexParameteri(	GL_TEXTURE_2D,
							GL_TEXTURE_MIN_FILTER,
							GL_LINEAR_MIPMAP_LINEAR);

		//fill the data
		glTexImage2D(	GL_TEXTURE_2D,
						0,				//image level
						GL_RGB,			//internal image format
						width,
						height,
						0,				//border width
						GL_RGB,			//external format / Pixel Format / External Image Format
						GL_UNSIGNED_BYTE,
						pImageData);

		glGenerateMipmap(GL_TEXTURE_2D);

		//glBindTexture(GL_TEXTURE_2D, 0);

		//free image data
		SOIL_free_image_data(pImageData);

		return(bResult);
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

	/* Vertex Shader code */

	//define the vaertex shader object
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//write vertex shader code
	const GLchar* vertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec4 vColor;" \
		"in vec3 vNormal;" \
		"in vec2 vTexcoord;" \
		
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_LKeyPressed;" \
		"uniform vec4 u_light_position;" \

		"out vec4 out_color;" \
		"out vec2 out_texcoord;" \
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
		"out_color = vColor;" \
		"out_texcoord = vTexcoord;" \
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

	//write shader code
	const GLchar* fragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 out_color;" \
		"in vec2 out_texcoord;" \
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
		"uniform sampler2D u_sampler;" \
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
		"phong_ads_light = (ambient + diffuse + specular) * vec3(out_color * texture(u_sampler, out_texcoord));" \
		"}" \
		"else" \
		"{" \
		"phong_ads_light = vec3(out_color * texture(u_sampler, out_texcoord));" \
		"}" \
		"FragColor =  vec4(phong_ads_light,1.0);" \
		"}";

		//"FragColor = (vec4(phong_ads_light, 1.0) *  texture * out_color);" \
	//Specify The code to the fragment shader object
	glShaderSource(	gFragmentShaderObject,
					1,
					(const GLchar **)&fragmentShaderSourceCode,
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
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_COLOR, "vColor");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_NORMAL, "vNormal");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_TEXCOODR_0, "vTexCoord");

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
	samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");

	GLfloat cubeVCNT[] =
	{
		1.0f, 1.0f, -1.0f,	1.0f, 0.0f, 0.0f,	0.0f, 1.0f,	0.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, -1.0f,	1.0f, 0.0f, 0.0f,	0.0f, 0.0f,	0.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 1.0f,	1.0f, 0.0f, 0.0f,	1.0f, 0.0f,	0.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 1.0f,	1.0f, 0.0f, 0.0f,	1.0f, 1.0f,	0.0f, 1.0f, 0.0f,

		1.0f, -1.0f, -1.0f,	0.0f, 1.0f, 0.0f,	1.0f, 1.0f,	0.0f, -1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,0.0f, 1.0f, 0.0f,	0.0f, 1.0f,	0.0f, -1.0f, 0.0f,
		-1.0f, -1.0f, 1.0f,	0.0f, 1.0f, 0.0f,	0.0f, 0.0f,	0.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,	0.0f, 1.0f, 0.0f,	1.0f, 0.0f,	0.0f, -1.0f, 0.0f,

		1.0f, 1.0f, 1.0f,	0.0f, 0.0f, 1.0f,	0.0f, 0.0f,	0.0f, 0.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,	0.0f, 0.0f, 1.0f,	1.0f, 0.0f,	0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,	0.0f, 0.0f, 1.0f,	1.0f, 1.0f,	0.0f, 0.0f, 1.0f,
		1.0f, -1.0f, 1.0f,	0.0f, 0.0f, 1.0f,	0.0f, 1.0f,	0.0f, 0.0f, 1.0f,

		1.0f, 1.0f, -1.0f,	0.0f, 1.0f, 1.0f,	1.0f, 0.0f,	0.0f, 0.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,	0.0f, 1.0f, 1.0f,	1.0f, 1.0f,	0.0f, 0.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,0.0f, 1.0f, 1.0f,	0.0f, 1.0f,	0.0f, 0.0f, -1.0f,
		1.0f, -1.0f, -1.0f,	0.0f, 1.0f, 1.0f,	0.0f, 0.0f,	0.0f, 0.0f, -1.0f,

		1.0f, 1.0f, -1.0f,	1.0f, 0.0f, 1.0f,	1.0f, 0.0f,	1.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 1.0f,	1.0f, 0.0f, 1.0f,	1.0f, 1.0f,	1.0f, 0.0f, 0.0f,
		1.0f, -1.0f, 1.0f,	1.0f, 0.0f, 1.0f,	0.0f, 1.0f,	1.0f, 0.0f, 0.0f,
		1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 1.0f,	0.0f, 0.0f,	1.0f, 0.0f, 0.0f,

		-1.0f, 1.0f, -1.0f,	1.0f, 1.0f, 0.0f,	0.0f, 0.0f,	-1.0f, 0.0f, 0.0f,
		-1.0f, 1.0f, 1.0f,	1.0f, 1.0f, 0.0f,	1.0f, 0.0f,	-1.0f, 0.0f, 0.0f,
		-1.0f, -1.0f, 1.0f,	1.0f, 1.0f, 0.0f,	1.0f, 1.0f,	-1.0f, 0.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,1.0f, 1.0f, 0.0f,	0.0f, 1.0f,	-1.0f, 0.0f, 0.0f
	};

	glGenVertexArrays(1, &vao_cube);
	glBindVertexArray(vao_cube);
	
	glGenBuffers(1, &vbo_cube);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_cube);
	glBufferData(GL_ARRAY_BUFFER, 24 * 11 * sizeof(GLfloat), cubeVCNT, GL_STATIC_DRAW);

	//position
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	//color
	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);

	//texcoord
	glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOODR_0, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOODR_0);

	//normal
	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void*)(8 * sizeof(GLfloat)));
	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0); 

	glBindVertexArray(0);

	//clear
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	
	//depth
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	//texture
	glEnable(GL_TEXTURE_2D);
	oglLoadTexture(&marble_texture, "marble.bmp");
	if (marble_texture == 0)
	{
		printf("Marble texture is 0\n");
		oglUninitialise();
		exit(0);
	}

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
	vmath::mat4 modelMatrix;
	vmath::mat4 viewMatrix;
	vmath::mat4 projectionMatrix;
	vmath::mat4 translationMatrix;
	vmath::mat4 rotationMatrix;

	//init above metrices to identity
	modelMatrix = vmath::mat4::identity();
	viewMatrix = vmath::mat4::identity();
	projectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();

	//do necessary transformations here
	translationMatrix = vmath::translate(0.0f, 0.0f, -6.0f);
	rotationMatrix = vmath::rotate(rotation_angle_rectangle, rotation_angle_rectangle, rotation_angle_rectangle);

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
	glBindTexture(GL_TEXTURE_2D, marble_texture);

	//push in fragment
	glUniform1i(samplerUniform, 0);

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
		glUniform3fv(Ka_uniform, 1, MaterialAmbient);
		glUniform3fv(Kd_uniform, 1, MaterialDiffuse);
		glUniform3fv(Ks_uniform, 1, MaterialSpecular);
		//shininess
		glUniform1f(shininess_uniform, MaterialShininess);
		//send light position
		glUniform4fv(lightPosition_uniform, 1, LightPosition);
	}
	else
	{
		//notify shader that we aren't pressed the "L" key
		glUniform1i(LKeyPressed_Uniform, 0);
	}

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

	//unuse program
	glUseProgram(0);

	glXSwapBuffers(gpDisplay, gWindow);
}

void oglUpdate(void)
{
	//code
	rotation_angle_rectangle = rotation_angle_rectangle + 0.5f;
	if (rotation_angle_rectangle >= 360.0f)
	{
		rotation_angle_rectangle = 0.0f;
	}
}

void oglUninitialise(void)
{
	//variable declaration
	GLXContext currentGLXContext;

	//code

	if (vbo_cube)
	{
		glDeleteBuffers(1, &vbo_cube);
		vbo_cube = 0;
	}
	if (vao_cube)
	{
		glDeleteVertexArrays(1, &vao_cube);
		vao_cube = 0;
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

