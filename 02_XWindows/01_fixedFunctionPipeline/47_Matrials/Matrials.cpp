//Standard Header Files
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#define _USE_MATH_DEFINES 1
#include<math.h>

//X11 Related Header Files
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>

//opengl Related Header files
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>

//namespaces 
using namespace std;

//global variable declaraton
bool			bFullScreen		=	false;
Display			*gpDisplay		= 	NULL;
XVisualInfo		*gpXVisualInfo	=	NULL;
Colormap		gColormap;
Window			gWindow;
int 			giWindowWidth	=	800;
int 			giWindowHeight	=	600;

//OpenGL Related Global Variable declaration
static GLXContext gGLXContext;

/* Materials */

//Light Related Variables
bool bLight = false;

//Light Configuration Declarations
GLfloat LightAmbient[] 		= 	{ 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat LightDiffuse[] 		= 	{ 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat LightPosition[] 	= 	{ 0.0f, 0.0f, 0.0f, 1.0f }; 

GLfloat light_model_ambient[] 		=	{ 0.2f, 0.2f, 0.2f, 1.0f };
GLfloat light_model_local_viewer[]	=	{ 0.0f };

GLUquadric *quadric[24];	

GLfloat anglOfXPosition = 0.0f;
GLfloat anglOfYPosition = 0.0f;
GLfloat anglOfZPosition = 0.0f;

GLint keyPress = 0;

//entry-point funtion
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
	bool	bDone		=	false;
	int 	winWidth 	= 	giWindowWidth;
	int 	winHeight 	=	giWindowHeight;
	char 	keys[26];

	//code
	oglCreateWindow();

	//initialise
	oglInitialise();

	//MessageLoop
	XEvent event;
	KeySym keysym;

	while(bDone == false)
	{
		while(XPending(gpDisplay))
		/* Here XPending Checks Whether There Is Any Pending Messages Which Were Came From Server(XServer) In Pool */
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
						case 'F':
						case 'f':
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
						case 'L':
						case 'l':
							if(bLight == false)
							{
								bLight = true;
								glEnable(GL_LIGHTING);
							}
							else
							{
								bLight = false;
								glDisable(GL_LIGHTING);
							}
						break;
						case 'x':
						case 'X':
							keyPress = 1;
							anglOfXPosition = 0.0f;
						break;

						case 'y':
						case 'Y':
							keyPress = 2;
							anglOfYPosition = 0.0f;
						break;

						case 'z':
						case 'Z':
							keyPress = 3;
							anglOfZPosition = 0.0f;
						break;
						default:
						break;
					}
				break;

				case ButtonPress:
					switch(event.xbutton.button)
					{
						case 1:
							//code
						break;

						case 2:
							//code
						break;

						case 3:
							//code
						break;

						default:
							//code
						break;
					}
				break;

				case MotionNotify:
					//code
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
	Screen 					*screen;
	int 					defaultScreen;
	int 					defaultDepth;
	int 					styleMask;
	int 					screen_width;
	int 					screen_height;
	int 					screen_count;
	int 					x, y;

	//OpenGL Specific Local Variables
	static int frameBufferAttributes[]	=	{	GLX_RGBA,
												GLX_DOUBLEBUFFER,
												GLX_RED_SIZE, 8,
												GLX_GREEN_SIZE, 8,
												GLX_BLUE_SIZE, 8,
												GLX_ALPHA_SIZE, 8,
												GLX_DEPTH_SIZE, 24,
												None};

	//code
	gpDisplay 		= 	XOpenDisplay(NULL);
	if(gpDisplay == NULL)
	{
		printf("ERROR : Unable To Open X Display.\n Exitting Now....\n");
		oglUninitialise();
		exit(1);
	}

	defaultScreen 	= 	XDefaultScreen(gpDisplay);

	defaultDepth 	=	DefaultDepth(gpDisplay, defaultScreen);

	gpXVisualInfo 	=	glXChooseVisual(	gpDisplay, 
											defaultScreen, 
											frameBufferAttributes);
	if(gpXVisualInfo == NULL)
	{
		printf("glxChooseVisual Failed.\nExitting Now...\n");
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
	gColormap						=	winAttribs.colormap;
	
	winAttribs.event_mask			= 	ExposureMask | VisibilityChangeMask | ButtonPressMask | KeyPressMask | PointerMotionMask | 	StructureNotifyMask;

	styleMask						=	CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;

	gWindow 						= 	XCreateWindow(	gpDisplay,
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

	XStoreName(gpDisplay, gWindow, "Spinning Plain Pyramid With Red And Blue Light");
	Atom windowManagerDelete	=	XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);
	XSetWMProtocols (gpDisplay, gWindow, &windowManagerDelete, 1);
	
	screen_count = ScreenCount(gpDisplay);

	for(int i = 0; i < screen_count; i++)
	{
		screen = ScreenOfDisplay(gpDisplay, i);
	}

	screen_width 	= 	screen->width;
	screen_height 	=	screen->height;

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
	//function declaration
	void oglUninitialise(void);
	void oglResize(int, int);

	//code
	gGLXContext = glXCreateContext(	gpDisplay,
									gpXVisualInfo,
									NULL,
									GL_TRUE);

	glXMakeCurrent(	gpDisplay, gWindow, gGLXContext);


	//3D 
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//Lights
	glEnable(GL_AUTO_NORMAL);
	glEnable(GL_NORMALIZE);

	glLightfv(GL_LIGHT0, GL_AMBIENT, LightAmbient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_model_ambient);
	glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, light_model_local_viewer);
	glEnable(GL_LIGHT0);

	for (int i = 0; i < 24; i++)
	{
		quadric[i] = gluNewQuadric();
	}
	
	//warm-up resize call  
	oglResize(giWindowWidth, giWindowHeight);
}	

void oglResize(int iWidth, int iHeight)
{
	//code
	if (iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	if (iWidth <= iHeight)
	{
		glOrtho(0.0f,
				15.5f,
				0.0f,
				(15.5f * ((GLfloat)iHeight / (GLfloat)iWidth)),
				-10.0f,
				10.0f);
	}
	else
	{
		glOrtho(0.0f,
				(15.5f * ((GLfloat)iWidth / (GLfloat)iHeight)),
				0.0f,
				15.5f,
				-10.0f,
				10.0f);
	}
}

void oglDisplay(void)
{	
	//FUNCTION PROTOTYPE
	void draw24Spheres(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	if (keyPress == 1)
	{
		glRotatef(anglOfXPosition, 1.0f, 0.0f, 0.0f);
		LightPosition[1] = anglOfXPosition;
	}
	else if (keyPress == 2)
	{
		glRotatef(anglOfYPosition, 0.0f, 1.0f, 0.0f);
		LightPosition[2] = anglOfYPosition;
	}
	else
	{
		glRotatef(anglOfZPosition, 0.0f, 0.0f, 1.0f);
		LightPosition[0] = anglOfZPosition;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);

	draw24Spheres();

	glXSwapBuffers(gpDisplay, gWindow);
}

void oglUpdate(void)
{
	//todo
	anglOfXPosition = anglOfXPosition + 0.2f;
	if (anglOfXPosition > 360.0f)
	{
		anglOfXPosition = 0.0f;
	}
	anglOfYPosition = anglOfYPosition + 0.2f;
	if (anglOfYPosition > 360.0f)
	{
		anglOfYPosition = 0.0f;
	}
	anglOfZPosition = anglOfZPosition + 0.2f;
	if (anglOfZPosition > 360.0f)
	{
		anglOfZPosition = 0.0f;
	}
}

void oglUninitialise(void)
{
	//variable declaration
	GLXContext currentGLXContext;

	//code
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

void draw24Spheres(void)
{
	//variable
	GLfloat MaterialAmbient[4];
	GLfloat MaterialSpecular[4];
	GLfloat MaterialDiffuse[4];
	GLfloat MaterialShininess[1];
	
	//code
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//-------------------------------------Emerald-------------------------------------//
	MaterialAmbient[0] = 0.0215f;
	MaterialAmbient[1] = 0.1745f;
	MaterialAmbient[2] = 0.0215f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.07568f;
	MaterialDiffuse[1] = 0.61424f;
	MaterialDiffuse[2] = 0.07568f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.633f;
	MaterialSpecular[1] = 0.727811f;
	MaterialSpecular[2] = 0.633f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.6f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 14.0f, 0.0f);
	gluSphere(quadric[0], 1.0f, 30, 30);

	//-------------------------------------Jade-------------------------------------//
	MaterialAmbient[0] = 0.135f;
	MaterialAmbient[1] = 0.2225f;
	MaterialAmbient[2] = 0.1575f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.54f;
	MaterialDiffuse[1] = 0.89f;
	MaterialDiffuse[2] = 0.63f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.316228f;
	MaterialSpecular[1] = 0.316228f;
	MaterialSpecular[2] = 0.316228f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.1f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 11.5f, 0.0f);
	gluSphere(quadric[1], 1.0f, 30, 30);

	//-------------------------------------Obsidian-------------------------------------//
	MaterialAmbient[0] = 0.05375f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.06625f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.18275f;
	MaterialDiffuse[1] = 0.17f;
	MaterialDiffuse[2] = 0.22525f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.332741f;
	MaterialSpecular[1] = 0.328634f;
	MaterialSpecular[2] = 0.346435f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.3f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 9.0f, 0.0f);
	gluSphere(quadric[2], 1.0f, 30, 30);

	//-------------------------------------Pearl-------------------------------------//
	MaterialAmbient[0] = 0.25f;
	MaterialAmbient[1] = 0.20725f;
	MaterialAmbient[2] = 0.20725f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 1.0f;
	MaterialDiffuse[1] = 0.829f;
	MaterialDiffuse[2] = 0.829f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.296648f;
	MaterialSpecular[1] = 0.296648f;
	MaterialSpecular[2] = 0.296648f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.088f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 6.5f, 0.0f);
	gluSphere(quadric[3], 1.0f, 30, 30);

	//-------------------------------------Ruby-------------------------------------//
	MaterialAmbient[0] = 0.1745f;
	MaterialAmbient[1] = 0.01175f;
	MaterialAmbient[2] = 0.01175f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.61424f;
	MaterialDiffuse[1] = 0.04136f;
	MaterialDiffuse[2] = 0.04136f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.727811f;
	MaterialSpecular[1] = 0.626959f;
	MaterialSpecular[2] = 0.626959f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.6f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 4.0f, 0.0f);
	gluSphere(quadric[4], 1.0f, 30, 30);

	//-------------------------------------Turquoise-------------------------------------//
	MaterialAmbient[0] = 0.1f;
	MaterialAmbient[1] = 0.18725f;
	MaterialAmbient[2] = 0.1745f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.396f;
	MaterialDiffuse[1] = 0.74151f;
	MaterialDiffuse[2] = 0.69102f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.297254f;
	MaterialSpecular[1] = 0.30829f;
	MaterialSpecular[2] = 0.306678f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.1f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(7.75f, 1.5f, 0.0f);
	gluSphere(quadric[5], 1.0f, 30, 30);

	//-------------------------------------Brass-------------------------------------//
	MaterialAmbient[0] = 0.329412f;
	MaterialAmbient[1] = 0.223529f;
	MaterialAmbient[2] = 0.027451f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.782392f;
	MaterialDiffuse[1] = 0.568627f;
	MaterialDiffuse[2] = 0.113725f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.992157f;
	MaterialSpecular[1] = 0.941176f;
	MaterialSpecular[2] = 0.807843f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.21794872f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 14.0f, 0.0f);
	gluSphere(quadric[6], 1.0f, 30, 30);

	//-------------------------------------Bronze-------------------------------------//
	MaterialAmbient[0] = 0.2125f;
	MaterialAmbient[1] = 0.1275f;
	MaterialAmbient[2] = 0.054f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.714f;
	MaterialDiffuse[1] = 0.4284f;
	MaterialDiffuse[2] = 0.18144f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.393548f;
	MaterialSpecular[1] = 0.271906f;
	MaterialSpecular[2] = 0.166721f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.2f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 11.5f, 0.0f);
	gluSphere(quadric[7], 1.0f, 30, 30);

	//-------------------------------------Chrome-------------------------------------//
	MaterialAmbient[0] = 0.25f;
	MaterialAmbient[1] = 0.25f;
	MaterialAmbient[2] = 0.25f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.4f;
	MaterialDiffuse[1] = 0.4f;
	MaterialDiffuse[2] = 0.4f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.774597f;
	MaterialSpecular[1] = 0.774597f;
	MaterialSpecular[2] = 0.774597f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.6f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 9.0f, 0.0f);
	gluSphere(quadric[8], 1.0f, 30, 30);

	//-------------------------------------Copper-------------------------------------//
	MaterialAmbient[0] = 0.19125f;
	MaterialAmbient[1] = 0.0735f;
	MaterialAmbient[2] = 0.0225f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.7038f;
	MaterialDiffuse[1] = 0.27048f;
	MaterialDiffuse[2] = 0.0828f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.256777f;
	MaterialSpecular[1] = 0.137622f;
	MaterialSpecular[2] = 0.086014f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.1f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 6.5f, 0.0f);
	gluSphere(quadric[9], 1.0f, 30, 30);

	//-------------------------------------Gold-------------------------------------//
	MaterialAmbient[0] = 0.24725f;
	MaterialAmbient[1] = 0.1995f;
	MaterialAmbient[2] = 0.0745f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.75164f;
	MaterialDiffuse[1] = 0.60648f;
	MaterialDiffuse[2] = 0.22648f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.628281f;
	MaterialSpecular[1] = 0.555802f;
	MaterialSpecular[2] = 0.366065f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.4f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 4.0f, 0.0f);
	gluSphere(quadric[10], 1.0f, 30, 30);


	//-------------------------------------Silver-------------------------------------//
	MaterialAmbient[0] = 0.19225f;
	MaterialAmbient[1] = 0.19225f;
	MaterialAmbient[2] = 0.19225f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.50754f;
	MaterialDiffuse[1] = 0.50754f;
	MaterialDiffuse[2] = 0.50754f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.508273f;
	MaterialSpecular[1] = 0.508273f;
	MaterialSpecular[2] = 0.508273f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.4f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(11.625f, 1.5f, 0.0f);
	gluSphere(quadric[11], 1.0f, 30, 30);

	//-------------------------------------Black Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.01f;
	MaterialDiffuse[1] = 0.01f;
	MaterialDiffuse[2] = 0.01f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.50f;
	MaterialSpecular[1] = 0.50f;
	MaterialSpecular[2] = 0.50f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 14.0f, 0.0f);
	gluSphere(quadric[12], 1.0f, 30, 30);

	//-------------------------------------Cyan Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.1f;
	MaterialAmbient[2] = 0.06f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.01f;
	MaterialDiffuse[1] = 0.50980392f;
	MaterialDiffuse[2] = 0.50980392f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.50196078f;
	MaterialSpecular[1] = 0.50196078f;
	MaterialSpecular[2] = 0.50196078f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 11.5f, 0.0f);
	gluSphere(quadric[13], 1.0f, 30, 30);

	//-------------------------------------Green Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.1f;
	MaterialDiffuse[1] = 0.35f;
	MaterialDiffuse[2] = 0.1f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.45f;
	MaterialSpecular[1] = 0.55f;
	MaterialSpecular[2] = 0.45f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 9.0f, 0.0f);
	gluSphere(quadric[14], 1.0f, 30, 30);

	//-------------------------------------Red Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.0f;
	MaterialDiffuse[2] = 0.0f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.7f;
	MaterialSpecular[1] = 0.6f;
	MaterialSpecular[2] = 0.6f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 6.5f, 0.0f);
	gluSphere(quadric[15], 1.0f, 30, 30);

	//-------------------------------------White Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.55f;
	MaterialDiffuse[1] = 0.55f;
	MaterialDiffuse[2] = 0.55f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.70f;
	MaterialSpecular[1] = 0.70f;
	MaterialSpecular[2] = 0.70f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 4.0f, 0.0f);
	gluSphere(quadric[16], 1.0f, 30, 30);

	//-------------------------------------White Plastic-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.0f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.60f;
	MaterialSpecular[1] = 0.60f;
	MaterialSpecular[2] = 0.50f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.25f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(15.5f, 1.5f, 0.0f);
	gluSphere(quadric[17], 1.0f, 30, 30);

	//-------------------------------------Black Rubber-------------------------------------//
	MaterialAmbient[0] = 0.02f;
	MaterialAmbient[1] = 0.02f;
	MaterialAmbient[2] = 0.02f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.01f;
	MaterialDiffuse[1] = 0.01f;
	MaterialDiffuse[2] = 0.01f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.4f;
	MaterialSpecular[1] = 0.4f;
	MaterialSpecular[2] = 0.4f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 14.0f, 0.0f);
	gluSphere(quadric[18], 1.0f, 30, 30);

	//-------------------------------------Cyan Rubber-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.05f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.4f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.5f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.04f;
	MaterialSpecular[1] = 0.7f;
	MaterialSpecular[2] = 0.7f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 11.5f, 0.0f);
	gluSphere(quadric[19], 1.0f, 30, 30);

	//-------------------------------------Green Rubber-------------------------------------//
	MaterialAmbient[0] = 0.0f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.4f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.4f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.04f;
	MaterialSpecular[1] = 0.7f;
	MaterialSpecular[2] = 0.04f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 9.0f, 0.0f);
	gluSphere(quadric[20], 1.0f, 30, 30);

	//-------------------------------------Red Rubber-------------------------------------//
	MaterialAmbient[0] = 0.05f;
	MaterialAmbient[1] = 0.0f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.4f;
	MaterialDiffuse[2] = 0.4f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.7f;
	MaterialSpecular[1] = 0.04f;
	MaterialSpecular[2] = 0.04f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 6.5f, 0.0f);
	gluSphere(quadric[21], 1.0f, 30, 30);

	//-------------------------------------White Rubber-------------------------------------//
	MaterialAmbient[0] = 0.05f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.05f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.5f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.7f;
	MaterialSpecular[1] = 0.7f;
	MaterialSpecular[2] = 0.7f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 4.0f, 0.0f);
	gluSphere(quadric[22], 1.0f, 30, 30);

	//-------------------------------------Yellow Rubber-------------------------------------//
	MaterialAmbient[0] = 0.05f;
	MaterialAmbient[1] = 0.05f;
	MaterialAmbient[2] = 0.0f;
	MaterialAmbient[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_AMBIENT, MaterialAmbient);

	MaterialDiffuse[0] = 0.5f;
	MaterialDiffuse[1] = 0.5f;
	MaterialDiffuse[2] = 0.4f;
	MaterialDiffuse[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MaterialDiffuse);

	MaterialSpecular[0] = 0.7f;
	MaterialSpecular[1] = 0.7f;
	MaterialSpecular[2] = 0.04f;
	MaterialSpecular[3] = 1.0f;
	glMaterialfv(GL_FRONT, GL_SPECULAR, MaterialSpecular);

	MaterialShininess[0] = 0.078125f * 128;
	glMaterialfv(GL_FRONT, GL_SHININESS, MaterialShininess);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(19.375f, 1.5f, 0.0f);
	gluSphere(quadric[23], 1.0f, 30, 30);
}

