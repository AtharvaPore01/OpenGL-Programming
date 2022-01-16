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
int 			giWindowWidth	=	1280;
int 			giWindowHeight	=	720;

//OpenGL Related Global Variable declaration
static GLXContext gGLXContext;

//Dynamic India Related GLobal Variables

//I
bool 		bITrue 						=	false;
GLfloat 	x_I1						=	-3.0f;

//A
bool		bATrue 						=	false;
GLfloat		x_A 						=	3.0f;
GLfloat 	r_A							=	0.0f;
GLfloat 	g_A							=	0.0f;
GLfloat 	b_A							=	0.0f;
GLfloat 	White_A						=	0.0f;

//N
bool 		bNTrue 						=	false;
GLfloat 	y_N							=	3.0f;

//I2
bool		bI2True						=	false;
GLfloat 	y_I2						=	-3.0f;

//D
bool		bDTrue						=	false;
GLfloat 	r 							= 	0.0f;
GLfloat 	g 							= 	0.0f;
GLfloat 	b 							= 	0.0f;

//Plane
bool		bPlaneTrue					=	false;
bool		bTopPlaneAtOrigin			=	false;
bool		bClipTopPlane				=	false;
bool		bClipBottomPlane			=	false;

GLfloat		x_plane						=	-22.0f;
GLfloat 	top_plane_rotate_angle		=	-60.0f;
GLfloat 	bottom_plane_rotate_angle	=	60.0f;

//Tricolour
bool 		bColourDone					=	false;
bool 		bStartIncrementingTop		=	false;
bool 		bStartDecrementingBottom 	=	false;

GLfloat 	x_CoordinateOfStrips_Orange = 	-12.0f;
GLfloat 	x_CoordinateOfStrips_White 	= 	-12.0f;
GLfloat 	x_CoordinateOfStrips_Green 	= 	-12.0f;

//Dynamic India Related Structures
struct TopPlane
{
	GLfloat x;
	GLfloat y;
	GLfloat Radius	=	10.0f;
	GLfloat angle 	=	M_PI;
}top;

struct BottomPlane
{
	GLfloat x;
	GLfloat y;
	GLfloat Radius	=	10.0f;
	GLfloat angle 	=	M_PI;
}bottom;

struct TriColors
{
	//top
	GLfloat angle_1top = 3.14159f;
	GLfloat angle_2top = 3.14659f;

	GLfloat angle_3top = 4.71238f;
	GLfloat angle_4top = 4.71738f;
	//bottom
	GLfloat angle_1bottom = 3.13659f;
	GLfloat angle_2bottom = 3.14159f;
	
	GLfloat angle_3bottom = 1.57079f;
	GLfloat angle_4bottom = 1.56579f;
}clr;

struct RGB
{
	GLfloat r 				= 1.0f;
	GLfloat g 				= 0.5f;
	GLfloat b 				= 1.0f;
	GLfloat White 			= 1.0f;

	bool 	bMiddleDone 	= 	false;
	bool 	bTopDone 		= 	false;
	bool 	bBottomDone 	= 	false;

}top_clr, middle_clr, bottom_clr;

//entry-point funtion
int main(void)
{
	//function prototype
	void oglCreateWindow(void);
	void oglToggleFullscreen(void);
	void oglInitialise(void);
	void oglResize(int, int);
	void oglDisplay(void);
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
/*					
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
						default:
						break;
					}
*/
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
		//Here Call Update And Display
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

	XStoreName(gpDisplay, gWindow, "Dynamic India");
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
	void oglToggleFullscreen(void);

	//code
	gGLXContext = glXCreateContext(	gpDisplay,
									gpXVisualInfo,
									NULL,
									GL_TRUE);

	glXMakeCurrent(	gpDisplay, gWindow, gGLXContext);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	oglToggleFullscreen();
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

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(	45.0f,
					((GLfloat)iWidth / (GLfloat)iHeight),
					0.1f, 
					100.0f);
}

void oglDisplay(void)
{	
	//function prototype
	void dynamic_india(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);

	dynamic_india();

	glXSwapBuffers(gpDisplay, gWindow);
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

void OGL_I1(void)
{
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(-1.15f, 0.7f);
	glVertex2f(-1.25f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(-1.25f, -0.7f);
	glVertex2f(-1.15f, -0.7f);
	glEnd();
}
void OGL_N(void)
{
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(-0.95f, 0.7f);
	glVertex2f(-1.05f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(-1.05f, -0.7f);
	glVertex2f(-0.95f, -0.7f);
	glEnd();

	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(-0.55f, 0.7f);
	glVertex2f(-0.65f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(-0.65f, -0.7f);
	glVertex2f(-0.55f, -0.7f);
	glEnd();

	glLineWidth(15.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(-0.95f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(-0.65f, -0.7f);
	glEnd();
}
void OGL_D(GLfloat R, GLfloat G, GLfloat B)
{
	//top 
	glBegin(GL_QUADS);
	glColor3f(R, G, B);
	glVertex2f(0.15f, 0.7f);
	glVertex2f(-0.45f, 0.7f);
	glVertex2f(-0.45f, 0.6f);
	glVertex2f(0.15f, 0.6f);
	glEnd();
	//bottom
	glBegin(GL_QUADS);
	glColor3f(0.0f, G, 0.0f);
	glVertex2f(0.15f, -0.7f);
	glVertex2f(-0.45f, -0.7f);
	glVertex2f(-0.45f, -0.6f);
	glVertex2f(0.15f, -0.6f);
	glEnd();
	//left
	glBegin(GL_QUADS);
	glColor3f(R, G, B);
	glVertex2f(0.15f, 0.7f);
	glVertex2f(0.05f, 0.7f);
	glColor3f(0.0f, G, 0.0f);
	glVertex2f(0.05f, -0.7f);
	glVertex2f(0.15f, -0.7f);
	glEnd();
	//right
	glBegin(GL_QUADS);
	glColor3f(R, G, B);
	glVertex2f(-0.25f, 0.6f);
	glVertex2f(-0.35f, 0.6f);
	glColor3f(0.0f, G, 0.0f);
	glVertex2f(-0.35f, -0.6f);
	glVertex2f(-0.25f, -0.6f);
	glEnd();
}
void OGL_I2(void)
{
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(0.35f, 0.7f);
	glVertex2f(0.25f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(0.25f, -0.7f);
	glVertex2f(0.35f, -0.7f);
	glEnd();
}
void OGL_A(void)
{

	glBegin(GL_QUADS);
	
	//left
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(0.75f, 0.4f);
	glVertex2f(0.75f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(0.45f, -0.4f);
	glVertex2f(0.45f, -0.7f);

	//right
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(0.75f, 0.4f);
	glVertex2f(0.75f, 0.7f);
	glColor3f(0.0f, 0.5f, 0.0f);
	glVertex2f(1.05f, -0.4f);
	glVertex2f(1.05f, -0.7f);

	glEnd();

}

void OGL_A_TriColourStrips(GLfloat R, GLfloat G, GLfloat B, GLfloat White)
{
/*
	glBegin(GL_QUADS);
	//orange
	glColor3f(R, G, 0.0f);
	glVertex2f(0.872f, -0.05f);
	glVertex2f(0.628f, -0.05f);
	glVertex2f(0.616f, -0.1f);
	glVertex2f(0.882f, -0.1f);

	glColor3f(R, White, B);
	glVertex2f(0.882f, -0.1f);
	glVertex2f(0.616f, -0.1f);
	glVertex2f(0.605f, -0.15f);
	glVertex2f(0.895f, -0.15f);

	glColor3f(0.0f, G, 0.0f);
	glVertex2f(0.898f, -0.15f);
	glVertex2f(0.60f, -0.15f);
	glVertex2f(0.594f, -0.2f);
	glVertex2f(0.908f, -0.2f);
	glEnd();
*/
//middle strips
	glLineWidth(5.0f);
	glBegin(GL_LINES);
	glColor3f(R, G, 0.0f);
	glVertex2f(0.65f, 0.025f);
	glVertex2f(0.852f, 0.025f);
	glEnd();

	glBegin(GL_LINES);
	glColor3f(R, White, B);
	glVertex2f(0.64f, 0.0f);
	glVertex2f(0.859f, 0.0f);
	glEnd();

	glBegin(GL_LINES);
	glColor3f(0.0f, G, 0.0f);
	glVertex2f(0.64f, -0.025f);
	glVertex2f(0.865f, -0.025f);
	glEnd();	
}

void FighterPlane_Top(void)
{
	//Body
	glBegin(GL_QUADS);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(-1.0f, 0.3f);
	glVertex2f(-1.0f, -0.3f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//BackEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(-0.3f, 0.0f);
	glVertex2f(-1.2f, 0.2f);
	glVertex2f(-1.2f, -0.2f);
	glEnd();

	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-1.0f, 0.15f);
	glVertex2f(-1.0f, -0.15f);
	glEnd();

	//FrontEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.8f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();


	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//Upper Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	//glColor3f(1.0f, 0.0f, 0.0f);
	glVertex2f(1.5f, 0.32f);
	glVertex2f(-0.6f, 1.5f);
	glVertex2f(-0.6f, 0.22f);
	glEnd();

	//Lower Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(1.5f, -0.32f);
	glVertex2f(-0.6f, -1.5f);
	glVertex2f(-0.6f, -0.22f);
	glEnd();
	/*------------------------------------IAF----------------------------------------*/
	//I
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-0.0f, 0.15f);
	glVertex2f(-0.0f, -0.15f);
	glEnd();

	//A
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.1f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.3f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.15f, 0.0f);
	glVertex2f(0.25f, 0.0f);
	glEnd();

	//F
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.4f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.55f, 0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.0f);
	glVertex2f(0.5f, 0.0f);
	glEnd();
}

void FighterPlane_Middle(void)
{
	//Body
	glBegin(GL_QUADS);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(-1.0f, 0.3f);
	glVertex2f(-1.0f, -0.3f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//BackEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(-0.3f, 0.0f);
	glVertex2f(-1.2f, 0.2f);
	glVertex2f(-1.2f, -0.2f);
	glEnd();

	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-1.0f, 0.15f);
	glVertex2f(-1.0f, -0.15f);
	glEnd();

	//FrontEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.8f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();


	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//Upper Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	//glColor3f(1.0f, 0.0f, 0.0f);
	glVertex2f(1.5f, 0.32f);
	glVertex2f(-0.6f, 1.5f);
	glVertex2f(-0.6f, 0.22f);
	glEnd();

	//Lower Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(1.5f, -0.32f);
	glVertex2f(-0.6f, -1.5f);
	glVertex2f(-0.6f, -0.22f);
	glEnd();
	/*------------------------------------IAF----------------------------------------*/
	//I
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-0.0f, 0.15f);
	glVertex2f(-0.0f, -0.15f);
	glEnd();

	//A
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.1f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.3f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.15f, 0.0f);
	glVertex2f(0.25f, 0.0f);
	glEnd();

	//F
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.4f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.55f, 0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.0f);
	glVertex2f(0.5f, 0.0f);
	glEnd();
}
void FighterPlane_Bottom(void)
{
	//Body
	glBegin(GL_QUADS);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(-1.0f, 0.3f);
	glVertex2f(-1.0f, -0.3f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//BackEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(-0.3f, 0.0f);
	glVertex2f(-1.2f, 0.2f);
	glVertex2f(-1.2f, -0.2f);
	glEnd();

	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-1.0f, 0.15f);
	glVertex2f(-1.0f, -0.15f);
	glEnd();

	//FrontEnd Triangle
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(2.8f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();


	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(2.0f, 0.35f);
	glVertex2f(2.0f, -0.35f);
	glEnd();

	//Upper Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	//glColor3f(1.0f, 0.0f, 0.0f);
	glVertex2f(1.5f, 0.32f);
	glVertex2f(-0.6f, 1.5f);
	glVertex2f(-0.6f, 0.22f);
	glEnd();

	//Lower Blade
	glBegin(GL_TRIANGLES);
	glColor3f(0.7294117f, 0.8862745f, 0.9333333f);
	glVertex2f(1.5f, -0.32f);
	glVertex2f(-0.6f, -1.5f);
	glVertex2f(-0.6f, -0.22f);
	glEnd();
	/*------------------------------------IAF----------------------------------------*/
	//I
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(-0.0f, 0.15f);
	glVertex2f(-0.0f, -0.15f);
	glEnd();

	//A
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.1f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.2f, 0.15f);
	glVertex2f(0.3f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.15f, 0.0f);
	glVertex2f(0.25f, 0.0f);
	glEnd();

	//F
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.4f, -0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.15f);
	glVertex2f(0.55f, 0.15f);
	glEnd();
	glBegin(GL_LINES);
	glVertex2f(0.4f, 0.0f);
	glVertex2f(0.5f, 0.0f);
	glEnd();
}

void dynamic_india(void)
{
	//FUNCTION PROTOTYPE
	void OGL_I1					(void);
	void OGL_N					(void);
	void OGL_D					(GLfloat, GLfloat, GLfloat);
	void OGL_I2					(void);
	void OGL_A 					(void);
	void OGL_A_TriColourStrips 	(GLfloat, GLfloat, GLfloat, GLfloat);
	void FighterPlane_Bottom	(void);
	void FighterPlane_Middle	(void);
	void FighterPlane_Top		(void);

	//code
	glTranslatef(x_I1, 0.0f, 0.0f);
	OGL_I1();
	if(x_I1 <= 0.0f)
	{
		x_I1	=	x_I1 	+	0.00034;
		if(x_I1 > 0.0f)
		{
			bITrue = true;
		}
	}

	//A
	if(bITrue == true)
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		glTranslatef(x_A, 0.0f, 0.0f);
		OGL_A();
		if(x_A > 0.0f)
		{
			x_A = x_A - 0.00034f;
			if(x_A < 0.0f)
			{
				bATrue = true;
			}
		}
	}

	//N
	if(bATrue == true)
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		glTranslatef(0.0f, y_N, 0.0f);
		OGL_N();
		if(y_N > 0.0f)
		{
			y_N = y_N - 0.000354f;
			if(y_N < 0.0f)
			{
				bNTrue = true;
			}
		}
	}

	if(bNTrue == true)
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		glTranslatef(0.0f, y_I2, 0.0f);
		OGL_I2();
		if(y_I2 < 0.0f)
		{
			y_I2 = y_I2 + 0.000354f;
			if(y_I2 > 0.0f)
			{
				bI2True = true;
			}
		}
	}

	if(bI2True == true)
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		OGL_D(r, g, b);
		if((r <= 1.0f) && (g < 0.5f))
		{
			r = r + 0.0002f;
			g = g + 0.0001f;
			if((r > 1.0f) && (g > 0.5f))
			{
				bDTrue = true;
			}
		}
	}

	if(bDTrue)
	{
		if(bClipTopPlane == false)
		{
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(0.0f, 0.0f, -3.0f);
			glTranslatef(top.x, top.y, -20.0f);
			glRotatef(top_plane_rotate_angle, 0.0f, 0.0f, 1.0f);

			FighterPlane_Top();
		}

		if(top_clr.bTopDone == false)
		{
			glLoadIdentity();
			glTranslatef(0.0f, 0.0f, -20.0f);
			glPointSize(5.0f);
			glBegin(GL_POINTS);
			for(GLfloat i = clr.angle_1top; i < clr.angle_2top; i = i + 0.000535f)
			{
				glColor3f(top_clr.r, top_clr.g, 0.0f);
				glVertex2f(top.Radius * cosf(i) - 7.0f, top.Radius * sinf(i) + 10.0f + 0.15);
				glColor3f(top_clr.r, top_clr.White, top_clr.b);
				glVertex2f(top.Radius * cosf(i) - 7.0f, top.Radius * sinf(i) + 10.0f);
				glColor3f(0.0f, top_clr.g, 0.0f);
				glVertex2f(top.Radius * cosf(i) - 7.0f, top.Radius * sinf(i) + 10.0f - 0.15);
			}
			glEnd();
			if(bStartIncrementingTop == true)
			{
				glBegin(GL_POINTS);
				for (GLfloat i = clr.angle_3top; i < clr.angle_4top; i = i + 0.00055f)
				{
					glColor3f(top_clr.r, top_clr.g, 0.0f);
					glVertex2f(top.Radius * cosf(i) + 6.5f, top.Radius * sinf(i) + 10.0f + 0.15);
					glColor3f(top_clr.r, top_clr.White, top_clr.b);
					glVertex2f(top.Radius * cosf(i) + 6.5f, top.Radius * sinf(i) + 10.0f);
					glColor3f(0.0f, top_clr.g, 0.0f);
					glVertex2f(top.Radius * cosf(i) + 6.5f, top.Radius * sinf(i) + 10.0f - 0.15);
				}
				glEnd();
			}
		}
		if (bClipBottomPlane == false)
		{
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(0.0f, 0.0f, -3.0f);
			glTranslatef(bottom.x, bottom.y, -20.0f);
			glRotatef(bottom_plane_rotate_angle, 0.0f, 0.0f, 1.0f);
			FighterPlane_Bottom();

		}
		if (bottom_clr.bBottomDone == false)
		{
			glLoadIdentity();
			glTranslatef(0.0f, 0.0, -20.0f);
			glPointSize(5.0f);

			glBegin(GL_POINTS);
			for (GLfloat i = clr.angle_1bottom; i > clr.angle_2bottom; i = i - 0.000235f)
			{
				glColor3f(bottom_clr.r, bottom_clr.g, 0.0f);
				glVertex2f(bottom.Radius * cosf(i) - 7.0f, bottom.Radius * sinf(i) - 10.0f + 0.15f);
				glColor3f(bottom_clr.r, bottom_clr.White, bottom_clr.b);
				glVertex2f(bottom.Radius * cosf(i) - 7.0f, bottom.Radius * sinf(i) - 10.0f);
				glColor3f(0.0f, bottom_clr.g, 0.0f);
				glVertex2f(bottom.Radius * cosf(i) - 7.0f, bottom.Radius * sinf(i) - 10.0f - 0.15f);
			}
			glEnd();
			if (bStartDecrementingBottom == true)
			{
				glBegin(GL_POINTS);
				for (GLfloat i = clr.angle_3bottom; i > clr.angle_4bottom; i = i - 0.00025f)
				{
					glColor3f(bottom_clr.r, bottom_clr.g, 0.0f);
					glVertex2f(bottom.Radius * cosf(i) + 6.5f, bottom.Radius * sinf(i) - 10.0f + 0.15f);
					glColor3f(bottom_clr.r, bottom_clr.White, bottom_clr.b);
					glVertex2f(bottom.Radius * cosf(i) + 6.5f, bottom.Radius * sinf(i) - 10.0f);
					glColor3f(0.0f, bottom_clr.g, 0.0f);
					glVertex2f(bottom.Radius * cosf(i) + 6.5f, bottom.Radius * sinf(i) - 10.0f - 0.15f);
				}
				glEnd();
			}
		}
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		glTranslatef(x_plane, 0.0f, -20.0f);

		FighterPlane_Middle();
		if (middle_clr.bMiddleDone == false)
		{
			glLineWidth(5.0f);
			glBegin(GL_LINES);
			glColor3f(middle_clr.r, middle_clr.g, 0.0f);
			glVertex2f(-1.20f, 0.15f);
			glVertex2f(x_CoordinateOfStrips_Orange, 0.15f);
			glColor3f(middle_clr.r, middle_clr.White, middle_clr.b);
			glVertex2f(-1.20f, 0.0f);
			glVertex2f(x_CoordinateOfStrips_Orange, 0.0f);
			glColor3f(0.0f, middle_clr.g, 0.0f);
			glVertex2f(-1.20f, -0.15f);
			glVertex2f(x_CoordinateOfStrips_Orange, -0.15f);
			glEnd();
		}
		if (x_plane <= 22.0f)
		{
			x_plane = x_plane + 0.00445f;
			x_CoordinateOfStrips_Orange = x_CoordinateOfStrips_Orange - 0.005f;
			//top
			if ((top.angle <= M_PI + M_PI_2) )
			{
				top.x = top.Radius * cosf(top.angle) - 8.0f;
				top.y = top.Radius * sinf(top.angle) + 10.0f;
				top.angle = top.angle + 0.0005;
				if (clr.angle_2top <= 4.71238)
				{
					clr.angle_2top = clr.angle_2top + 0.000475;
				}
				if (top.angle > M_PI + M_PI_2)
				{
					bClipTopPlane = true;
				}
				
			}
			//bottom
			if (bottom.angle >= M_PI_2)
			{
				bottom.x = bottom.Radius * cosf(bottom.angle) - 8.0f;
				bottom.y = bottom.Radius * sinf(bottom.angle) - 10.0f;
				bottom.angle = bottom.angle - 0.0005;
				if (clr.angle_2bottom >= M_PI_2)
				{
					clr.angle_2bottom = clr.angle_2bottom - 0.000475;
				}
				if (bottom.angle <= M_PI_2)
				{
					bClipBottomPlane = true;
				}
			}
			//angle
			if ((top_plane_rotate_angle <= 0.0f) && (bottom_plane_rotate_angle >= 0.0f))
			{
				top_plane_rotate_angle = top_plane_rotate_angle + 0.021;
				bottom_plane_rotate_angle = bottom_plane_rotate_angle - 0.021;
			}

			if ((x_plane > 8.0f))
			{
				bClipTopPlane = false;
				bClipBottomPlane = false;

				bStartIncrementingTop = true;
				bStartDecrementingBottom = true;
				if (top.angle <= 2 * M_PI)
				{
					top.x = top.Radius * cosf(top.angle) + 8.0f;
					top.y = top.Radius * sinf(top.angle) + 10.0f;
					top.angle = top.angle + 0.0005;
					if (clr.angle_4top <= 6.28318)
					{
						clr.angle_4top = clr.angle_4top + 0.00038;
					}
					if (top.angle > 2 * M_PI)
					{
						bClipTopPlane = true;
					}
				}
				if (bottom.angle >= 0.0f)
				{
					bottom.x = bottom.Radius * cosf(bottom.angle) + 8.0f;
					bottom.y = bottom.Radius * sinf(bottom.angle) - 10.0f;
					bottom.angle = bottom.angle - 0.0005;
					if (clr.angle_4bottom >= 0.0f)
					{
						clr.angle_4bottom = clr.angle_4bottom - 0.00038;
					}
					if (bottom.angle <= 0.0f)
					{
						bClipBottomPlane = true;
					}
				}
				top_plane_rotate_angle = top_plane_rotate_angle + 0.021;
				bottom_plane_rotate_angle = bottom_plane_rotate_angle - 0.021;
				if (x_plane > 22.0f)
				{
					bPlaneTrue = true;
				}
			}
		}	
	}

	if (bPlaneTrue == true)
	{
		//top
		if (top_clr.bTopDone == false)
		{
			top_clr.r = top_clr.r - 0.005f;
			top_clr.g = top_clr.g - 0.005f;
			top_clr.b = top_clr.b - 0.005f;
			top_clr.White = top_clr.White - 0.005f;
		}
		//bottom
		if (bottom_clr.bBottomDone == false)
		{
			bottom_clr.r = bottom_clr.r - 0.005f;
			bottom_clr.g = bottom_clr.g - 0.005f;
			bottom_clr.b = bottom_clr.b - 0.005f;
			bottom_clr.White = bottom_clr.White - 0.005f;
		}
		//middle
		if (middle_clr.bMiddleDone == false)
		{
			
			middle_clr.r = middle_clr.r - 0.005f;
			middle_clr.g = middle_clr.g - 0.005f;
			middle_clr.b = middle_clr.b - 0.005f;
			middle_clr.White = middle_clr.White - 0.005f;
		}
		
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
		OGL_A_TriColourStrips(r_A, g_A, b_A, White_A);
		if ((r_A <= 1.0f) && (g_A <= 0.5f) && (b_A <= 1.0f) && (White_A <= 1.0f))
		{
			
			r_A = r_A + 0.002f;
			g_A = g_A + 0.001f;
			b_A = b_A + 0.002f;
			White_A = White_A + 0.002f;
			if ((r > 1.0f) && (g > 0.5f) && (b > 1.0f) && (White_A > 1.0f))
			{
				bColourDone = true;
			}
			
		}
		if ((top_clr.r < 0.0f) && (top_clr.g < 0.0f) && (top_clr.b < 0.0f) && (top_clr.White < 0.0f))
		{
			top_clr.bTopDone = true;
		}
		if ((bottom_clr.r < 0.0f) && (bottom_clr.g < 0.0f) && (bottom_clr.b < 0.0f) && (bottom_clr.White < 0.0f))
		{
			bottom_clr.bBottomDone = true;
		}
		if ((middle_clr.r < 0.0f) && (middle_clr.g < 0.0f) && (middle_clr.b < 0.0f) && (middle_clr.White < 0.0f))
		{
			middle_clr.bMiddleDone = true;
		}
		
	}
}
