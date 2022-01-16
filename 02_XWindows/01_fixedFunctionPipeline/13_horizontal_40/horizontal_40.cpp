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
	int 					defaultScreen;
	int 					defaultDepth;
	int 					styleMask;
	//Status 					status;

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

	XStoreName(gpDisplay, gWindow, "Point In The Middle Of The Screen");
	Atom windowManagerDelete	=	XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);
	XSetWMProtocols (gpDisplay, gWindow, &windowManagerDelete, 1);
	
	XMapWindow(gpDisplay, gWindow);																																																		
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

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//warm-up resize call
	oglResize(giWindowWidth, giWindowHeight);
}	

void oglResize(int iWidth, int iHeight)
{
	//code
/*	
	if(iHeight == 0)
	{
		iHeight = 1;
	}
*/
	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);
/*
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(	45.0f,
					((GLfloat)iWidth / (GLfloat)iHeight),
					0.1f, 
					100.0f);
*/
}

void oglDisplay(void)
{
	//function prototype
	void horizontal_40(void);

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glBegin(GL_LINES);

	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex2f(1.0f, 0.0f);
	glVertex2f(-1.0f, 0.0f);

	glEnd();

	horizontal_40();
	
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

void horizontal_40(void)
{
	glLineWidth(1.0f);
	glBegin(GL_LINES);

	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex2f(1.0f, -0.95f);
	glVertex2f(-1.0f, -0.95f);

	glVertex2f(1.0f, -0.90f);
	glVertex2f(-1.0f, -0.90f);

	glVertex2f(1.0f, -0.85f);
	glVertex2f(-1.0f, -0.85f);

	glVertex2f(1.0f, -0.80f);
	glVertex2f(-1.0f, -0.80f);

	glVertex2f(1.0f, -0.75f);
	glVertex2f(-1.0f, -0.75f);

	glVertex2f(1.0f, -0.70f);
	glVertex2f(-1.0f, -0.70f);

	glVertex2f(1.0f, -0.65f);
	glVertex2f(-1.0f, -0.65f);

	glVertex2f(1.0f, -0.60f);
	glVertex2f(-1.0f, -0.60f);

	glVertex2f(1.0f, -0.55f);
	glVertex2f(-1.0f, -0.55f);

	glVertex2f(1.0f, -0.50f);
	glVertex2f(-1.0f, -0.50f);

	glVertex2f(1.0f, -0.45f);
	glVertex2f(-1.0f, -0.45f);

	glVertex2f(1.0f, -0.40f);
	glVertex2f(-1.0f, -0.40f);

	glVertex2f(1.0f, -0.35f);
	glVertex2f(-1.0f, -0.35f);

	glVertex2f(1.0f, -0.30f);
	glVertex2f(-1.0f, -0.30f);

	glVertex2f(1.0f, -0.25f);
	glVertex2f(-1.0f, -0.25f);

	glVertex2f(1.0f, -0.20f);
	glVertex2f(-1.0f, -0.20f);

	glVertex2f(1.0f, -0.15f);
	glVertex2f(-1.0f, -0.15f);

	glVertex2f(1.0f, -0.10f);
	glVertex2f(-1.0f, -0.10f);

	glVertex2f(1.0f, -0.05f);
	glVertex2f(-1.0f, -0.05f);

	

	//Right Half
	
	glVertex2f(1.0f, 0.95f);
	glVertex2f(-1.0f, 0.95f);

	glVertex2f(1.0f, 0.90f);
	glVertex2f(-1.0f, 0.90f);

	glVertex2f(1.0f, 0.85f);
	glVertex2f(-1.0f, 0.85f);

	glVertex2f(1.0f, 0.80f);
	glVertex2f(-1.0f, 0.80f);

	glVertex2f(1.0f, 0.75f);
	glVertex2f(-1.0f, 0.75f);

	glVertex2f(1.0f, 0.70f);
	glVertex2f(-1.0f, 0.70f);

	glVertex2f(1.0f, 0.65f);
	glVertex2f(-1.0f, 0.65f);

	glVertex2f(1.0f, 0.60f);
	glVertex2f(-1.0f, 0.60f);

	glVertex2f(1.0f, 0.55f);
	glVertex2f(-1.0f, 0.55f);

	glVertex2f(1.0f, 0.50f);
	glVertex2f(-1.0f, 0.50f);

	glVertex2f(1.0f, 0.45f);
	glVertex2f(-1.0f, 0.45f);

	glVertex2f(1.0f, 0.40f);
	glVertex2f(-1.0f, 0.40f);

	glVertex2f(1.0f, 0.35f);
	glVertex2f(-1.0f, 0.35f);

	glVertex2f(1.0f, 0.30f);
	glVertex2f(-1.0f, 0.30f);

	glVertex2f(1.0f, 0.25f);
	glVertex2f(-1.0f, 0.25f);

	glVertex2f(1.0f, 0.20f);
	glVertex2f(-1.0f, 0.20f);

	glVertex2f(1.0f, 0.15f);
	glVertex2f(-1.0f, 0.15f);

	glVertex2f(1.0f, 0.10f);
	glVertex2f(-1.0f, 0.10f);

	glVertex2f(1.0f, 0.05f);
	glVertex2f(-1.0f, 0.05f);

	glEnd();
}
