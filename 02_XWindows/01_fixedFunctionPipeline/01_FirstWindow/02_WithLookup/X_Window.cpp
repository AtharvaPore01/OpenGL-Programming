//Standard Headers Files
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

//X11 Related Header File
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>

//namespaces
using namespace std;

//global variable declaration
bool 			bFullScreen		=	false;
Display 		*gpDisplay		= 	NULL;
XVisualInfo 	gXVisualInfo;
Colormap 		gColormap;
Window 			gWindow;
int 			giWindowWidth 	= 	800;
int 			giWindowHeight 	= 	600;

//entry-point function
int main(void)
{
	//function prototype
	void CreateWindow(void);
	void ToggleFullScreen(void);
	void uninitialise(void);

	//variable decalaration
	int 	winWidth 	=	giWindowWidth;
	int 	winHeight 	= 	giWindowHeight;
	char 	keys[26];

	//code
	CreateWindow();

	//Message Loop
	XEvent event;
	KeySym keysym;

	while(1)
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
						uninitialise();
						exit(0);
					break;
				}
				/*
				*New Change, This Change Is Done To Bring The Same Behavior As WM_CHAR In SDK.
				*Because In Regular Code The XK_F Doesn't Work. 
				*/
				XLookupString(&event.xkey, keys, sizeof(keys), NULL, NULL);
				/*
				*	Syntax:-
				*	int XLookupString(	event_struct(Where Do I See), 
										buffer_return(Where Do I Copy), 
										bytes_buffer(Size Of A Buffer Where I Have To Copy), 
										keysym_return(Whether KeySym Will get Used or Not), 
										status_in_out(struct XComposeState's Parameter));
				*/
				switch(keys[0])
				{
					case 'F':
					case 'f':
						if(bFullScreen == false)
						{
							ToggleFullScreen();
							bFullScreen = true;
						}
						else
						{
							ToggleFullScreen();
							bFullScreen = false;
						}
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
			break;

			case Expose:
				//code
			break;

			case DestroyNotify:
				//code
			break;

			case 33:
				uninitialise();
				exit(0);
			break;

			default:
			break;
		}
	}

	uninitialise();
	return(0);
}

void CreateWindow(void)
{
	//function prototype
	void uninitialise(void);

	//variable declaration
	XSetWindowAttributes 	winAttribs;
	int 					defaultScreen;
	int 					defaultDepth;
	int 					styleMask;
	//Status 					status;

	//code
	gpDisplay = XOpenDisplay(NULL);
	if(gpDisplay == NULL)
	{
		printf("ERROR : Unable To Open X Display.\n Exitting Now....\n");
		uninitialise();
		exit(1);
	}

	defaultScreen = XDefaultScreen(gpDisplay);

	defaultDepth = DefaultDepth(gpDisplay, defaultScreen);

	Status status = XMatchVisualInfo(gpDisplay, defaultScreen, defaultDepth, TrueColor, &gXVisualInfo);
	if(status == 0)
	{
		printf("ERROR : XMatchVisualInfo Is Failed\nExiting Now...\n");
		uninitialise();
		exit(1);
	}

	winAttribs.border_pixel			=	0;
	winAttribs.border_pixmap		=	0;
	winAttribs.background_pixmap 	=	0;
	winAttribs.background_pixel 	= 	BlackPixel(gpDisplay, defaultScreen);
	winAttribs.colormap 			=	XCreateColormap(	gpDisplay,
															RootWindow(gpDisplay, gXVisualInfo.screen),
															gXVisualInfo.visual,
															AllocNone);
	gColormap						=	winAttribs.colormap;

	winAttribs.event_mask			= 	ExposureMask | VisibilityChangeMask | ButtonPressMask | KeyPressMask | PointerMotionMask | StructureNotifyMask;

	styleMask						=	CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;

	gWindow 						=	XCreateWindow(	gpDisplay,
														RootWindow(gpDisplay, gXVisualInfo.screen),
														0,
														0,
														giWindowWidth,
														giWindowHeight,
														0,
														gXVisualInfo.depth,
														InputOutput,
														gXVisualInfo.visual,
														styleMask,
														&winAttribs);
	if(!gWindow)
	{
		printf("ERROR : Failed To Create Main Window.\n Exitting Now...\n");
		uninitialise();
		exit(1);
	}

	XStoreName(gpDisplay, gWindow, "First Window With Lookup");

	Atom windowManagerDelete	= 	XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay, gWindow, &windowManagerDelete, 1);

	XMapWindow(gpDisplay, gWindow);
}

void ToggleFullScreen(void)
{
	//variable declaration
	Atom 	wm_state;
	Atom 	fullscreen;
	XEvent 	xev 	=	{0};

	//code
	wm_state = XInternAtom(gpDisplay, "_NET_WM_STATE", False);
	memset(&xev, 0, sizeof(xev));

	xev.type 					=	ClientMessage;
	xev.xclient.window 			=	gWindow;
	xev.xclient.message_type 	=	wm_state;
	xev.xclient.format			= 	32;
	xev.xclient.data.l[0]		=	bFullScreen ? 0 : 1;

	fullscreen 					=	XInternAtom(gpDisplay, "_NET_WM_STATE_FULLSCREEN", False);
	xev.xclient.data.l[1]		=	fullscreen;

	XSendEvent(	gpDisplay,
				RootWindow(gpDisplay, gXVisualInfo.screen),
				False,
				StructureNotifyMask,
				&xev);	

}

void uninitialise(void)
{
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
	
	if(gpDisplay)
	{
		free(gpDisplay);
		gpDisplay = NULL;
	}
}
