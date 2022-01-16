//standard Headers
#include <iostream>															//c++ Header
#include <stdio.h>															//for printf()
#include <stdlib.h>															//for exit()
#include <memory.h>															//for memset

//Xlib Headers
//Path :- /usr/include/X11
#include <X11/Xlib.h>														//analogous to windows.h
#include <X11/Xutil.h>														//for XVisualInfo Structure
#include <X11/XKBlib.h>														//for Keyboard utilization Header
#include <X11/keysym.h>														//for Key Symbols (For e.g. VK_NUMPAD1 In SDK)

//namespaces
using namespace std;

//global variable declaration
bool bFullScreen = false;
Display *gpDisplay = NULL;
//XVisualInfo *gpXVisualInfo = NULL;											//analogous to PIXELFORMATDISCRIPTOR
XVisualInfo gXVisualInfo;													//For XMatchVisualInfo Version_2
Colormap gColormap;															//this is struct and this is array of color cell
Window gWindow;																//struct variable
int giWindowWidth = 800;													//WIN_WIDTH
int giWindowHeight = 600;													//WIN_HEIGHT

/*
*All Window Applications Are Commandline Argument Base Applications, Those Applications Are Recommended For Headless Communication
*/

//entry-point function
int main(void)
{
	//function prototype
	void CreateWindnow(void);
	void ToggleFullScreen(void);
	void uninitialise(void);

	//variable declaration
	int winWidth = giWindowWidth;
	int winHeight = giWindowHeight;

	//code
	CreateWindnow();

	//Message Loop
	XEvent event;
	KeySym keysym;

	while(1)
	{
		XNextEvent(gpDisplay, &event);										//Analogous to GetMessage And SDK's &msg Xlib's &event
		switch(event.type)													//all WM_ Messages in Windows SDK's switch (iMsg) Xlib's switch(event.type)
		{
			//WM_CREATE
			case MapNotify:
				break;

			//WM_KEYDOWN
			case KeyPress:
			keysym = XkbKeycodeToKeysym(gpDisplay, event.xkey.keycode, 0, 0);
			switch(keysym)
			{
				//VK_ESCAPE
				case XK_Escape:
					uninitialise();
					exit(0);
					break;
				//F key
				case XK_F:
				case XK_f:
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
				default:
					break;
			}
				break;
			//WM_LBUTTONDOWN, WM_RBUTTONDOWN
			case ButtonPress:
				switch(event.xbutton.button)
				{
					//LEFT_BUTTON
					case 1:
						break;
					//MIDDLE_BUTTON
					case 2:
						break;
					//RIGHT_BUTTON
					case 3:
						break;
					default:
						break;
				}
				break;
			//WM_MOUSEMOVE
			case MotionNotify:
				break;
			//WM_SIZE
			case ConfigureNotify:
				winWidth = event.xconfigure.width;
				winHeight = event.xconfigure.height;
				break;
			//WM_PAINT
			case Expose:
				break;
			//WM_DESTROY
			case DestroyNotify:
				//it internally work as a WM_CLOSE NOT LIKE WM_DESTROY
				break;
			case 33:
				uninitialise();
				exit(0);
				break;
			default:
				break;
		}												
	}
	//the code will come when there will be abort condition
	uninitialise();
	return(0);
}

/*Create Window Steps
*1]		Open The Connection With Xserver And Get The Display
*2]		Get Default Screen.
*3]		Made Default Depth.
*4]		Get Matching Visual Info.
*5]		Fill Window Attributes.
*6]		Define The Window Styles.
*7]		Create An Actual Window Based On Above All Things.
*8]		Name That Window
*9]		Handle The Functionality To Close Window On Close Button and Close Menu.
*10]	Map The Window With (10, 10, 200, 200) Which XServer's Window.
*/

void CreateWindnow(void)
{
	//function prototype
	void uninitialise(void);

	//variable decalaration
	XSetWindowAttributes winAttribs;		//Analogous To WNDCLASSEX wndclass.
	int defaultScreen;
	int defaultDepth;
	int styleMask;

	//code
	gpDisplay = XOpenDisplay(NULL);																		//Step-1
	/* NULL = Give Me Default Display Structure */
	if(gpDisplay == NULL)
	{
		printf("ERROR : Unable To Open X Display.\nExiting Now...\n");
		uninitialise();
		exit(1);
	}

	defaultScreen = XDefaultScreen(gpDisplay);															//Step-2

	defaultDepth = XDefaultDepth(gpDisplay, defaultScreen);												//Step-3
/*
	//To Use The Imagination's Code Just UnComment It And Change gXVisualInfo To gpXVisualInfo
	//and change gXVisualInfo.something to gpXVisualInfo->something and Uncomment The unitialise code.
	//Imagination's Code 
	gpXVisualInfo = (XVisualInfo *)malloc(sizeof(XVisualInfo));											//Step-4(v_1)
	if(gpXVisualInfo == NULL)
	{
		printf("ERROR : Unable To Allocate Memory For Visual Info.\nExiting Now.....\n");
		uninitialise();
		exit(1);
	}

	XMatchVisualInfo(gpDisplay, defaultScreen, defaultDepth, TrueColor, gpXVisualInfo);
	if(gpXVisualInfo == NULL)
	{
		printf("ERROR : Unable To Get A Visual.\nExiting Now.....\n");
		uninitialise();
		exit(1);
	}
*/

	//This Code Is Good Practice but showing core dumped												//Step-4(v_2)
	Status status = XMatchVisualInfo(gpDisplay, defaultScreen, defaultDepth, TrueColor, &gXVisualInfo);
	if(status == 0)
	{
		printf("ERROR : XMatchVisualInfo Is Failed\nExiting Now...\n");
		uninitialise();
		exit(1);
	}
																							//Step-5
	winAttribs.border_pixel = 0;											//set border color default
	winAttribs.border_pixmap = 0;											//Do We Want To Set An Image To Border
	winAttribs.background_pixmap = 0;										//Do We Want To Set Image To background
	winAttribs.background_pixel = BlackPixel(gpDisplay, defaultScreen);		//Analogous to HBRUSH
	winAttribs.colormap = XCreateColormap(gpDisplay,						//similar to wndExtra
						  RootWindow(gpDisplay, gXVisualInfo.screen),
						  gXVisualInfo.visual,
						  AllocNone);
	gColormap = winAttribs.colormap;
	//Following Attribute Is To Tell In How Many Messages We Are Interested.
	winAttribs.event_mask = ExposureMask | VisibilityChangeMask | ButtonPressMask | KeyPressMask | PointerMotionMask | StructureNotifyMask;

																							//Step-6
	styleMask = CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;

	gWindow = XCreateWindow(gpDisplay,																	//Step-7
		      RootWindow(gpDisplay, gXVisualInfo.screen),
		      0,
		      0,
		      giWindowWidth,
		      giWindowHeight,
		      0,						//for border width
		      gXVisualInfo.depth,
		      InputOutput,				//Our Window Will Do Both i/p and o/p
		      gXVisualInfo.visual,
		      styleMask,
		      &winAttribs);
	if(!gWindow)
	{
		printf("ERROR : Failed To Create Main Window.\nExiting Now...\n");
		uninitialise();
		exit(1);
	}
	
	XStoreName(gpDisplay, gWindow, "First Window");														//Step-8

	Atom windowManagerDelete = XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);						//Step-9
	/*
	*XInternAtom	-> 	Internal Atoms Of XServer Which Immutable.
	*True 			->	If There Is Already Available, Still Do That Again.
	*false 			->	If There Is Already Available, The Don't Do That Again.
	*/

	XSetWMProtocols(gpDisplay, gWindow, &windowManagerDelete, 1);
	/*Parameters:- 
	*gpDisplay
	*Which Window
	*Which Protocol
	*We Don't Have An Array We Have One Variable So '1'.
	*/
	XMapWindow(gpDisplay, gWindow);																		//Step-10
}

/*ToggleFullScreen Steps:-
*1]		Take A Current State Of A Window
*2]		Make An Event, Check The State(Whether It Is Full Screen Or not) and The Toggle.
*3]		Fire The Event
*/

void ToggleFullScreen(void)
{
	//variable declaration
	Atom wm_state;
	Atom fullscreen;
	XEvent xev = { 0 };

	//code
	wm_state = XInternAtom(gpDisplay, "_NET_WM_STATE", False);							//Step-1

	memset(&xev, 0, sizeof(xev));

																						//Step-2
	xev.type = ClientMessage;
	xev.xclient.window = gWindow;		//For Which Window The Message Is
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = bFullScreen ? 0 : 1;

	fullscreen = XInternAtom(gpDisplay, "_NET_WM_STATE_FULLSCREEN", False);
	xev.xclient.data.l[1] = fullscreen;

																						//Step-3
	XSendEvent(gpDisplay,
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
/*
	if(gpXVisualInfo)
	{
		free(gpXVisualInfo);
		gpXVisualInfo = NULL;
	}
*/
	
	if(gpDisplay)
	{
		free(gpDisplay);
		gpDisplay = NULL;
	}
}
