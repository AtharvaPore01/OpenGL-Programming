//Standard Headers Files
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

//X11 Related Header Files
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>

//namespaces
using namespace std;

//global variable declaration
bool 			bFullScreen 			= 	false;
Display 		*gpDisplay				=	NULL;
XVisualInfo 	gXVisualInfo;
Colormap		gColormap;
Window			gWindow;
int 			giWindowWidth			=	800;
int 			giWindowHeight			=	600;

//entry-point function
int main(void)
{
	//fucntion prototype
	void CreateWindow(void);
	void ToggleFullscreen(void);
	void uninitialise(void);

	//variable declaration
	int 					winWidth 		=	giWindowWidth;
	int 					winHeight 		= 	giWindowHeight;
	char 					keys[26];
/*
	// Contents Of XFontStruct
	typedef struct 
	{
		XExtData *ext_data;			//hook for extension to hang data 
		Font fid;					// Font id for this font 
		unsigned direction;			// hint about the direction font is painted 
		unsigned min_char_or_byte2;	// first character 
		unsigned max_char_or_byte2;	// last character 
		unsigned min_byte1;			// first row that exists 
		unsigned max_byte1;			// last row that exists 
		Bool all_chars_exist;		// flag if all characters have nonzero size 
		unsigned default_char;		// char to print for undefined character 
		int n_properties;			// how many properties there are 
		XFontProp *properties;		// pointer to array of additional properties 
		XCharStruct min_bounds;		// minimum bounds over all existing char 
		XCharStruct max_bounds;		// maximum bounds over all existing char 
		XCharStruct *per_char;		// first_char to last_char information 
		int ascent;					// logical extent above baseline for spacing 
		int descent;				// logical decent below baseline for spacing 
	} XFontStruct;


*/

	static XFontStruct 		*pXFontStruct	=	NULL;
	static GC 				gc;
	XGCValues				gcValues;		/* 	This Is Structure Which Will Get Filled In XCreateGC's Last Parameter */
	XColor 					text_color;		/* 	This Is Structure Variable Which Give Nearby Value Of Given Color Or Exact Value Of 
											*	Give Color.
											*/
	char 					str[]			=	"Hello World!!!";
	int						strLen;
	int						strWidth;
	int						fontHeight;

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
				pXFontStruct = XLoadQueryFont(gpDisplay, "fixed");

				if(pXFontStruct == NULL)
				{
					printf("Query For Fixed Font(Default Pitched Font) Is Failed.\nExitting Now...\n");
					uninitialise();
					exit(1);
				}

			break;

			case KeyPress:
				keysym = XkbKeycodeToKeysym(gpDisplay, event.xkey.keycode, 0, 0);
				switch(keysym)
				{
					case XK_Escape:
						XFreeGC		(gpDisplay, gc);
						XUnloadFont	(gpDisplay, pXFontStruct->fid);
						uninitialise();
						exit(0);
					break;
				}
				XLookupString(&event.xkey, keys, sizeof(keys), NULL, NULL);
				switch(keys[0])
				{
					case 'F':
					case 'f':
						if(bFullScreen == false)
						{
							ToggleFullscreen();
							bFullScreen = true;
						}
						else
						{
							ToggleFullscreen();
							bFullScreen = false;
						}
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
			winHeight 	= 	event.xconfigure.height;
			break;

			case Expose:
				gc 			= 	XCreateGC	(gpDisplay, gWindow, 0, &gcValues);
				
				XSetFont					(gpDisplay, gc, pXFontStruct->fid);
				
				XAllocNamedColor			(gpDisplay, gColormap, "green", &text_color, &text_color);
				
				XSetForeground				(gpDisplay, gc, text_color.pixel);

				strLen		=				strlen(str);

				strWidth	=				XTextWidth(pXFontStruct, str, strLen);

				fontHeight	=				pXFontStruct->ascent + pXFontStruct->descent;

				XDrawString					(	gpDisplay,
												gWindow,
												gc,
												(winWidth / 2) - (strWidth / 2),
												(winHeight / 2) - (fontHeight / 2),
												str,
												strLen);
			break;
			
			case DestroyNotify:
				break;

			case 33:
				XFreeGC		(gpDisplay, gc);
				XUnloadFont	(gpDisplay, pXFontStruct->fid);
				uninitialise();
				exit(0);
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
	XSetWindowAttributes	winAttribs;
	int 					defaultScreen;
	int 					defaultDepth;
	int 					styleMask;

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
	winAttribs.border_pixmap 		=	0;
	winAttribs.background_pixmap 	=	0;
	winAttribs.background_pixel 	=	BlackPixel(gpDisplay, defaultScreen);
	winAttribs.colormap 			= 	XCreateColormap(	gpDisplay,
															RootWindow(gpDisplay, gXVisualInfo.screen),
															gXVisualInfo.visual,
															AllocNone);
	gColormap 						=	winAttribs.colormap;
	
	winAttribs.event_mask			=	ExposureMask | VisibilityChangeMask | ButtonPressMask | KeyPressMask | PointerMotionMask | StructureNotifyMask;
	
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
	
	XStoreName(gpDisplay, gWindow, "Hello World");

	Atom windowManagerDelete	=	XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);

	XSetWMProtocols(gpDisplay, gWindow, &windowManagerDelete, 1);

	XMapWindow(gpDisplay, gWindow);																											
}

void ToggleFullscreen(void)
{
	//variable declaraion
	Atom 	wm_state;
	Atom 	fullscreen;
	XEvent 	xev 	=	{0};

	//code
	wm_state = XInternAtom(gpDisplay, "_NET_WM_STATE", False);
	memset(&xev, 0, sizeof(xev));

	xev.type 					=	ClientMessage;
	xev.xclient.window 			=	gWindow;
	xev.xclient.message_type	=	wm_state;
	xev.xclient.format			=	32;
	xev.xclient.data.l[0]		=	bFullScreen ? 0 : 1;

	fullscreen 					=	XInternAtom(gpDisplay, "_NET_WM_STATE_FULLSCREEN", False);
	xev.xclient.data.l[1]		= 	fullscreen;

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
