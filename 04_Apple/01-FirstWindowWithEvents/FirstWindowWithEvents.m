//headers
#import <Foundation/Foundation.h>       //for NSAutoreleasePool
#import <Cocoa/Cocoa.h>                 //for NSApp

//interface declaration
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface MyView : NSView
@end

//entry-point function
int main(int argc, const char *argv[])
{
    //code
    NSAutoreleasePool *pPool = [[NSAutoreleasePool alloc]init];
    
    NSApp = [NSApplication sharedAppliation];

    [NSApp setDelegate:[[AppDelegate alloc]init]];

    [NSApp run];

    [pPool release];

    return(0);
}

//interface implementation
@implementation AppDelegate
{
@private 
    NSWindow *window;
    MyView *view;
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    //code
    //window
    NSRect win_rect;
    win_rect = NSMakeRect(0.0, 0.0, 800.0, 600.0);

    //create simple window
    window = [[NSWindow alloc]  initWithContentRect :win_rect
                                styleMask           :NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                                backing             :NSBackingStoreBuffered
                                defer               :NO];
    [window setTitle:@"macOS Window"];
    [window center];

    view = [[MyView alloc]initWithFrame:win_rect];

    [window setContentView:view];
    [window setDelegate:self];
    [window makeKeyAndOrderFront:self];
}

- (void)applicationWillTerminate:(NSNotification *)notification
{
    //code
}

- (void) windowWillClose:(NSNotification *)notification
{
    //code
    [NSApp terminate:self];
}

- (void)dealloc
{
    //code
    [view release];

    [window release];

    [super dealloc];
}
@end

@implementation MyView
{
@private 
    NSString *centralText;
}

- (id)initWithFrame:(NSRect)frame;
{
    //code
    self = [super initWithFrame:frame];

    if(self)
    {
        [[self window]setContentView:self];

        centralText=@"Hello World !!!";
    }
    return(self);
}

- (void)drawRect:(NSRect)dirtyRect
{
    //code
    //black background 
    NSColor *fillColor=[NSColor blackColor];
    [fillColor set];
    NSRectFill(dirtyRect);

    //dictionary with key value coding
    NSDictionary *dictionaryForTextAttributes=[NSDictionary dictionaryWithObjectsAndKeys:
    [NSFont fontWithName:@"Helvetica" size:32],
    [NSColor greenColor], NSForegroundColorAttributeName,
    nil];

    NSSize textSize=[centralText sizeWithAttributes:dictionaryForTextAttributes];

    NSPoint point;
    point.x=(dirtyRect.size.width/2)-(textSize.width/2);
    point.y=(dirtyRect.size.height/2)-(textSize.height/2) + 12;

    [centralText drawAtPoint:point withAttributes:dictionaryForTextAttributes];
}

- (BOOL)acceptFirstResponder
{
    //code
    [[self window]makeFirstResponder:self];
    return(YES);
}

- (void)keyDown:(NSEvent *)theEvent
{
    //code
    int key = (int)[[theEvent characters]characterAtIndex:0];
    switch(key)
    {
        case 27:
            [self release];
            [NSApp terminate:self];
            break;
        case 'F':
        case 'f':
            centralText=@"'F' or 'f' Key Is Pressed";
            [[self window]toggleFullScreen:self];
            break;
        default:
            break;
    }
}

- (void)mouseDown:(NSEvent *)theEvent
{
    //code
    centralText=@"left mouse button is pressed.";
    [self setNeedsDisplay:YES]; //repainting
}

- (void)mouseDragged:(NSEvent *)theEvent
{
    //code
}

- (void)rightMouseDown:(NSEvent *)theEvent
{
    //code
    centralText=@"right mouse button is pressed.";
    [self setNeedsDisplay:YES]; //repainting
}

- (void) dealloc
{
    //code
    [super dealloc];
}

@end
