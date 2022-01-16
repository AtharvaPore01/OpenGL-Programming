
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#import <QuartzCore/CVDisplayLink.h>    //core video display link

//opengl related header
#import <OpenGL/gl3.h>
#import <OpenGL/gl3ext.h>
#import "vmath.h"
#import <math.h>

enum
{
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_NORMAL,
    AMC_ATTRIBUTE_TEXCOODR_0
};
//global function declaration
CVReturn MyDisplayLinkCallback(CVDisplayLinkRef,const CVTimeStamp *, const CVTimeStamp *, CVOptionFlags, CVOptionFlags *, void *);

//global variable
FILE *gpFile = NULL;

//interface declaration
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface GLView : NSOpenGLView
@end

//entry point function
int main(int argc, char *argv[])
{
    //code
    NSAutoreleasePool *pPool_ap = [[NSAutoreleasePool alloc]init];
    
    NSApp=[NSApplication sharedApplication];
    
    [NSApp setDelegate:[[AppDelegate alloc]init]];
    
    [NSApp run];
    
    [pPool_ap release];
    
    return(0);
}

//interface implementation
@implementation AppDelegate
{
@private
    NSWindow *window;
    GLView *glView;
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    //code
    
    //log file
    NSBundle *mainBundle=[NSBundle mainBundle];
    NSString *appDirName=[mainBundle bundlePath];
    NSString *parentDirPath=[appDirName stringByDeletingLastPathComponent];
    NSString *logFileNameWithPath=[NSString stringWithFormat:@"%@/Log.txt", parentDirPath];
    const char *pszLogFileNameWithPath=[logFileNameWithPath cStringUsingEncoding:NSASCIIStringEncoding];
    
    gpFile=fopen(pszLogFileNameWithPath,"w");
    if(gpFile==NULL)
    {
        printf("Can Not Create A Log File.\nExitting...\n");
        [self release];
        [NSApp terminate:self];
    }
    fprintf(gpFile, "Log File Created Successfully.\n");
    
    //window
    NSRect win_rect;
    win_rect=NSMakeRect(0.0,0.0,800.0,600.0);
    
    //create simple window
    window=[[NSWindow alloc] initWithContentRect:win_rect
                                       styleMask:NSWindowStyleMaskTitled |
            NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                                         backing:NSBackingStoreBuffered
                                           defer:NO];
    [window setTitle:@"macOS Window:Graph Paper"];
    [window center];
    
    glView=[[GLView alloc]initWithFrame:win_rect];
    
    [window setContentView:glView];
    [window setDelegate:self];
    [window makeKeyAndOrderFront:self];
}

- (void)applicationWillTerminate:(NSNotification *)notification
{
    //code
    
    if(gpFile)
    {
        fclose(gpFile);
        gpFile=NULL;
    }
}

- (void)windowWillClose:(NSNotification *)notification
{
    //code
    [NSApp terminate:self];
}

- (void)dealloc
{
    //code
    [glView release];
    
    [window release];
    
    [super dealloc];
}
@end

@implementation GLView
{
    @private
    CVDisplayLinkRef displayLink;   //meansCVDisplayLink *displayLink;
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
    
}

- (id)initWithFrame:(NSRect)frame;
{
    //code
    self=[super initWithFrame:frame];
    
    if(self)
    {
        [[self window]setContentView:self];
        
        NSOpenGLPixelFormatAttribute attrs[]=
        {
            //must specify the 4.1 core version
            NSOpenGLPFAOpenGLProfile,
            NSOpenGLProfileVersion4_1Core,
            //specify the display ID to associates the GL context with(main display for now)
            NSOpenGLPFAScreenMask, CGDisplayIDToOpenGLDisplayMask(kCGDirectMainDisplay),
            NSOpenGLPFANoRecovery,
            NSOpenGLPFAAccelerated,
            NSOpenGLPFAColorSize, 24,
            NSOpenGLPFADepthSize, 24,
            NSOpenGLPFAAlphaSize, 8,
            NSOpenGLPFADoubleBuffer,
            0};//last 0 is must
        NSOpenGLPixelFormat *pixelFormat=[[[NSOpenGLPixelFormat alloc]initWithAttributes:attrs] autorelease];
        
        if(pixelFormat==nil)
        {
            fprintf(gpFile, "No Valid OpenGL Pixel Format Is Available.Exitting...");
            [self release];
            [NSApp terminate:self];
        }
        fprintf(gpFile, "pixelFormat is not nil\n");
        NSOpenGLContext *glContext=[[[NSOpenGLContext alloc]initWithFormat:pixelFormat shareContext:nil]autorelease];
        
        [self setPixelFormat:pixelFormat];
        
        [self setOpenGLContext:glContext];
    }
    return(self);
}

- (CVReturn)getFrameForTime:(const CVTimeStamp *)pOutputTime
{
    //code
    NSAutoreleasePool *pool=[[NSAutoreleasePool alloc]init];
    
    [self drawView];
    
    [pool release];
    return(kCVReturnSuccess);
}

- (void)prepareOpenGL
{
    //variables
    GLint iShaderCompileStatus = 0;
    GLint iProgramLinkStatus = 0;
    GLint iInfoLogLength = 0;
    GLchar *szInfoLog = NULL;
    
    //code
    //OpenGL Info
    fprintf(gpFile, "OpenGL Version : %s\n", glGetString(GL_VERSION));
    fprintf(gpFile, "GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    [[self openGLContext]makeCurrentContext];
    
    GLint swapInt=1;
    
    [[self openGLContext]setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];
    
    //define vertex shader object
    gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

    //write vertex shader code
    const GLchar *vertexShaderSourceCode =
        "#version 410" \
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
        1.    Call glGetShaderiv(), and get the compile status of that object.
        2.    check that compile status, if it is GL_FALSE then shader has compilation error.
        3.    if(GL_FALSE) call again the glGetShaderiv() function and get the
            infoLogLength.
        4.    if(infoLogLength > 0) then call glGetShaderInfoLog() function to get the error
            information.
        5.    Print that obtained logs in file.
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

                fprintf(gpFile, "Vertex Shader Error : \n %s \n", szInfoLog);
                free(szInfoLog);
                [self release];
                [NSApp terminate:self];
            }
        }
    }

    /* Fragment Shader Code */

    //define fragment shader object
    gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

    //write shader code
    const GLchar *fragmentShaderSourceCode =
        "#version 410" \
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
                fprintf(gpFile, "Fragment Shader Error : \n %s \n", szInfoLog);
                free(szInfoLog);
                [self release];
                [NSApp terminate:self];
                
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
                fprintf(gpFile, "Program Link Error : \n %s\n", szInfoLog);
                free(szInfoLog);
                [self release];
                [NSApp terminate:self];
                
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
    
    //depth
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    //make orthograhic projection matrix a identity matrix
    perspectiveProjectionMatrix = vmath::mat4::identity();    //set background color
    
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
    CVDisplayLinkSetOutputCallback(displayLink, &MyDisplayLinkCallback, self);
    CGLContextObj cglContext=(CGLContextObj)[[self openGLContext]CGLContextObj];
    CGLPixelFormatObj cglPixelFormat=(CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
    CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink, cglContext, cglPixelFormat);
    fprintf(gpFile, "before CVDisplayLinkStart\n");
    CVDisplayLinkStart(displayLink);
}

- (void)reshape
{
    //code
    CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
    
    NSRect rect=[self bounds];
    
    GLfloat width=rect.size.width;
    GLfloat height=rect.size.height;
    
    if(height==0)
        height=1;
    
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
    
    perspectiveProjectionMatrix = vmath::perspective(45.0f, ((GLfloat)width / (GLfloat)height), 0.1f, 100.0f);
    
    CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
}

- (void)drawRect:(NSRect)dirtyRect
{
    //code
    
    [self drawView];
}

- (void)drawView
{
    //code
    [[self openGLContext]makeCurrentContext];
    
    CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
    
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
    CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
    CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
}

- (BOOL)acceptsFirstResponder
{
    //code
    [[self window]makeFirstResponder:self];
    return(YES);
}

- (void)keyDown:(NSEvent *)theEvent
{
    //code
    int key=(int)[[theEvent characters]characterAtIndex:0];
    switch(key)
    {
        case 27:    //escape key
            [self release];
            [NSApp terminate:self];
            break;
            
        case 'F':
        case 'f':
            [[self window]toggleFullScreen:self];    //repainting occures automatically
            
            break;
        default:
            break;
    }
}

- (void)mouseDown:(NSEvent *)theEvent
{
    //code
    
}

- (void)mouseDragged:(NSEvent *)theEvent
{
    //code
}

- (void)rightMouseDown:(NSEvent *)theEvent
{
    //code
    
}

- (void)dealloc
{
    //code
    
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
    [super dealloc];
}
@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink,const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext)
{
    CVReturn result = [(GLView *)pDisplayLinkContext getFrameForTime:pOutputTime];
    return(result);
}
