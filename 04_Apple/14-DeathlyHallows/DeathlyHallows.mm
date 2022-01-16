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

//deathly hallow structure
struct deathlyHallow
{
    //for distance finding and semi-perimeter
    GLfloat a = 0.0f, b = 0.0f, c = 0.0f;
    GLfloat Perimeter = 0.0f;
    const GLfloat x1 = 0.0f;
    const GLfloat x2 = -1.0f;
    const GLfloat x3 = 1.0f;
    const GLfloat y1 = 1.0f;
    const GLfloat y2 = -1.0f;
    const GLfloat y3 = -1.0f;

    //for area of triangle
    GLfloat AreaOfTriangle = 0.0f;
    //for circle
    GLfloat x_center = 0.0f;
    GLfloat y_center = 0.0f;
    GLfloat radius = 0.0f;
};
deathlyHallow dh;

//initial position of triangle, circle, line
GLfloat x_triangle = 3.0f;
GLfloat y_triangle = -3.0f;
GLfloat x_circle = -3.0f;
GLfloat y_circle = -3.0f;
GLfloat y_line = 3.0f;

GLfloat rotationAngle;
bool bCircle = false;
bool bLine = false;

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
    [window setTitle:@"macOS Window:Blue window"];
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

    GLuint vao_triangle;
    GLuint vao_circle;
    GLuint vao_line;

    GLuint vbo_triangle;
    GLuint vbo_circle;
    GLuint vbo_line;
    GLuint mvpUniform;
    vmath::mat4 perspectiveProjectionMatrix;}

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
        "uniform mat4 u_mvp_matrix;" \
        "void main(void)" \
        "{" \
        "gl_Position = u_mvp_matrix * vPosition;" \
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
        "out vec4 FragColor;" \
        "void main(void)" \
        "{" \
        "FragColor = vec4(1.0, 1.0, 0.0, 1.0);" \
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

    //triangle vertices declaration
    const GLfloat triangleVertices[] =
    {
        0.0f, 1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    const GLfloat lineVertices[] =
    {
        0.0f, 1.0f, 0.0f,
        0.0f, -1.0f, 0.0f
    };

    //create vao and vbo

    //triangle
    glGenVertexArrays(1, &vao_triangle);
    glBindVertexArray(vao_triangle);
    glGenBuffers(1, &vbo_triangle);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_triangle);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //line
    glGenVertexArrays(1, &vao_line);
    glBindVertexArray(vao_line);
    glGenBuffers(1, &vbo_line);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_line);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lineVertices), lineVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //circle
    glGenVertexArrays(1, &vao_circle);
    glBindVertexArray(vao_circle);
    glGenBuffers(1, &vbo_circle);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_circle);
    glBufferData(GL_ARRAY_BUFFER, 1 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //clear the window
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
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

- (void)calculateSemiPerimeter
{
    //code
    dh.a = sqrtf((powf((dh.x2 - dh.x1), 2) + powf((dh.y2 - dh.y1), 2)));
    dh.b = sqrtf((powf((dh.x3 - dh.x2), 2) + powf((dh.y3 - dh.y2), 2)));
    dh.c = sqrtf((powf((dh.x1 - dh.x3), 2) + powf((dh.y1 - dh.y3), 2)));
    
    //Semi Perimeter
    dh.Perimeter = (dh.a + dh.b + dh.c) / 2;
}

- (void) calculateAreaOfTriangle
{
    //code
    dh.AreaOfTriangle = sqrtf(dh.Perimeter * (dh.Perimeter - dh.a) * (dh.Perimeter - dh.b) * (dh.Perimeter - dh.c));
}

- (void) calculateRadius
{
    //code
    dh.radius = dh.AreaOfTriangle / dh.Perimeter;
}

- (void) calculateCenterOfTheCircle
{
    //code
    dh.x_center = ((dh.a * dh.x3) + (dh.b * dh.x1) + (dh.c * dh.x2)) / (dh.a + dh.b + dh.c);
    dh.y_center = ((dh.a * (dh.y3)) + (dh.b * (dh.y1)) + (dh.c * (dh.y2))) / (dh.a + dh.b + dh.c);
}

- (void) deathlyHallowsCircle
{
    GLfloat circleVertices[3];

    //code
    //bind with vao
    glBindVertexArray(vao_circle);
    for (GLfloat angle = 0.0f; angle < (2.0f * M_PI); angle = angle + 0.01f)
    {
        circleVertices[0] = ((cosf(angle) * dh.radius) + dh.x_center);
        circleVertices[1] = ((sinf(angle) * dh.radius) + dh.y_center);
        circleVertices[2] = 0.0f;

        //vertices
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circle);
        glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //draw scene
        glPointSize(1.5f);
        glDrawArrays(GL_POINTS, 0, 1);
        //glDrawArrays(GL_LINE_LOOP, 0, 10);
    }

    //unbind vao
    glBindVertexArray(0);
}

- (void)deathlyHallowsLine
{
    //bind with vao
    glBindVertexArray(vao_line);

    glDrawArrays(GL_LINES, 0, 2);

    glBindVertexArray(0);
}

-(void) deathlyHallowTriangle
{
    //code
    [self calculateSemiPerimeter];
    [self calculateAreaOfTriangle ];
    [self calculateRadius ];
    [self calculateCenterOfTheCircle ];

    //bind with vao
    glBindVertexArray(vao_triangle);

    glDrawArrays(GL_LINES, 0, 2);
    glDrawArrays(GL_LINES, 2, 2);
    glDrawArrays(GL_LINES, 4, 2);

    //unbind vao
    glBindVertexArray(0);
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
    vmath::mat4 translationMatrix;
    vmath::mat4 rotationMatrix;
    vmath::mat4 translationMatrix_circle;
    vmath::mat4 translationMatrix_triangle;
    vmath::mat4 translationMatrix_line;

    //init above metrices to identity
    modelViewMatrix = vmath::mat4::identity();
    modelViewProjectionMatrix = vmath::mat4::identity();
    translationMatrix = vmath::mat4::identity();
    rotationMatrix = vmath::mat4::identity();
    translationMatrix_triangle = vmath::mat4::identity();
    translationMatrix_circle = vmath::mat4::identity();
    translationMatrix_line = vmath::mat4::identity();

    //triangle
    //deathly hallows creation code will be here
    translationMatrix = vmath::translate(0.0f, 0.0f, -6.0f);
    translationMatrix_triangle = vmath::translate(x_triangle, y_triangle, 0.0f);
    rotationMatrix = vmath::rotate(rotationAngle, 0.0f, 1.0f, 0.0f);
    
    //do necessary transformations here
    modelViewMatrix *= translationMatrix;
    modelViewMatrix *= translationMatrix_triangle;
    modelViewMatrix *= rotationMatrix;

    //do necessary matrix multiplication
    modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
        
    //send necessary matrics to shaders in respective uniforms
    glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

    [self deathlyHallowTriangle ];
    if (x_triangle >= 0.0f && y_triangle <= 0.0f)
    {
        y_triangle = y_triangle + 0.005f;
        x_triangle = x_triangle - 0.005f;
        if (y_triangle > 0.0f)
        {
            bCircle = true;
        }
    }

    //circle
    if (bCircle == true)
    {
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();
        translationMatrix = vmath::mat4::identity();
        rotationMatrix = vmath::mat4::identity();

        //deathly hallows creation code will be here
        translationMatrix = vmath::translate(0.0f, 0.0f, -6.0f);
        translationMatrix_circle = vmath::translate(x_circle, y_circle, 0.0f);
        rotationMatrix = vmath::rotate(rotationAngle, 0.0f, 1.0f, 0.0f);
    
        //do necessary transformations here
        modelViewMatrix *= translationMatrix;
        modelViewMatrix *= translationMatrix_circle;
        modelViewMatrix *= rotationMatrix;

        //do necessary matrix multiplication
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        [self deathlyHallowsCircle ];
        if ((x_circle <= 0.0f && y_circle <= 0.0f))
        {
            y_circle = y_circle + 0.005f;
            x_circle = x_circle + 0.005f;
            if (x_circle > 0.0f)
            {
                bLine = true;
            }
        }
    }
    
    //line
    if (bLine == true)
    {
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();
        translationMatrix = vmath::mat4::identity();
        
        //deathly hallows creation code will be here
        translationMatrix = vmath::translate(0.0f, 0.0f, -6.0f);
        translationMatrix_line = vmath::translate(0.0f, y_line, 0.0f);
        
        //do necessary transformations here
        modelViewMatrix *= translationMatrix;
        modelViewMatrix *= translationMatrix_line;
        modelViewMatrix *= rotationMatrix;

        //do necessary matrix multiplication
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        [self deathlyHallowsLine ];
        if ((y_line >= 0.0f))
        {
            y_line = y_line - 0.005f;
        }
    }

    //unuse program
    glUseProgram(0);
    CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
    CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
    rotationAngle = rotationAngle + 1.0f;
    if (rotationAngle >= 360.0f)
    {
        rotationAngle = 0.0f;
    }}

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
    //code
    if (vbo_line)
    {
        glDeleteBuffers(1, &vbo_line);
        vbo_line = 0;
    }
    if (vbo_circle)
    {
        glDeleteBuffers(1, &vbo_circle);
        vbo_circle = 0;
    }
    if (vbo_triangle)
    {
        glDeleteBuffers(1, &vbo_triangle);
        vbo_triangle = 0;
    }

    if (vao_circle)
    {
        glDeleteVertexArrays(1, &vao_circle);
        vao_circle = 0;
    }
    if (vao_line)
    {
        glDeleteVertexArrays(1, &vao_line);
        vao_line = 0;
    }
    if (vao_triangle)
    {
        glDeleteVertexArrays(1, &vao_triangle);
        vao_triangle = 0;
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
