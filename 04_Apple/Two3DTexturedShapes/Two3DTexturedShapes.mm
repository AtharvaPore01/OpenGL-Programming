//headers
#import <Foundation/Foundation.h>       //for NSAutoreleasePool
#import <Cocoa/Cocoa.h>                 //for NSApp

#import <QuartzCore/CVDisplayLink.h>        //CoreDisplayVideoLink

#import <OpenGL/gl3.h>
#import <OpenGL/gl3ext.h>

#import "vmath.h"

//enum
enum
{
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOODR_0
};

// global function declaration
CVReturn MyDisplayLinkCallback(CVDisplayLinkRef, const CVTimeStamp *, const CVTimeStamp *, CVOptionFlags, CVOptionFlags *, void *);

//global variables
FILE *gpFile = NULL;

//interface declaration
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface GLView : NewOpenGLView
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
    GLView *view;
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    //code
    //log file
    NSBundle *mainBundle=[NSBundle mainBundle];
    NSString *appDirName=[mainBundle bundlePath];
    NSString *parentDirName=[appDirName stringByDeletingLastPathComponent];
    NSString *logFileNameWithPath=[NSString stringWithFormat:@"%@/Log.txt", parentDirPath];
    const char *pszLogFileNameWithPath=[logFileNameWithPath cStringUsingEncoding:NSASCIIStringEncoding];
    gpFile=fopen(pszLogFileNameWithPath, "w");
    if(gpFile==NULL)
    {
        printf("Can not create the log file.\nExitting...\n");
        [self release];
        [NSApp terminate:self];
    }
    fprintf(gpFile, "Program Is Started Successfully.\n");
    
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

    view = [[GLView alloc]initWithFrame:win_rect];

    [window setContentView:view];
    [window setDelegate:self];
    [window makeKeyAndOrderFront:self];
}

- (void)applicationWillTerminate:(NSNotification *)notification
{
    //code
    fprintf(gpFile, "Program IsTerminated Successfully.\n");

    if(gpFile)
    {
        fclose(gpFile);
        gpFile=NULL;
    }
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

@implementation GLView
{
@private    
    CVDisplayLink displayLink;

    GLuint gVertexShaderObject;
    GLuint gFragmentShaderObject;
    GLuint gShaderProgramObject;

    GLuint vao_pyramid;
    GLuint vbo_pyramid_position;
    GLuint vbo_pyramid_texture;

    GLuint vao_cube;
    GLuint vbo_cube_position;
    GLuint vbo_cube_texture;

    GLuint samplerUniform;
    GLuint texture_kundali;
    GLuint texture_stone;

    GLuint mvpUniform;
    vmath::mat4 perspectiveProjectionMatrix;

    //Rotation variables
    GLfloat rotation_angle_triangle = 0.0f;
    GLfloat rotation_angle_rectangle = 0.0f;
}

- (id)initWithFrame:(NSRect)frame;
{
    //code
    self = [super initWithFrame:frame];

    if(self)
    {
        [[self window]setContentView:self];

        NSOpenGLPixelFormatAttribute attrs[]=
        {
            //must specify the 4.1 core profile
            NSOpenGLPFAOpenGLProfile,
            NSOpenGLProfileVersion4_1Core,
            //Specify the display ID to associate the GL Context with main display for now.
            NSOpenGLPFAScreenMask, CGDisplayIDToOpenGLDisplayMask(lCGDirectMainDisplay),
            NSOpenGLPFANoRecovery,
            NSOpenGLPFAAccelerated,
            NSOpenGLPFAColorSize,24,
            NSOpenGLPFADepthSize,24,
            NSOpenGLPFAAlphaSize,8,
            NSOpenGLPFADoubleBuffer,
            0};

        NSOpenGLPixelFormat *pixelFormat=[[[NSOpenGLPixelFormat alloc]
        initWithAttributes:attrs] autorelease];
        if(pixelFormat==nil)
        {
            fprintf(gpFile, "No valid OpenGL Pixel Format Is Available.\nExitting...\n");
            [self release];
            [NSApp terminate:self];
        }

        NSOpenGLContext *glContext=[[[NSOpenGLContext alloc]
        initWithFormat:pixelFormat shareContext:nil]autorelease];

        [self setPixelFormat:pixelFormat];

        [self setOpenGLContext:glContext];//it automatically release the older context, if present, and sets the newer one.
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

-(void)prepareOpenGL
{
    //code
    //opengl info
    fprintf(gpFile, "OpenGL Version : %s\n", glGetString(GL_VERSION));
    fprintf(gpFile, "GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    [[self openGLContext]makeCurrentContext];

    GLint swapInt=1;
    [[self openGLContext]setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];

    /* Vertex Shader Code */

	//define vertex shader object
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//write vertex shader code
	const GLchar *vertexShaderSourceCode =
		"#version 410" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexCoord;" \
		"out vec2 out_texcoord;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_texcoord = vTexCoord;" \
		"}";

	//specify above source code to vertex shader object
	glShaderSource(gVertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

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

				fprintf_s(gpFile, "Vertex Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
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
		"in vec2 out_texcoord;" \
		"out vec4 FragColor;" \
		"uniform sampler2D u_sampler;" \
		"void main(void)" \
		"{" \
		"FragColor = texture(u_sampler, out_texcoord);" \
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
				fprintf_s(gpFile, "Fragment Shader Error : \n %s \n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
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
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_TEXCOODR_0, "vTexCoord");

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
				fprintf_s(gpFile, "Program Link Error : \n %s\n", szInfoLog);
				free(szInfoLog);
				oglUninitialise();
				DestroyWindow(hwnd);
				exit(0);
			}
		}
	}

	//post linking retriving uniform location
	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");
	samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");

	//triangle vertices declaration
	const GLfloat pyramidVertices[] =
	{
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,

		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};
	const GLfloat pyramidTexcoord[]
	{
		0.5f, 1.0f,
		0.0f, 0.0f, 
		1.0f, 0.0f,

		0.5f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		0.5f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		0.5f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLfloat cubeVertices[] =
	{
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,

		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, -1.0f
	};

	const GLfloat cubeTexcoord[] = 
	{
		0.0f, 1.0f,
		0.0f, 0.0f, 
		1.0f, 0.0f,
		1.0f, 1.0f,

		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f, 
		0.0f, 1.0f,

		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,

		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f, 
		0.0f, 0.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f
	};

	//create vao and vbo
	//TRIANGLE

	//position
	glGenVertexArrays(1, &vao_pyramid);
	glBindVertexArray(vao_pyramid);
	glGenBuffers(1, &vbo_pyramid_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pyramid_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//texture
	glGenBuffers(1, &vbo_pyramid_texture);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pyramid_texture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidTexcoord), pyramidTexcoord, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOODR_0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOODR_0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	
	//RECTANGLE
	for (int i = 0; i < 72; i++)
	{
		if (cubeVertices[i] == -1.0f)
		{
			cubeVertices[i] = cubeVertices[i] + 0.25f;
		}
		else if (cubeVertices[i] == 1.0f)
		{
			cubeVertices[i] = cubeVertices[i] - 0.25f;
		}
	}

	glGenVertexArrays(1, &vao_cube);
	glBindVertexArray(vao_cube);
	glGenBuffers(1, &vbo_cube_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//texture
	glGenBuffers(1, &vbo_cube_texture);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_texture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeTexcoord), cubeTexcoord, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOODR_0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOODR_0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//clear the window
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glDisable(GL_CULL_FACE);
	
	//depth
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	//texture
	glEnable(GL_TEXTURE_2D);
    texture_stone=[self loadTexture:"Stone.bmp"];
    texture_kundali=[self loadTexture:"Kundali.bmp"];
	
	//make orthograhic projection matrix a identity matrix
	perspectiveProjectionMatrix = vmath::mat4::identity();
    
    CVDisplayLinkCreateWithActiveCGDisplay(&displayLink);
    CVDisplayLinkSetOutputCallback(displayLink, &MyDisplayLinkCallback, self);
    CGLContextObj cglContext=(CGLContextObj)[[self openGLContext]CGLContextObj];
    CGLPixelFormatObj cglPixelFormat=(CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
    CVDisplayLinkSetCurrentCGDDisplayFromOpenGLContext(displayLink, cglContext, cglPixelFormat);
    CVDisplayLinkStart(displayLink);
}

- (GLuint)loadTexture:(const char *)texFileName
{
    NSBundle *mainBundle=[NSBundle mainBundle];
    NSString *appDirName=[mainBundle bundlePath];
    NSString *parentDirName=[appDirName stringByDeletingLastPathComponent];
    NSString *textureFileNameWithPath=[NSString stringWithFormat:@"%@/%s", parentDirName, texFileName];

    NSImage *bmpImage=[[NSImage alloc]initWithContentOfFile:textureFileNameWithPath];
    if(!bmpImage)
    {
        NSLog(@"Can't Find %@", textureFileNameWithPath);
        return(0);
    }

    CGImageRef cgImage=[bmpImage CGImageForProposedRect:nil context:nil hints:nil];

    int w = (int)CGImageGetWidth(cgImage);
    int h = (int)CGImageGetHeight(cgImage);

    CGDataRef imageData = CGDataProviderCopyData(CGImageGetDataProvider(cgImage));

    void *pixels = (void *)CGDataGetBytePtr(imageData);

    GLuint bmpTexture;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	//generate texture
	glGenTextures(1, bmpTexture);

	//bind texture
	glBindTexture(GL_TEXTURE_2D, *bmpTexture);

	//set parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RGBA,
		w,
		h,
		0,
		GL_RGBA, //GL_BGR_EXT,
		GL_UNSIGNED_BYTE,
		pixels);

	glGenerateMipmap(GL_TEXTURE_2D);

    CGRelease(imageData);
    return(bmpTexture);
}

- (void)reshape
{
    //code
    CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

    NSRect rect=[self bounds];

    GLfloat width=rect.size.width;
    GLfloat height=rect.size.height;

    if(height==0)
    {
        height=1;
    }

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    perspectiveProjectionMatrix = vmath::perspective(45.0f, ((GLfloat)width / (GLfloat)height), 0.1f, 100.0f);

    CGLUnlockContext((CGLContext)[self openGLContext]CGLContextObj);
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

    glClearColor(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(gShaderProgramObject);

	//declaration of metrices
	vmath::mat4 modelViewMatrix;
	vmath::mat4 modelViewProjectionMatrix;
	vmath::mat4 translationMatrix;
	vmath::mat4 rotationMatrix;

	/* TRIANGLE */

	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();

	//do necessary transformations here
	translationMatrix = vmath::translate(-2.0f, 0.0f, -8.0f);
	rotationMatrix = vmath::rotate(rotation_angle_triangle, 0.0f, 1.0f, 0.0f);

	//do necessary matrix multiplication
	modelViewMatrix = modelViewMatrix * translationMatrix;
	modelViewMatrix = modelViewMatrix * rotationMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	//acitve texture
	glActiveTexture(GL_TEXTURE0);

	//bind texture
	glBindTexture(GL_TEXTURE_2D, texture_stone);

	//push in fragment
	glUniform1i(samplerUniform, 0);

	//bind with vao
	glBindVertexArray(vao_pyramid);


	//draw scene
	glDrawArrays(GL_TRIANGLES, 0, 12);

	//unbind texture
	//glBindTexture(GL_TEXTURE_2D, 0);

	//unbind vao
	glBindVertexArray(0);

	/* RECTANGLE */
	//init above metrices to identity
	modelViewMatrix = vmath::mat4::identity();
	modelViewProjectionMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();

	//do necessary transformations here
	translationMatrix = vmath::translate(2.0f, 0.0f, -8.0f);
	rotationMatrix = vmath::rotate(rotation_angle_rectangle, rotation_angle_rectangle, rotation_angle_rectangle);

	//do necessary matrix multiplication
	modelViewMatrix = modelViewMatrix * translationMatrix;
	modelViewMatrix = modelViewMatrix * rotationMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrics to shaders in respective uniforms
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	//acitve texture
	glActiveTexture(GL_TEXTURE0);

	//bind texture
	glBindTexture(GL_TEXTURE_2D, texture_kundali);

	//push in fragment
	glUniform1i(samplerUniform, 0);

	//bind with vao
	glBindVertexArray(vao_cube);

	//bind with texture 

	//draw scene

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	//unbind texture
	//glBindTexture(GL_TEXTURE_2D, 0);

	//unbind vao
	glBindVertexArray(0);

	//unuse program
	glUseProgram(0);

    CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
    CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
    rotation_angle_triangle = rotation_angle_triangle + 0.05f;
	if (rotation_angle_triangle >= 360.0f)
	{
		rotation_angle_triangle = 0.0f;
	}
	rotation_angle_rectangle = rotation_angle_rectangle + 0.05f;
	if (rotation_angle_rectangle >= 360.0f)
	{
		rotation_angle_rectangle = 0.0f;
	}
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
}

- (void)mouseDragged:(NSEvent *)theEvent
{
    //code
}

- (void)rightMouseDown:(NSEvent *)theEvent
{
    //code
}

- (void) dealloc
{
    //code
    if (vbo_pyramid_position)
	{
		glDeleteBuffers(1, &vbo_pyramid_position);
		vbo_pyramid_position = 0;
	}
	if (vbo_pyramid_texture)
	{
		glDeleteBuffers(1, &vbo_pyramid_texture);
		vbo_pyramid_texture = 0;
	}
	if (vao_pyramid)
	{
		glDeleteVertexArrays(1, &vao_pyramid);
		vao_pyramid = 0;
	}

	if (vbo_cube_position)
	{
		glDeleteBuffers(1, &vbo_cube_position);
		vbo_cube_position = 0;
	}
	if (vbo_cube_texture)
	{
		glDeleteBuffers(1, &vbo_cube_texture);
		vbo_cube_texture = 0;
	}
	if (vao_cube)
	{
		glDeleteVertexArrays(1, &vao_cube);
		vao_cube = 0;
	}

    //destroy texture
    if(texture_stone)
    {
        glDeleteTexture(1, &texture_stone);
        texture_stone = 0;
    }

    if(texture_kundali)
    {
        glDeleteTexture(1, &texture_kundali);
        texture_kundali = 0;
    }

    //detach vertex and fragment shader from the program
    glDetachShader(gShaderProgramObject, gVertexShaderObject);
    glDetachShader(gShaderProgramObject, gFragmentShaderObject);
    //delete vertex shader object
    glDeleteShader(gVertexShaderObject);
    gVertexShaderObject = 0;
	//delete fragment shader object
    glDeleteShader(gFragmentShaderObject);
    gFragmentShaderObject = 0;
    //delete shader program object
    glDeleteProgram(gShaderProgramObject);
    gShaderProgramObject = 0;
/*
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
*/
    CVDisplayLinkStop(displayLink);
    CVDisplayLinkRelease(displayLink);

    [super dealloc];
}

@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext)
{
    CVReturn result=[(GLView *)pDisplayLinkContext getFrameForTime:pOutputTime];
    return(result);
}
