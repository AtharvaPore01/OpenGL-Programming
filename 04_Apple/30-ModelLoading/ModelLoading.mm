#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#import <QuartzCore/CVDisplayLink.h>    //core video display link

//opengl related header
#import <OpenGL/gl3.h>
#import <OpenGL/gl3ext.h>
#import "vmath.h"

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
GLfloat RotationAngle = 0.0f;
BOOL bAnimate = NO;
BOOL bLight = NO;

//light values
float LightAmbient[4] = { 0.25f, 0.25f, 0.25f, 0.25f };
float LightDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightPosition[4] = { 100.0f, 100.0f, 100.0f, 1.0f };            //{ 1.0f, 1.0f, 1.0f, 1.0f };

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 128.0f;                            //{128.0f};

//model loading variables
struct vec_int
{
    int *p;
    int size;
};

struct vec_float
{
    float *pf;
    int size;
};

#define BUFFER_SIZE 1024
char buffer[BUFFER_SIZE];

FILE *gpMeshFile = NULL;

struct vec_float *gpVertices = NULL;
struct vec_float *gpTexture = NULL;
struct vec_float *gpNormal = NULL;

struct vec_float *gp_sorted_vertices = NULL;
struct vec_float *gp_sorted_texture = NULL;
struct vec_float *gp_sorted_normal = NULL;

struct vec_int *gp_indices_vertices = NULL;
struct vec_int *gp_indices_texture = NULL;
struct vec_int *gp_indices_normal = NULL;


//interface declaration
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface GLView : NSOpenGLView
@end

//entry point function
int main(int argc, char *argv[])
{
    //code
    NSAutoreleasePool *pPool = [[NSAutoreleasePool alloc]init];
    
    NSApp=[NSApplication sharedApplication];
    
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
    [window setTitle:@"macOS Window:Model Loading"];
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

    GLuint vao;
    GLuint vbo_vertices;
    GLuint vbo_normal;
    GLuint vbo_texcoord;
    GLuint element_buffer_vertices;

    GLuint model_uniform;
    GLuint view_uniform;
    GLuint projection_uniform;

    GLuint La_uniform;
    GLuint Ld_uniform;
    GLuint Ls_uniform;
    GLuint lightPosition_uniform;

    GLuint Ka_uniform;
    GLuint Kd_uniform;
    GLuint Ks_uniform;
    GLuint shininess_uniform;
    GLuint LKeyPressed_Uniform;

    GLuint samplerUniform;
    
    GLuint marble_texture;
    
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

- (GLuint)loadTexture:(const char *)texFileName
{
    NSBundle *mainBundle = [NSBundle mainBundle];
    NSString *appDirName = [mainBundle bundlePath];
    NSString *parentDirPath = [appDirName stringByDeletingLastPathComponent];
    NSString *textureFileNameWithPath = [NSString stringWithFormat:@"%@/%s", parentDirPath, texFileName];
    
    //convert the image in cocoa format
    NSImage *bmpImage = [[NSImage alloc]initWithContentsOfFile:textureFileNameWithPath];
    if(!bmpImage)
    {
        NSLog(@"can't find %@", textureFileNameWithPath);
        return(0);
    }
    
    CGImageRef cgImage = [bmpImage CGImageForProposedRect:nil context:nil hints:nil];
    
    int w = (int)CGImageGetWidth(cgImage);
    int h = (int)CGImageGetHeight(cgImage);
    
    CFDataRef imageData = CGDataProviderCopyData(CGImageGetDataProvider(cgImage));
    void *pixel = (void *)CFDataGetBytePtr(imageData);
    
    GLuint bmpTexture;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    //generate texture
    glGenTextures(1, &bmpTexture);

    //bind texture
    glBindTexture(GL_TEXTURE_2D, bmpTexture);

    //set parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 w,
                 h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 pixel);

    glGenerateMipmap(GL_TEXTURE_2D);
    
    CFRelease(imageData);
    return(bmpTexture);
}

- (void)prepareOpenGL
{
    //variables
    GLint iShaderCompileStatus = 0;
    GLint iProgramLinkStatus = 0;
    GLint iInfoLogLength = 0;
    GLchar *szInfoLog = NULL;
    
    void oglLoadMesh(void);
    
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
        "in vec3 vNormal;" \
        "in vec2 vTexcoord;" \
        
        "uniform mat4 u_model_matrix;" \
        "uniform mat4 u_view_matrix;" \
        "uniform mat4 u_projection_matrix;" \
        "uniform int u_LKeyPressed;" \
        "uniform vec4 u_light_position;" \

        "out vec2 out_texcoord;" \
        "out vec3 t_norm;" \
        "out vec3 light_direction;" \
        "out vec3 viewer_vector;" \
        "void main(void)" \
        "{" \
        "if (u_LKeyPressed == 1)" \
        "{" \
        "vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" \
        "mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" \
        "t_norm = normal_matrix * vNormal;" \
        "light_direction = vec3(u_light_position - eye_coordinates);" \
        "viewer_vector = vec3(-eye_coordinates);" \
        "}" \
        "gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
        "out_texcoord = vTexcoord;" \
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
        "in vec2 out_texcoord;" \
        "in vec3 t_norm;" \
        "in vec3 light_direction;" \
        "in vec3 viewer_vector;" \

        "uniform int u_LKeyPressed;" \
        "uniform vec3 u_La;" \
        "uniform vec3 u_Ld;" \
        "uniform vec3 u_Ls;" \
        "uniform vec4 u_light_position;" \
        "uniform vec3 u_Ka;" \
        "uniform vec3 u_Kd;" \
        "uniform vec3 u_Ks;" \
        "uniform float shininess;" \

        "out vec3 phong_ads_light;" \
        "out vec4 FragColor;" \
        "uniform sampler2D u_sampler;" \
        "void main(void)" \
        "{" \
        "if(u_LKeyPressed == 1)" \
        "{" \
        "vec3 normalised_transformed_normal = normalize(t_norm);" \
        "vec3 normalised_light_direction = normalize(light_direction);" \
        "vec3 normalised_viewer_vector = normalize(viewer_vector);" \
        "vec3 reflection_vector = reflect(-normalised_light_direction, normalised_transformed_normal);" \
        "float tn_dot_LightDirection = max(dot(normalised_light_direction, normalised_transformed_normal), 0.0);" \
        "vec3 ambient = u_La * u_Ka;" \
        "vec3 diffuse = u_Ld * u_Kd * tn_dot_LightDirection;" \
        "vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalised_viewer_vector), 0.0), shininess);" \
        "phong_ads_light = (ambient + diffuse + specular) * vec3(texture(u_sampler, out_texcoord));" \
        "}" \
        "else" \
        "{" \
        "phong_ads_light = vec3(texture(u_sampler, out_texcoord));" \
        "}" \
        "FragColor =  vec4(phong_ads_light,1.0);" \
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
    glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_TEXCOODR_0, "vTexcoord");
    glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_NORMAL, "vNormal");
    
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

    //load mesh
    oglLoadMesh();

    //post linking retriving uniform location
    model_uniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
    view_uniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
    projection_uniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
    LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject, "u_LKeyPressed");
    La_uniform = glGetUniformLocation(gShaderProgramObject, "u_La");
    Ld_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
    Ls_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ls");
    Ka_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
    Kd_uniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
    Ks_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
    shininess_uniform = glGetUniformLocation(gShaderProgramObject, "shininess");
    lightPosition_uniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");

    samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");
    //create vao and vbo
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    //position
    glGenBuffers(1, &vbo_vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);

    glBufferData(    GL_ARRAY_BUFFER,
                    (gp_sorted_vertices->size) * sizeof(float),
                    gp_sorted_vertices->pf,
                    GL_STATIC_DRAW);

    glVertexAttribPointer(    AMC_ATTRIBUTE_POSITION,
                            3,
                            GL_FLOAT,
                            GL_FALSE,
                            0,
                            NULL);

    glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //normal
    glGenBuffers(1, &vbo_normal);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);

    glBufferData(    GL_ARRAY_BUFFER,
                    (gp_sorted_normal->size) * sizeof(float),
                    gp_sorted_normal->pf,
                    GL_STATIC_DRAW);

    glVertexAttribPointer(    AMC_ATTRIBUTE_NORMAL,
                            3,
                            GL_FLOAT,
                            GL_FALSE,
                            0,
                            NULL);

    glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //texcoord
    glGenBuffers(1, &vbo_texcoord);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoord);

    glBufferData(    GL_ARRAY_BUFFER,
                    (gp_sorted_texture->size) * sizeof(float),
                    gp_sorted_texture->pf,
                    GL_STATIC_DRAW);

    glVertexAttribPointer(    AMC_ATTRIBUTE_TEXCOODR_0,
                            2,
                            GL_FLOAT,
                            GL_FALSE,
                            0,
                            NULL);

    glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOODR_0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //element buffer
    
    //vertices
    glGenBuffers(1, &element_buffer_vertices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_vertices);
    glBufferData(    GL_ELEMENT_ARRAY_BUFFER,
                    gp_indices_vertices->size * sizeof(int),
                    gp_indices_vertices->p,
                    GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    glBindVertexArray(0);
    
    //depth
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    
    //texture
    glEnable(GL_TEXTURE_2D);
    marble_texture = [self loadTexture:"marble.bmp"];

    //make orthograhic projection matrix a identity matrix
    perspectiveProjectionMatrix = vmath::mat4::identity();    //set background color
    
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    
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

    vmath::mat4 modelMatrix;
    vmath::mat4 viewMatrix;
    vmath::mat4 projectionMatrix;
    vmath::mat4 translationMatrix;
    vmath::mat4 rotationMatrix;

    //init above metrices to identity
    modelMatrix = vmath::mat4::identity();
    viewMatrix = vmath::mat4::identity();
    projectionMatrix = vmath::mat4::identity();
    translationMatrix = vmath::mat4::identity();
    rotationMatrix = vmath::mat4::identity();

    //do necessary transformations here
    translationMatrix = vmath::translate(0.0f, -1.5f, -10.0f);
    rotationMatrix = vmath::rotate(RotationAngle, 0.0f, 1.0f, 0.0f);

    //do necessary matrix multiplication
    modelMatrix = modelMatrix * translationMatrix;
    modelMatrix = modelMatrix * rotationMatrix;
    projectionMatrix *= perspectiveProjectionMatrix;

    //send necessary matrics to shaders in respective uniforms
    glUniformMatrix4fv(model_uniform, 1, GL_FALSE, modelMatrix);
    glUniformMatrix4fv(view_uniform, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(projection_uniform, 1, GL_FALSE, projectionMatrix);

    //active texture
    glActiveTexture(GL_TEXTURE0);

    //bind with texture
    glBindTexture(GL_TEXTURE_2D, marble_texture);

    //push in fragment shader
    glUniform1i(samplerUniform, 0);

    //if light is enabled
    if (bLight)
    {
        //notify shader that we pressed the "L" key
        glUniform1i(LKeyPressed_Uniform, 1);
        //send light intensityx
        glUniform3fv(La_uniform, 1, LightAmbient);
        glUniform3fv(Ld_uniform, 1, LightDiffuse);
        glUniform3fv(Ls_uniform, 1, LightSpecular);
        //send coeff. of material's reflectivity
        glUniform3fv(Ka_uniform, 1, MaterialAmbient);
        glUniform3fv(Kd_uniform, 1, MaterialDiffuse);
        glUniform3fv(Ks_uniform, 1, MaterialSpecular);
        //shininess
        glUniform1f(shininess_uniform, MaterialShininess);
        //send light position
        glUniform4fv(lightPosition_uniform, 1, LightPosition);
    }
    else
    {
        //notify shader that we aren't pressed the "L" key
        glUniform1i(LKeyPressed_Uniform, 0);
    }
    
    glBindVertexArray(vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_vertices);
    glDrawElements(    GL_TRIANGLES,
                    (gp_indices_vertices->size),
                    GL_UNSIGNED_INT,
                    NULL);
    
    glBindVertexArray(0);
    
    //unuse program
    glUseProgram(0);
    CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
    CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
    
    if(bAnimate == YES)
    {
        RotationAngle = RotationAngle + 1.0f;
        if (RotationAngle >= 360.0f)
        {
            RotationAngle = 0.0f;
        }
    }
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
        case 'L':
        case 'l':
            if (bLight == NO)
            {
                bLight = YES;
            }
            else
            {
                bLight = NO;
            }
            break;

        case 'A':
        case 'a':
            if (bAnimate == NO)
            {
                bAnimate = YES;
            }
            else
            {
                bAnimate = NO;
            }
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
    int destroy_vec_int(struct vec_int *p_vec_int);
    int destroy_vec_float(struct vec_float *p_vec_float);
    //code
    if (vbo_vertices)
    {
        glDeleteBuffers(1, &vbo_vertices);
        vbo_vertices = 0;
    }
    if (vbo_texcoord)
    {
        glDeleteBuffers(1, &vbo_texcoord);
        vbo_texcoord = 0;
    }
    if (vbo_normal)
    {
        glDeleteBuffers(1, &vbo_normal);
        vbo_normal = 0;
    }
    if (vao)
    {
        glDeleteVertexArrays(1, &vao);
        vao = 0;
    }

    if(element_buffer_vertices)
    {
        glDeleteBuffers(1, &element_buffer_vertices);
        element_buffer_vertices = 0;
    }

    
    
    if(gpVertices)
    {
        destroy_vec_float(gpVertices);
        gpVertices = NULL;
    }

    if(gpTexture)
    {
        destroy_vec_float(gpTexture);
        gpTexture = NULL;
    }

    if(gpNormal)
    {
        destroy_vec_float(gpNormal);
        gpNormal = NULL;
    }

    if(gp_sorted_vertices)
    {
        destroy_vec_float(gp_sorted_vertices);
        gp_sorted_vertices = NULL;
    }

    if(gp_sorted_texture)
    {
        destroy_vec_float(gp_sorted_texture);
        gp_sorted_texture = NULL;
    }

    if(gp_sorted_normal)
    {
        destroy_vec_float(gp_sorted_normal);
        gp_sorted_normal = NULL;
    }

    if(gp_indices_vertices)
    {
        destroy_vec_int(gp_indices_vertices);
        gp_indices_vertices = NULL;
    }

    if(gp_indices_texture)
    {
        destroy_vec_int(gp_indices_texture);
        gp_indices_texture = NULL;
    }

    if(gp_indices_normal)
    {
        destroy_vec_int(gp_indices_normal);
        gp_indices_normal = NULL;
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

void oglLoadMesh(void)
{
    //function declarations
    struct vec_int *create_vec_int(void);
    struct vec_float *create_vec_float(void);
    int push_back_vec_int(struct vec_int *p_vec_int, int data);
    int push_back_vec_float(struct vec_float *p_vec_float, float data);
    void show_vec_float(struct vec_float *p_vec_float);
    void show_vec_int(struct vec_int *p_vec_int);
    int destroy_vec_float(struct vec_float *p_vec_float);

    //variable declaration
    char *space = " ";
    char *slash = "/";
    char *first_token = NULL;
    char *token = NULL;
    char *f_enteries[3] = { NULL, NULL, NULL };
    
    int nr_vert_cords = 0;
    int nr_tex_cords = 0;
    int nr_norm_cords = 0;
    int nr_faces = 0;

    int i, vi;
    fprintf(gpFile, "In Load Mesh Function.\n");
    gpMeshFile = fopen("/Users/user158739/Desktop/MyOpenGL_Assignments/Atharva/01-Mac_Assignments/30-ModelLoading/teapot.obj", "r");
    if(gpMeshFile == NULL)
    {
        fprintf(gpFile, "error : unable to open obj file\n");
        exit(EXIT_FAILURE);
    }

    gpVertices     =     create_vec_float();
    gpTexture     =     create_vec_float();
    gpNormal     =    create_vec_float();

    gp_indices_vertices     =    create_vec_int();
    gp_indices_texture        =    create_vec_int();
    gp_indices_normal         =    create_vec_int();

    while(fgets(buffer, BUFFER_SIZE, gpMeshFile) != NULL)
    {
        first_token = strtok(buffer, space);
        if(strcmp(first_token, "v") == 0)
        {
            nr_vert_cords++;
            while((token = strtok(NULL, space)) != NULL)
            {
                push_back_vec_float(gpVertices, atof(token));
            }

        }

        else if(strcmp(first_token, "vt") == 0)
        {
            nr_tex_cords++;
            while((token = strtok(NULL, space)) != NULL)
            {
                push_back_vec_float(gpTexture, atof(token));
            }
            
        }

        else if(strcmp(first_token, "vn") == 0)
        {
            nr_norm_cords++;
            while((token = strtok(NULL, space)) != NULL)
            {
                push_back_vec_float(gpNormal, atof(token));
            }
            
        }

        else if(strcmp(first_token, "f") == 0)
        {
            nr_faces++;
            for(i = 0; i < 3; i++)
            {
                f_enteries[i] = strtok(NULL, space);
            }

            for(i = 0; i < 3; i++)
            {
                token = strtok(f_enteries[i], slash);
                push_back_vec_int(gp_indices_vertices, atoi(token) - 1);

                token = strtok(NULL, slash);
                push_back_vec_int(gp_indices_texture, atoi(token) - 1);

                token = strtok(NULL, slash);
                push_back_vec_int(gp_indices_normal, atoi(token) - 1);
            }
        }
    }

    gp_sorted_vertices = create_vec_float();
    for(int i = 0; i < gp_indices_vertices->size; i++)
    {
        push_back_vec_float(gp_sorted_vertices, gpVertices->pf[i]);
    }

    gp_sorted_texture = create_vec_float();
    for(int i = 0; i < gp_indices_texture->size; i++)
    {
        push_back_vec_float(gp_sorted_texture, gpTexture->pf[i]);
    }

    gp_sorted_normal = create_vec_float();
    for(int i = 0; i < gp_indices_normal->size; i++)
    {
        push_back_vec_float(gp_sorted_normal, gpNormal->pf[i]);
    }
    
    fprintf(gpFile, "Returning From Load Mesh Function.\n");
    fclose(gpMeshFile);
    gpMeshFile = NULL;
}

struct vec_int *create_vec_int(void)
{
    //code
    struct vec_int *p = (struct vec_int *)malloc(sizeof(struct vec_int));
    if(p != NULL)
    {
        memset(p, 0, sizeof(struct vec_int));
        return (p);
    }
    return(NULL);
}

struct vec_float *create_vec_float(void)
{
    //code
    struct vec_float *p = (struct vec_float *)malloc(sizeof(struct vec_float));
    if(p != NULL)
    {
        memset(p, 0, sizeof(struct vec_float));
        return (p);
    }
    return(NULL);
}
int push_back_vec_int(struct vec_int *p_vec_int, int data)
{
    //code
    p_vec_int->p = (int *)realloc(p_vec_int->p, (p_vec_int->size + 1) * sizeof(int));
    p_vec_int->size = p_vec_int->size + 1;
    p_vec_int->p[p_vec_int->size-1] = data;
    return (0);
}
int push_back_vec_float(struct vec_float *p_vec_float, float data)
{
    //code
    p_vec_float->pf = (float *)realloc(p_vec_float->pf, (p_vec_float->size + 1) * sizeof(float));
    p_vec_float->size = p_vec_float->size + 1;
    p_vec_float->pf[p_vec_float->size-1] = data;
    return (0);
}
void show_vec_float(struct vec_float *p_vec_float)
{
    //code
    int i = 0;
    for(i = 0; i < p_vec_float->size; i++)
    {
        fprintf(gpFile, "%f\n", p_vec_float->pf[i]);
    }
}
void show_vec_int(struct vec_int *p_vec_int)
{
    //code
    int i = 0;
    for(i = 0; i < p_vec_int->size; i++)
    {
        fprintf(gpFile, "%d\n", p_vec_int->p[i]);
    }
}
int destroy_vec_float(struct vec_float *p_vec_float)
{
    //code
    free(p_vec_float->pf);
    free(p_vec_float);
    p_vec_float = NULL;
    return (0);
}

int destroy_vec_int(struct vec_int *p_vec_int)
{
    //code
    free(p_vec_int->p);
    free(p_vec_int);
    p_vec_int = NULL;
    return (0);
}



CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink,const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext)
{
    CVReturn result = [(GLView *)pDisplayLinkContext getFrameForTime:pOutputTime];
    return(result);
}
