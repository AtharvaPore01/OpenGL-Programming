#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#import <QuartzCore/CVDisplayLink.h>    //core video display link

//opengl related header
#import <OpenGL/gl3.h>
#import <OpenGL/gl3ext.h>
#import "vmath.h"
#import "sphere.h"

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

struct
{
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
}vertex, fragment;

//spher related variables
float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_texture[764];
short sphere_elements[2280];
unsigned int gNumVertices;
unsigned int gNumElements;

//flags
BOOL bLight = FALSE;
BOOL bPerVertex = TRUE;
BOOL bPerFragment = FALSE;

//light values
float LightAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float LightDiffuse[4] = { 0.5f, 0.2f, 0.7f, 1.0f };                    //albedo
float LightSpecular[4] = { 0.7f, 0.7f, 0.7f, 1.0f };
float LightPosition[4] = { 100.0f, 100.0f, 100.0f, 1.0f };            //{ 1.0f, 1.0f, 1.0f, 1.0f };

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 128.0f;                            //{128.0f};

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
    [window setTitle:@"macOS Window:Per Vertex Per Fragment Toggle"];
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
    GLuint gVertexShaderObject_perVertex;
    GLuint gVertexShaderObject_perFragment;
    GLuint gFragmentShaderObject_perVertex;
    GLuint gFragmentShaderObject_perFragment;
    GLuint gShaderProgramObject_perVertex;
    GLuint gShaderProgramObject_perFragment;
    	
    GLuint vao_sphere;
    GLuint vbo_sphere_position;
    GLuint vbo_sphere_normal;
    GLuint vbo_sphere_element;
    
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
    
    /**** Per Vertex ****/

    /* Vertex Shader Code */

    //define vertex shader object
    gVertexShaderObject_perVertex = glCreateShader(GL_VERTEX_SHADER);

    //write vertex shader code
    const GLchar *vertexShaderSourceCode_perVertex =
        "#version 410" \
        "\n" \
        "in vec4 vPosition;" \
        "in vec3 vNormal;" \
        "uniform mat4 u_model_matrix;" \
        "uniform mat4 u_view_matrix;" \
        "uniform mat4 u_projection_matrix;" \
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
        "void main(void)" \
        "{" \
        "if (u_LKeyPressed == 1)" \
        "{" \
        "vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" \
        "mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" \
        "vec3 transformed_normal = normalize(normal_matrix * vNormal);" \
        "vec3 light_direction = normalize(vec3(u_light_position - eye_coordinates));" \
        "float tn_dot_LightDirection = max(dot(light_direction, transformed_normal), 0.0);" \
        "vec3 reflection_vector = reflect(-light_direction, transformed_normal);" \
        "vec3 viewer_vector = normalize(vec3(-eye_coordinates.xyz));" \

        "vec3 ambient = u_La * u_Ka;" \
        "vec3 diffuse = u_Ld * u_Kd * tn_dot_LightDirection;" \
        "vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, viewer_vector), 0.0), shininess);" \

        "phong_ads_light = ambient + diffuse + specular;" \
        "}" \
        "else" \
        "{" \
        "phong_ads_light = vec3(1.0, 1.0, 1.0);" \
        "}" \
        "gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
        "}";

    //specify above source code to vertex shader object
    glShaderSource(gVertexShaderObject_perVertex, 1, (const GLchar **)&vertexShaderSourceCode_perVertex, NULL);

    //compile the vertex shader
    glCompileShader(gVertexShaderObject_perVertex);

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
    glGetShaderiv(gVertexShaderObject_perVertex, GL_COMPILE_STATUS, &iShaderCompileStatus);

    if (iShaderCompileStatus == GL_FALSE)
    {
        glGetShaderiv(gVertexShaderObject_perVertex, GL_INFO_LOG_LENGTH, &iInfoLogLength);

        if (iInfoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(iInfoLogLength);

            if (szInfoLog != NULL)
            {
                GLsizei Written;
                glGetShaderInfoLog(gVertexShaderObject_perVertex,
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
    gFragmentShaderObject_perVertex = glCreateShader(GL_FRAGMENT_SHADER);

    //write shader code
    const GLchar *fragmentShaderSourceCode_perVertex =
        "#version 410" \
        "\n" \
        "in vec3 phong_ads_light;" \
        "out vec4 FragColor;" \
        "void main(void)" \
        "{" \
        "FragColor = vec4(phong_ads_light, 0.0);" \
        "}";
    //specify above shader code to fragment shader object
    glShaderSource(gFragmentShaderObject_perVertex, 1, (const GLchar **)&fragmentShaderSourceCode_perVertex, NULL);

    //compile the shader
    glCompileShader(gFragmentShaderObject_perVertex);

    //error checking
    iShaderCompileStatus = 0;
    iInfoLogLength = 0;
    szInfoLog = NULL;

    glGetShaderiv(gFragmentShaderObject_perVertex, GL_COMPILE_STATUS, &iShaderCompileStatus);

    if (iShaderCompileStatus == GL_FALSE)
    {
        glGetShaderiv(gFragmentShaderObject_perVertex, GL_INFO_LOG_LENGTH, &iInfoLogLength);

        if (iInfoLogLength > 0)
        {

            szInfoLog = (GLchar *)malloc(iInfoLogLength);
            if (szInfoLog != NULL)
            {
                GLsizei Written;
                glGetShaderInfoLog(gFragmentShaderObject_perVertex,
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
    gShaderProgramObject_perVertex = glCreateProgram();

    //Attach Vertex Shader
    glAttachShader(gShaderProgramObject_perVertex, gVertexShaderObject_perVertex);

    //Attach Fragment Shader
    glAttachShader(gShaderProgramObject_perVertex, gFragmentShaderObject_perVertex);

    //pre linking bonding to vertex attributes
    glBindAttribLocation(gShaderProgramObject_perVertex, AMC_ATTRIBUTE_POSITION, "vPosition");
    glBindAttribLocation(gShaderProgramObject_perVertex, AMC_ATTRIBUTE_NORMAL, "vNormal");

    //link the shader porgram
    glLinkProgram(gShaderProgramObject_perVertex);

    //error checking

    iInfoLogLength = 0;
    szInfoLog = NULL;

    glGetProgramiv(gShaderProgramObject_perVertex, GL_LINK_STATUS, &iProgramLinkStatus);

    if (iProgramLinkStatus == GL_FALSE)
    {
        glGetProgramiv(gShaderProgramObject_perVertex, GL_INFO_LOG_LENGTH, &iInfoLogLength);

        if (iInfoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(iInfoLogLength);

            if (szInfoLog != NULL)
            {
                GLsizei Written;
                glGetProgramInfoLog(gShaderProgramObject_perVertex, iInfoLogLength, &Written, szInfoLog);
                fprintf(gpFile, "Program Link Error : \n %s\n", szInfoLog);
                free(szInfoLog);
                [self release];
                [NSApp terminate:self];
            }
        }
    }

    //post linking retriving uniform location
    vertex.LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_LKeyPressed");

    vertex.model_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_model_matrix");
    vertex.view_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_view_matrix");
    vertex.projection_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_projection_matrix");

    vertex.La_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_La");
    vertex.Ld_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ld");
    vertex.Ls_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ls");
    vertex.Ka_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ka");
    vertex.Kd_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Kd");
    vertex.Ks_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_Ks");
    vertex.shininess_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "shininess");
    vertex.lightPosition_uniform = glGetUniformLocation(gShaderProgramObject_perVertex, "u_light_position");
    /**** Per Fragment ****/

    /* Vertex Shader Code */

    //define vertex shader object
    gVertexShaderObject_perFragment = glCreateShader(GL_VERTEX_SHADER);

    //write vertex shader code
    const GLchar *vertexShaderSourceCode_perFragment =
        "#version 410" \
        "\n" \
        "in vec4 vPosition;" \
        "in vec3 vNormal;" \
        "uniform mat4 u_model_matrix;" \
        "uniform mat4 u_view_matrix;" \
        "uniform mat4 u_projection_matrix;" \
        "uniform mat4 u_mvp_matrix;" \
        "uniform int u_LKeyPressed;" \
        "uniform vec4 u_light_position;" \
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
        "}";

    //specify above source code to vertex shader object
    glShaderSource(gVertexShaderObject_perFragment, 1, (const GLchar **)&vertexShaderSourceCode_perFragment, NULL);

    //compile the vertex shader
    glCompileShader(gVertexShaderObject_perFragment);

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
    glGetShaderiv(gVertexShaderObject_perFragment, GL_COMPILE_STATUS, &iShaderCompileStatus);

    if (iShaderCompileStatus == GL_FALSE)
    {
        glGetShaderiv(gVertexShaderObject_perFragment, GL_INFO_LOG_LENGTH, &iInfoLogLength);

        if (iInfoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(iInfoLogLength);

            if (szInfoLog != NULL)
            {
                GLsizei Written;
                glGetShaderInfoLog(gVertexShaderObject_perFragment,
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
    gFragmentShaderObject_perFragment = glCreateShader(GL_FRAGMENT_SHADER);

    //write shader code
    const GLchar *fragmentShaderSourceCode_perFragment =
        "#version 410" \
        "\n" \
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
                "phong_ads_light = ambient + diffuse + specular;" \
            "}" \
            "else" \
            "{" \
                "phong_ads_light = vec3(1.0, 1.0, 1.0);" \
            "}" \
            "FragColor = vec4(phong_ads_light, 0.0);" \
        "}";
    //specify above shader code to fragment shader object
    glShaderSource(gFragmentShaderObject_perFragment, 1, (const GLchar **)&fragmentShaderSourceCode_perFragment, NULL);

    //compile the shader
    glCompileShader(gFragmentShaderObject_perFragment);

    //error checking
    iShaderCompileStatus = 0;
    iInfoLogLength = 0;
    szInfoLog = NULL;

    glGetShaderiv(gFragmentShaderObject_perFragment, GL_COMPILE_STATUS, &iShaderCompileStatus);

    if (iShaderCompileStatus == GL_FALSE)
    {
        glGetShaderiv(gFragmentShaderObject_perFragment, GL_INFO_LOG_LENGTH, &iInfoLogLength);

        if (iInfoLogLength > 0)
        {

            szInfoLog = (GLchar *)malloc(iInfoLogLength);
            if (szInfoLog != NULL)
            {
                GLsizei Written;
                glGetShaderInfoLog(gFragmentShaderObject_perFragment,
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
    gShaderProgramObject_perFragment = glCreateProgram();

    //Attach Vertex Shader
    glAttachShader(gShaderProgramObject_perFragment, gVertexShaderObject_perFragment);

    //Attach Fragment Shader
    glAttachShader(gShaderProgramObject_perFragment, gFragmentShaderObject_perFragment);

    //pre linking bonding to vertex attributes
    glBindAttribLocation(gShaderProgramObject_perFragment, AMC_ATTRIBUTE_POSITION, "vPosition");
    glBindAttribLocation(gShaderProgramObject_perFragment, AMC_ATTRIBUTE_NORMAL, "vNormal");

    //link the shader porgram
    glLinkProgram(gShaderProgramObject_perFragment);

    //error checking

    iInfoLogLength = 0;
    szInfoLog = NULL;

    glGetProgramiv(gShaderProgramObject_perFragment, GL_LINK_STATUS, &iProgramLinkStatus);

    if (iProgramLinkStatus == GL_FALSE)
    {
        glGetProgramiv(gShaderProgramObject_perFragment, GL_INFO_LOG_LENGTH, &iInfoLogLength);

        if (iInfoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(iInfoLogLength);

            if (szInfoLog != NULL)
            {
                GLsizei Written;
                glGetProgramInfoLog(gShaderProgramObject_perFragment, iInfoLogLength, &Written, szInfoLog);
                fprintf(gpFile, "Program Link Error : \n %s\n", szInfoLog);
                free(szInfoLog);
                [self release];
                [NSApp terminate:self];
            }
        }
    }

    //post linking retriving uniform location
    fragment.LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_LKeyPressed");

    fragment.model_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_model_matrix");
    fragment.view_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_view_matrix");
    fragment.projection_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_projection_matrix");

    fragment.La_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_La");
    fragment.Ld_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ld");
    fragment.Ls_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ls");
    fragment.Ka_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ka");
    fragment.Kd_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Kd");
    fragment.Ks_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_Ks");
    fragment.shininess_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "shininess");
    fragment.lightPosition_uniform = glGetUniformLocation(gShaderProgramObject_perFragment, "u_light_position");
    
    //sphere vertices
    getSphereVertexData(sphere_vertices, sphere_normals, sphere_texture, sphere_elements);
    gNumVertices = getNumberOfSphereVertices();
    gNumElements = getNumberOfSphereElements();

    glGenVertexArrays(1, &vao_sphere);
    glBindVertexArray(vao_sphere);
    //position
    glGenBuffers(1, &vbo_sphere_position);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_position);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_vertices), sphere_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //normal
    glGenBuffers(1, &vbo_sphere_normal);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_normal);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_normals), sphere_normals, GL_STATIC_DRAW);
    glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //elements
    glGenBuffers(1, &vbo_sphere_element);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements), sphere_elements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
    
    //depth
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

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
    
    //declaration of metrices
    vmath::mat4 modelMatrix_perVertex;
    vmath::mat4 modelMatrix_perFragment;
    vmath::mat4 viewMatrix_perVertex;
    vmath::mat4 viewMatrix_perFragment;
    vmath::mat4 projectionMatrix_perVertex;
    vmath::mat4 projectionMatrix_perFragment;
    vmath::mat4 translationMatrix_perVertex;
    vmath::mat4 translationMatrix_perFragment;

    //init above metrices to identity
    modelMatrix_perVertex = vmath::mat4::identity();
    viewMatrix_perVertex = vmath::mat4::identity();
    projectionMatrix_perVertex = vmath::mat4::identity();
    translationMatrix_perVertex = vmath::mat4::identity();

    //do necessary transformations here
    translationMatrix_perVertex = vmath::translate(0.0f, 0.0f, -3.0f);

    //do necessary matrix multiplication
    modelMatrix_perVertex = modelMatrix_perVertex * translationMatrix_perVertex;
    projectionMatrix_perVertex *= perspectiveProjectionMatrix;

    //init above metrices to identity
    modelMatrix_perFragment = vmath::mat4::identity();
    viewMatrix_perFragment = vmath::mat4::identity();
    projectionMatrix_perFragment = vmath::mat4::identity();
    translationMatrix_perFragment = vmath::mat4::identity();

    //do necessary transformations here
    translationMatrix_perFragment = vmath::translate(0.0f, 0.0f, -3.0f);

    //do necessary matrix multiplication
    modelMatrix_perFragment = modelMatrix_perFragment * translationMatrix_perFragment;
    projectionMatrix_perFragment *= perspectiveProjectionMatrix;

    if (bPerVertex)
    {
        glUseProgram(gShaderProgramObject_perVertex);

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(vertex.model_uniform, 1, GL_FALSE, modelMatrix_perVertex);
        glUniformMatrix4fv(vertex.view_uniform, 1, GL_FALSE, viewMatrix_perVertex);
        glUniformMatrix4fv(vertex.projection_uniform, 1, GL_FALSE, projectionMatrix_perVertex);

        //if light is enabled
        if (bLight)
        {
            //notify shader that we pressed the "L" key
            glUniform1i(vertex.LKeyPressed_Uniform, 1);
            //send light intensity
            glUniform3fv(vertex.La_uniform, 1, LightAmbient);
            glUniform3fv(vertex.Ld_uniform, 1, LightDiffuse);
            glUniform3fv(vertex.Ls_uniform, 1, LightSpecular);
            //send coeff. of material's reflectivity
            glUniform3fv(vertex.Ka_uniform, 1, MaterialAmbient);
            glUniform3fv(vertex.Kd_uniform, 1, MaterialDiffuse);
            glUniform3fv(vertex.Ks_uniform, 1, MaterialSpecular);
            //shininess
            glUniform1f(vertex.shininess_uniform, MaterialShininess);
            //send light position
            glUniform4fv(vertex.lightPosition_uniform, 1, LightPosition);
        }
        else
        {
            //notify shader that we aren't pressed the "L" key
            glUniform1i(vertex.LKeyPressed_Uniform, 0);
        }
    }
    
    if (bPerFragment)
    {
        glUseProgram(gShaderProgramObject_perFragment);
        
        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(fragment.model_uniform, 1, GL_FALSE, modelMatrix_perFragment);
        glUniformMatrix4fv(fragment.view_uniform, 1, GL_FALSE, viewMatrix_perFragment);
        glUniformMatrix4fv(fragment.projection_uniform, 1, GL_FALSE, projectionMatrix_perFragment);

        //if light is enabled
        if (bLight)
        {
            //notify shader that we pressed the "L" key
            glUniform1i(fragment.LKeyPressed_Uniform, 1);
            //send light intensityx
            glUniform3fv(fragment.La_uniform, 1, LightAmbient);
            glUniform3fv(fragment.Ld_uniform, 1, LightDiffuse);
            glUniform3fv(fragment.Ls_uniform, 1, LightSpecular);
            //send coeff. of material's reflectivity
            glUniform3fv(fragment.Ka_uniform, 1, MaterialAmbient);
            glUniform3fv(fragment.Kd_uniform, 1, MaterialDiffuse);
            glUniform3fv(fragment.Ks_uniform, 1, MaterialSpecular);
            //shininess
            glUniform1f(fragment.shininess_uniform, MaterialShininess);
            //send light position
            glUniform4fv(fragment.lightPosition_uniform, 1, LightPosition);
        }

        else
        {
            //notify shader that we aren't pressed the "L" key
            glUniform1i(fragment.LKeyPressed_Uniform, 0);
        }
    }

    if (bPerVertex == TRUE || bPerFragment == TRUE)
    {
        //bind with vao
        glBindVertexArray(vao_sphere);

        //draw scene
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element);
        glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);

        //unbind vao
        glBindVertexArray(0);
    }
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
            [[self window]toggleFullScreen:self];    //repainting occures automatically
            
            break;
        case 'Q':
        case 'q':
            [self release];
            [NSApp terminate:self];
            break;
            
        case 'F':
        case 'f':
            if (bPerFragment == FALSE)
            {
                //MessageBox(NULL, TEXT("bPerFragment Flag True"), TEXT("bPerFragment Flag Message."), MB_OK);
                bPerFragment = TRUE;
                if (bPerVertex == TRUE)
                {
                    //MessageBox(NULL, TEXT("bPerVertex Flag False"), TEXT("bPerFragment Flag Message."), MB_OK);
                    bPerVertex = FALSE;
                }
            }
            else
            {
                //MessageBox(NULL, TEXT("bPerFragment Flag False"), TEXT("bPerFragment Flag Message."), MB_OK);
                bPerFragment = FALSE;
            }
            break;
            
        case 'L':
        case 'l':
            if (bLight == FALSE)
            {
                //MessageBox(NULL, TEXT("Light Flag True"), TEXT("bLight Flag Message."), MB_OK);
                bLight = TRUE;

            }
            else
            {
                //MessageBox(NULL, TEXT("Light Flag False"), TEXT("bLight Flag Message."), MB_OK);
                bLight = FALSE;
            }
            break;

        case 'V':
        case 'v':
            if (bPerVertex == FALSE)
            {
                //MessageBox(NULL, TEXT("bPerVertex Flag True"), TEXT("bPerVertex Flag Message."), MB_OK);
                bPerVertex = TRUE;
                if (bPerFragment == TRUE)
                {
                    //MessageBox(NULL, TEXT("bPerFragment Flag False"), TEXT("bPerVertex Flag Message."), MB_OK);
                    bPerFragment = FALSE;
                }
            }
            else
            {
                //MessageBox(NULL, TEXT("bPerVertex Flag False"), TEXT("bPerVertex Flag Message."), MB_OK);
                bPerVertex = FALSE;
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
    //code
    
    if (vbo_sphere_element)
    {
        glDeleteBuffers(1, &vbo_sphere_element);
        vbo_sphere_element = 0;
    }
    if (vbo_sphere_normal)
    {
        glDeleteBuffers(1, &vbo_sphere_normal);
        vbo_sphere_normal = 0;
    }
    if (vbo_sphere_position)
    {
        glDeleteBuffers(1, &vbo_sphere_position);
        vbo_sphere_position = 0;
    }
    if (vao_sphere)
    {
        glDeleteVertexArrays(1, &vao_sphere);
        vao_sphere = 0;
    }

    //safe release

    if (gShaderProgramObject_perVertex)
    {
        GLsizei shaderCount;
        GLsizei shaderNumber;

        glUseProgram(gShaderProgramObject_perVertex);

        //ask program how many shaders are attached
        glGetProgramiv(gShaderProgramObject_perVertex, GL_ATTACHED_SHADERS, &shaderCount);

        GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

        if (pShaders)
        {
            glGetAttachedShaders(gShaderProgramObject_perVertex, shaderCount, &shaderCount, pShaders);

            for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
            {
                //detach shader
                glDetachShader(gShaderProgramObject_perVertex, pShaders[shaderNumber]);
                //delete shader
                glDeleteShader(pShaders[shaderNumber]);
                pShaders[shaderNumber] = 0;
            }
            free(pShaders);
        }
        glDeleteProgram(gShaderProgramObject_perVertex);
        gShaderProgramObject_perVertex = 0;
        glUseProgram(0);
    }

    if (gShaderProgramObject_perFragment)
    {
        GLsizei shaderCount;
        GLsizei shaderNumber;

        glUseProgram(gShaderProgramObject_perFragment);

        //ask program how many shaders are attached
        glGetProgramiv(gShaderProgramObject_perFragment, GL_ATTACHED_SHADERS, &shaderCount);

        GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

        if (pShaders)
        {
            glGetAttachedShaders(gShaderProgramObject_perFragment, shaderCount, &shaderCount, pShaders);

            for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
            {
                //detach shader
                glDetachShader(gShaderProgramObject_perFragment, pShaders[shaderNumber]);
                //delete shader
                glDeleteShader(pShaders[shaderNumber]);
                pShaders[shaderNumber] = 0;
            }
            free(pShaders);
        }
        glDeleteProgram(gShaderProgramObject_perFragment);
        gShaderProgramObject_perFragment = 0;
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
