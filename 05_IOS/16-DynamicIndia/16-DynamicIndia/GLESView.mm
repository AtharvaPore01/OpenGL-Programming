#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>

#import "vmath.h"

#import "GLESView.h"

enum
{
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_NORMAL,
    AMC_ATTRIBUTE_TEXCOODR_0
};

GLuint vao_I;
GLuint vao_N;
GLuint vao_D;
GLuint vao_i;
GLuint vao_A;
GLuint vao_middleStrips;
GLuint vao_plane;

GLuint vbo_I_position;
GLuint vbo_I_color;
GLuint vbo_N_position;
GLuint vbo_N_color;
GLuint vbo_D_position;
GLuint vbo_D_color;
GLuint vbo_i_position;
GLuint vbo_i_color;
GLuint vbo_A_position;
GLuint vbo_A_color;
GLuint vbo_middleStrips_position;
GLuint vbo_middleStrips_color;
GLuint vbo_plane_position;
GLuint vbo_plane_color;

//flags
bool b_I_Done = false;
bool b_N_Done = false;
bool b_D_Done = false;
bool b_i_Done = false;
bool b_A_Done = false;
bool b_clip_top_plane = false;
bool b_clip_bottom_plane = false;
bool b_unclip_top_plane = false;
bool b_unclip_bottom_plane = false;

bool b_appear_middle_strip = true;

bool b_top_plane_smoke_done = false;
bool b_bottom_plane_smoke_done = false;
bool b_middle_plane_smoke_done = false;

bool b_start_decrementing = false;
bool b_start_incrementing = false;
bool b_PlaneTrue = false;


//variables for translation of I,N,D,i,A
GLfloat f_Translate_I = -3.0f;        //translate along x
GLfloat f_Translate_N = 3.0f;        //translate along y
GLfloat f_Translate_D = 0.0f;
GLfloat f_Translate_i = -3.0f;        //translate along y
GLfloat f_Translate_A = 3.0f;        //translate along x

//color values for D
GLfloat f_DRedColor = 0.0f;
GLfloat f_DGreenColor = 0.0f;
GLfloat f_DBlueColor = 0.0f;

//A middle strips colors
GLfloat f_ARedColor = 0.0f;
GLfloat f_AGreenColor = 0.0f;
GLfloat f_ABlueColor = 0.0f;
GLfloat f_AWhiteColor = 0.0f;

//smoke colors
GLfloat f_red = 1.0f;
GLfloat f_green = 0.5f;
GLfloat f_blue = 1.0f;
GLfloat f_white = 1.0f;

/* angles to draw a smoke */
GLfloat top_angle_1 = 3.14159f;
GLfloat top_angle_2 = 3.14659f;

GLfloat top_angle_3 = 4.71238f;
GLfloat top_angle_4 = 4.71738f;

GLfloat bottom_angle_1 = 3.13659f;
GLfloat bottom_angle_2 = 3.14159f;

GLfloat bottom_angle_3 = 1.57079f;
GLfloat bottom_angle_4 = 1.56579f;

//plane initial position variable
GLfloat x_plane_pos = 0.0f;
GLfloat y_plane_pos = 0.0f;

GLfloat middle_plane_smoke = 0.0f;

//struct
struct plane
{
    GLfloat _x;
    GLfloat _y;
    GLfloat radius = 10.0f;
    GLfloat angle = (GLfloat)M_PI;
    GLfloat rotation_angle = 0.0f;
}top, bottom;

@implementation GLESView
{
    EAGLContext *eaglContext_ap;
    
    GLuint defaultFramebuffer;
    GLuint colorRenderbuffer;
    GLuint depthRenderbuffer;
    
    id displayLink;
    NSInteger animationFrameInterval;
    BOOL isAnimating;
    
    GLuint gVertexShaderObject;
    GLuint gFragmentShaderObject;
    GLuint gShaderProgramObject;
    
   

    GLuint mvpUniform;
    vmath::mat4 perspectiveProjectionMatrix;
    
}

- (void)oglInitData
{
    //code
    top.rotation_angle = -60.0f;
    bottom.rotation_angle = 60.0f;
    x_plane_pos = -22.0f;        //plane starting position
}

- (id)initWithFrame:(CGRect)frameRect
{
    //variables
    GLint iShaderCompileStatus = 0;
    GLint iProgramLinkStatus = 0;
    GLint iInfoLogLength = 0;
    GLchar *szInfoLog = NULL;
    
    //code
    self=[super initWithFrame:frameRect];
    if(self)
    {
        //initialise code here
        
        CAEAGLLayer *eaglLayer=(CAEAGLLayer *)super.layer;
        
        eaglLayer.opaque = YES;
        eaglLayer.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys:
                                        [NSNumber numberWithBool:FALSE],
                                        kEAGLDrawablePropertyRetainedBacking,
                                        kEAGLColorFormatRGBA8,
                                        kEAGLDrawablePropertyColorFormat,
                                        nil];
        
        eaglContext_ap = [[EAGLContext alloc]initWithAPI:kEAGLRenderingAPIOpenGLES3];
        if(eaglContext_ap==nil)
        {
            [self release];
            return(nil);
        }
        [EAGLContext setCurrentContext:eaglContext_ap];
        
        glGenFramebuffers(1, &defaultFramebuffer);
        glGenRenderbuffers(1, &colorRenderbuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
        
        [eaglContext_ap renderbufferStorage:GL_RENDERBUFFER fromDrawable:eaglLayer];
        
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRenderbuffer);
        
        GLint backingWidth;
        GLint backingHeight;
        
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &backingWidth);
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &backingHeight);
        
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
        
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            printf("Failed To Create Complete Framebuffer Object %x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
            
            glDeleteFramebuffers(1, &defaultFramebuffer);
            glDeleteRenderbuffers(1, &colorRenderbuffer);
            glDeleteRenderbuffers(1, &depthRenderbuffer);
            
            return(nil);
        }
        
        printf("Renderer : %s | GL version : %s | GLSL version : %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));
        
        //hard coded initialization
        isAnimating = NO;
        animationFrameInterval = 60;    //default since iOS 8.2
        
        /* Vertex Shader */
        //define vertex shader object
        gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
        
        //write vertex shader code
        const GLchar *vertexShaderSourceCode =
        "#version 300 es" \
        "\n" \
        "in vec4 vPosition;" \
        "in vec4 vColor;"
        "out vec4 out_color;"
        "uniform mat4 u_mvp_matrix;" \
        "void main(void)" \
        "{" \
        "gl_Position = u_mvp_matrix * vPosition;" \
        "out_color = vColor;"
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
                    
                    printf("Vertex Shader Error : \n %s \n", szInfoLog);
                    free(szInfoLog);
                    [self release];
                }
            }
        }
        /* Fragment Shader Code */
        
        //define fragment shader object
        gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
        
        //write shader code
        const GLchar *fragmentShaderSourceCode =
        "#version 300 es" \
        "\n" \
        "precision highp float;" \
        "out vec4 FragColor;" \
        "in vec4 out_color;"
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
                    printf("Fragment Shader Error : \n %s \n", szInfoLog);
                    free(szInfoLog);
                    [self release];
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
                    printf("Program Link Error : \n %s\n", szInfoLog);
                    free(szInfoLog);
                    [self release];
                }
            }
        }
        
        //post linking retriving uniform location
        mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");
        
        //vertices declaration
        const GLfloat I_vertices[] =
        {
            -1.15f, 0.7f, 0.0f,
            -1.25f, 0.7f, 0.0f,
            -1.25f, -0.7f, 0.0f,
            -1.15f, -0.7f, 0.0f
        };
        
        const GLfloat N_vertices[] =
        {
            -0.95f, 0.7f, 0.0f,
            -1.05f, 0.7f, 0.0f,
            -1.05f, -0.7f, 0.0f,
            -0.95f, -0.7f, 0.0f,
            
            -0.55f, 0.7f, 0.0f,
            -0.65f, 0.7f, 0.0f,
            -0.65f, -0.7f, 0.0f,
            -0.55f, -0.7f, 0.0f,
            
            -0.95f, 0.7f, 0.0f,
            -0.95f, 0.5f, 0.0f,
            -0.65f, -0.7f, 0.0f,
            -0.65f, -0.5f, 0.0f
        };
        
        const GLfloat D_vertices[] =
        {
            //top
            0.15f, 0.7f, 0.0f,
            -0.45f, 0.7f, 0.0f,
            -0.45f, 0.6f, 0.0f,
            0.15f, 0.6f, 0.0f,
            
            //bottom
            0.15f, -0.7f, 0.0f,
            -0.45f, -0.7f, 0.0f,
            -0.45f, -0.6f, 0.0f,
            0.15f, -0.6f, 0.0f,
            
            //left
            0.15f, 0.7f, 0.0f,
            0.05f, 0.7f, 0.0f,
            0.05f, -0.7f, 0.0f,
            0.15f, -0.7f, 0.0f,
            
            //right
            -0.25f, 0.6f, 0.0f,
            -0.35f, 0.6f, 0.0f,
            -0.35f, -0.6f, 0.0f,
            -0.25f, -0.6f, 0.0f
        };
        
        const GLfloat i_vertices[] =
        {
            0.35f, 0.7f, 0.0f,
            0.25f, 0.7f, 0.0f,
            0.25f, -0.7f, 0.0f,
            0.35f, -0.7f, 0.0f
        };
        
        const GLfloat A_vertices[] =
        {
            //left
            0.75f, 0.7f, 0.0f,
            0.75f, 0.5f, 0.0f,
            0.55f, -0.7f, 0.0f,
            0.45f, -0.7f, 0.0f,
            //right
            0.75f, 0.7f, 0.0f,
            0.75f, 0.5f, 0.0f,
            0.95f, -0.7f, 0.0f,
            1.05f, -0.7f, 0.0f,
        };
        
        const GLfloat planeVertices[] =
        {
            /* Vertices */
            //body
            2.0f, 0.35f, 0.0f,                                    //0
            -1.0f, 0.3f, 0.0f,                                    //1
            -1.0f, -0.3f, 0.0f,                                    //2
            2.0f, -0.35f, 0.0f,                                    //3

            //exahaust
            -0.3f, 0.0f, 0.0f,                                    //4
            -1.2f, 0.4f, 0.0f,                                    //5
            -1.2f, -0.4f, 0.0f,                                    //6

            //orange
            -1.2f, 0.3f, 0.0f,
            -2.5f, 0.3f, 0.0f,
            -2.5f, 0.1f, 0.0f,
            -1.2f, 0.1f, 0.0f,

            //white
            -1.2f, 0.1f, 0.0f,
            -2.5f, 0.1f, 0.0f,
            -2.5f, -0.1f, 0.0f,
            -1.2f, -0.1f, 0.0f,

            //green
            -1.2f, -0.1f, 0.0f,
            -2.5f, -0.1f, 0.0f,
            -2.5f, -0.3f, 0.0f,
            -1.2f, -0.3f, 0.0f,

            //separator line between exhaust and body
            -1.0f, 0.15f, 0.0f,                                    //7
            -1.0f, -0.15f, 0.0f,                                //8

            //front tip
            2.8f, 0.0f, 0.0f,                                    //9
            2.0f, 0.35f, 0.0f,                                    //10
            2.0f, -0.35f, 0.0f,                                    //11

            //sperator line between front tip and body
            2.0f, 0.35f, 0.0f,                                    //12
            2.0f, -0.35f, 0.0f,                                    //13

            //upper wing
            1.5f, 0.32f, 0.0f,                                    //14
            -0.6f, 1.5f, 0.0f,                                    //15
            -0.6f, 0.22f, 0.0f,                                    //16

            //lower wing
            1.5f, -0.32f, 0.0f,                                    //17
            -0.6f, -1.5f, 0.0f,                                    //18
            -0.6f, -0.22f, 0.0f,                                //19

            //IAF Letters
            /* Vertices */
            //1. I
            -0.0f, 0.15f, 0.0f,                                    //20
            -0.0f, -0.15f, 0.0f,                                //21

            //2. A
            0.2f, 0.15f, 0.0f,                                    //22
            0.1f, -0.15f, 0.0f,                                    //23

            0.2f, 0.15f, 0.0f,                                    //24
            0.3f, -0.15f, 0.0f,                                    //25

            0.15f, 0.0f, 0.0f,                                    //26
            0.25f, 0.0f, 0.0f,                                    //27

            //3. F
            0.4f, 0.15f, 0.0f,                                    //28
            0.4f, -0.15f, 0.0f,                                    //29

            0.4f, 0.15f, 0.0f,                                    //30
            0.55f, 0.15f, 0.0f,                                    //31

            0.4f, 0.0f, 0.0f,                                    //32
            0.5f, 0.0f, 0.0f                                    //33
        };

        const GLfloat middleStrips_vertices[] =
        {
            //orange
            -1.2f, 0.3f, 0.0f,
            -2.5f, 0.3f, 0.0f,
            -2.5f, 0.1f, 0.0f,
            -1.2f, 0.1f, 0.0f,

            //white
            -1.2f, 0.1f, 0.0f,
            -2.5f, 0.1f, 0.0f,
            -2.5f, -0.1f, 0.0f,
            -1.2f, -0.1f, 0.0f,

            //green
            -1.2f, -0.1f, 0.0f,
            -2.5f, -0.1f, 0.0f,
            -2.5f, -0.3f, 0.0f,
            -1.2f, -0.3f, 0.0f,
        };

        
        //color declaration
        const GLfloat I_color[] =
        {
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f
        };
        
        const GLfloat N_color[] =
        {
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
        };
        
        const GLfloat D_color[] =
        {
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
        };
        
        const GLfloat i_color[] =
        {
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
        };
        
        const GLfloat A_color[] =
        {
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f,
        };
        
         const GLfloat planeColor[] =
           {
               
               /* color */
               //body
               0.7294117f, 0.8862745f, 0.9333333f,                                    //0
               0.7294117f, 0.8862745f, 0.9333333f,                                    //1
               0.7294117f, 0.8862745f, 0.9333333f,                                    //2
               0.7294117f, 0.8862745f, 0.9333333f,                                    //3

               //exahaust
               0.7294117f, 0.8862745f, 0.9333333f,                                    //4
               0.7294117f, 0.8862745f, 0.9333333f,                                    //5
               0.7294117f, 0.8862745f, 0.9333333f,                                    //6

               //orange
               1.0f, 0.5f, 0.0f,    //right top                                        //7
               0.0f, 0.0f, 0.0f,    //right bottom                                    //8
               0.0f, 0.0f, 0.0f,    //left top                                        //9
               1.0f, 0.5f, 0.0f,    //left top                                        //10

               //white
               1.0f, 1.0f, 1.0f,    //right top                            //11
               0.0f, 0.0f, 0.0f,    //right bottom        //12
               0.0f, 0.0f, 0.0f,    //left top            13
               1.0f, 1.0f, 1.0f,    //left top            14

               //green
               0.0f, 0.5f, 0.0f,    //right top            15
               0.0f, 0.0f, 0.0f,    //right bottom        16
               0.0f, 0.0f, 0.0f,    //left top            17
               0.0f, 0.5f, 0.0f,    //left top            18


               //separator line between exhaust and body
               0.0f, 0.0f, 0.0f,                                                    //19
               0.0f, 0.0f, 0.0f,                                                    //20

               //front tip
               0.7294117f, 0.8862745f, 0.9333333f,                                    //21
               0.7294117f, 0.8862745f, 0.9333333f,                                    //22
               0.7294117f, 0.8862745f, 0.9333333f,                                    //23

               //sperator line between front tip and body
               0.0f, 0.0f, 0.0f,                                                    //24
               0.0f, 0.0f, 0.0f,                                                    //25

               //upper wing
               0.7294117f, 0.8862745f, 0.9333333f,                                    //26
               0.7294117f, 0.8862745f, 0.9333333f,                                    //27
               0.7294117f, 0.8862745f, 0.9333333f,                                    //28

               //lower wing
               0.7294117f, 0.8862745f, 0.9333333f,                                    //29
               0.7294117f, 0.8862745f, 0.9333333f,                                    //30
               0.7294117f, 0.8862745f, 0.9333333f,                                    //31

               //IAF Letters
               /* color */
               //1. I
               0.0f, 0.0f, 0.0f,                                                    //32
               0.0f, 0.0f, 0.0f,                                                    //33

               //2. A
               0.0f, 0.0f, 0.0f,                                                    //34
               0.0f, 0.0f, 0.0f,                                                    //35

               0.0f, 0.0f, 0.0f,                                                    //36
               0.0f, 0.0f, 0.0f,                                                    //37

               0.0f, 0.0f, 0.0f,                                                    //38
               0.0f, 0.0f, 0.0f,                                                    //39

               //3. F
               0.0f, 0.0f, 0.0f,                                                    //40
               0.0f, 0.0f, 0.0f,                                                    //41

               0.0f, 0.0f, 0.0f,                                                    //42
               0.0f, 0.0f, 0.0f,                                                    //43

               0.0f, 0.0f, 0.0f,                                                    //44
               0.0f, 0.0f, 0.0f,                                                    //45
           };
        
        //create vao and vbo
        
        //I
        glGenVertexArrays(1, &vao_I);
        glBindVertexArray(vao_I);
        
        //vertices
        glGenBuffers(1, &vbo_I_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_I_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(I_vertices), I_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //color
        glGenBuffers(1, &vbo_I_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_I_color);
        glBufferData(GL_ARRAY_BUFFER, sizeof(I_color), I_color, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        //N
        glGenVertexArrays(1, &vao_N);
        glBindVertexArray(vao_N);
        
        //vertices
        glGenBuffers(1, &vbo_N_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_N_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(N_vertices), N_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //color
        glGenBuffers(1, &vbo_N_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_N_color);
        glBufferData(GL_ARRAY_BUFFER, sizeof(N_color), N_color, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        //D
        glGenVertexArrays(1, &vao_D);
        glBindVertexArray(vao_D);
        
        //vertices
        glGenBuffers(1, &vbo_D_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_D_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(D_vertices), D_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //color
        glGenBuffers(1, &vbo_D_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_D_color);
        glBufferData(GL_ARRAY_BUFFER, sizeof(D_color), D_color, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        //i
        glGenVertexArrays(1, &vao_i);
        glBindVertexArray(vao_i);
        
        //vertices
        glGenBuffers(1, &vbo_i_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_i_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(i_vertices), i_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //color
        glGenBuffers(1, &vbo_i_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_i_color);
        glBufferData(GL_ARRAY_BUFFER, sizeof(i_color), i_color, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        //A
        glGenVertexArrays(1, &vao_A);
        glBindVertexArray(vao_A);
        
        //vertices
        glGenBuffers(1, &vbo_A_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_A_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(A_vertices), A_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //color
        glGenBuffers(1, &vbo_A_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_A_color);
        glBufferData(GL_ARRAY_BUFFER, sizeof(A_color), A_color, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        //middleStrips
        glGenVertexArrays(1, &vao_middleStrips);
        glBindVertexArray(vao_middleStrips);

        //vertices
        glGenBuffers(1, &vbo_middleStrips_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_middleStrips_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(middleStrips_vertices), middleStrips_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //color
        glGenBuffers(1, &vbo_middleStrips_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_middleStrips_color);
        glBufferData(GL_ARRAY_BUFFER, 12 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);

        //plane
        glGenVertexArrays(1, &vao_plane);
        glBindVertexArray(vao_plane);

        //vertices
        glGenBuffers(1, &vbo_plane_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_plane_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //color
        glGenBuffers(1, &vbo_plane_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_plane_color);
        glBufferData(GL_ARRAY_BUFFER, sizeof(planeColor), planeColor, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);

        [self oglInitData];
        
        //depth
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        
        //make orthograhic projection matrix a identity matrix
        perspectiveProjectionMatrix = vmath::mat4::identity();
        
        //clear color
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        
        //GESTURE RECOGNITION
        //Tap Gesture Code
        UITapGestureRecognizer *singleTapGestureRecognizer=[[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onSingleTap:)];
        [singleTapGestureRecognizer setNumberOfTapsRequired:1];
        [singleTapGestureRecognizer setNumberOfTouchesRequired:1];  //touch of 1 finger
        [singleTapGestureRecognizer setDelegate:self];
        [self addGestureRecognizer:singleTapGestureRecognizer];
        
        UITapGestureRecognizer *doubleTapGestureRecognizer=
        [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onDoubleTap:)];
        [doubleTapGestureRecognizer setNumberOfTapsRequired:2];
        [doubleTapGestureRecognizer setNumberOfTouchesRequired:1];
        [doubleTapGestureRecognizer setDelegate:self];
        [self addGestureRecognizer:doubleTapGestureRecognizer];
        
        //this allow to differentiate between single tap and double tap
        [singleTapGestureRecognizer requireGestureRecognizerToFail:doubleTapGestureRecognizer];
        
        //swipe gesture
        UISwipeGestureRecognizer *swipeGestureRecognizer=[[UISwipeGestureRecognizer alloc]initWithTarget:self action:@selector(onSwipe:)];
        [self addGestureRecognizer:swipeGestureRecognizer];
        
        //long press gesture
        UILongPressGestureRecognizer *longPressGestureRecognizer=[[UILongPressGestureRecognizer alloc]initWithTarget:self action:@selector(onLongPress:)];
        [self addGestureRecognizer:longPressGestureRecognizer];
    }
    return(self);
}

/*
//only override draw rect:if we perform custom drawing.
//an empty implementation adversly affects performance during animation
- (void)drawRect:(CGRect)rect
{
   //drawing code
}
*/

+(Class)layerClass
{
    //code
    return([CAEAGLLayer class]);
}

void oglDraw_I(void)
{
    //code
    glBindVertexArray(vao_I);
    
    //draw scene
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    
    //unbind vao
    glBindVertexArray(0);
}

void oglDraw_N(void)
{
    //code
    glBindVertexArray(vao_N);
    
    //draw scene
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
    
    glLineWidth(20.0f);
    glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
    
    //unbind vao
    glBindVertexArray(0);
}

void oglDraw_D(void)
{
    //code
    glBindVertexArray(vao_D);

    GLfloat D_color[] =
    {
        f_DRedColor, f_DGreenColor, 0.0f,
        f_DRedColor, f_DGreenColor, 0.0f,
        f_DRedColor, f_DGreenColor, 0.0f,
        f_DRedColor, f_DGreenColor, 0.0f,

        0.0f, f_DGreenColor, 0.0f,
        0.0f, f_DGreenColor, 0.0f,
        0.0f, f_DGreenColor, 0.0f,
        0.0f, f_DGreenColor, 0.0f,

        f_DRedColor, f_DGreenColor, 0.0f,
        f_DRedColor, f_DGreenColor, 0.0f,
        0.0f, f_DGreenColor, 0.0f,
        0.0f, f_DGreenColor, 0.0f,

        f_DRedColor, f_DGreenColor, 0.0f,
        f_DRedColor, f_DGreenColor, 0.0f,
        0.0f, f_DGreenColor, 0.0f,
        0.0f, f_DGreenColor, 0.0f
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_D_color);
    glBufferData(GL_ARRAY_BUFFER, sizeof(D_color), D_color, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //draw scene
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 12, 4);

    //unbind vao
    glBindVertexArray(0);

}

void oglDraw_i(void)
{
    //code
    glBindVertexArray(vao_i);
    
    //draw scene
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    
    //unbind vao
    glBindVertexArray(0);
}

void oglDraw_A(void)
{
    //code
    glBindVertexArray(vao_A);
    
    //draw scene
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
    
    //unbind vao
    glBindVertexArray(0);
}

void oglDraw_middleStrips(void)
{
    //code
    glBindVertexArray(vao_middleStrips);

    const GLfloat middleStrips_color[] =
    {
        f_ARedColor, f_AGreenColor, 0.0f,
        f_ARedColor, f_AGreenColor, 0.0f,
        f_ARedColor, f_AGreenColor, 0.0f,
        f_ARedColor, f_AGreenColor, 0.0f,

        f_ARedColor, f_AWhiteColor, f_ABlueColor,
        f_ARedColor, f_AWhiteColor, f_ABlueColor,
        f_ARedColor, f_AWhiteColor, f_ABlueColor,
        f_ARedColor, f_AWhiteColor, f_ABlueColor,

        0.0f, f_AGreenColor, 0.0f,
        0.0f, f_AGreenColor, 0.0f,
        0.0f, f_AGreenColor, 0.0f,
        0.0f, f_AGreenColor, 0.0f
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_middleStrips_color);
    glBufferData(GL_ARRAY_BUFFER, sizeof(middleStrips_color), middleStrips_color, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //draw scene
    glLineWidth(3.0f);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 8, 4);

    //unbind vao
    glBindVertexArray(0);
}

void oglDraw_plane(void)
{
    //code
    glBindVertexArray(vao_plane);

    //plane
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    //body
    glDrawArrays(GL_TRIANGLES, 4, 3);        //exahaust
    glDrawArrays(GL_TRIANGLE_FAN, 7, 4);    //orange flag
    glDrawArrays(GL_TRIANGLE_FAN, 11, 4);    //white    flag
    glDrawArrays(GL_TRIANGLE_FAN, 15, 4);    //green    flag
    glDrawArrays(GL_LINES, 19, 2);            //separator line between exhaust and body
    glDrawArrays(GL_TRIANGLES, 21, 3);        //front tip
    glDrawArrays(GL_LINES, 24, 2);            //sperator line between front tip and body
    glDrawArrays(GL_TRIANGLES, 26, 3);        //upper wing
    glDrawArrays(GL_TRIANGLES, 29, 3);        //lower wing

    //IAF
    glDrawArrays(GL_LINES, 32, 2);            //I

    glDrawArrays(GL_LINES, 34, 2);            //A
    glDrawArrays(GL_LINES, 36, 2);
    glDrawArrays(GL_LINES, 38, 2);

    glDrawArrays(GL_LINES, 40, 2);            //F
    glDrawArrays(GL_LINES, 42, 2);
    glDrawArrays(GL_LINES, 44, 2);
    
    //unbind vao
    glBindVertexArray(0);
}

-(void)drawView:(id)sender
{
    //code
    [EAGLContext setCurrentContext:eaglContext_ap];
    
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glUseProgram(gShaderProgramObject);
    
   //declaration of metrices
    vmath::mat4 modelViewMatrix;
    vmath::mat4 modelViewProjectionMatrix;
    vmath::mat4 translationMatrix;
    vmath::mat4 rotationMatrix;        //for plane rotation

    //init above metrices to identity
    translationMatrix = vmath::mat4::identity();
    modelViewMatrix = vmath::mat4::identity();
    modelViewProjectionMatrix = vmath::mat4::identity();

    //do necessary transformations here
    translationMatrix = vmath::translate(f_Translate_I, 0.0f, -3.0f);

    //do necessary matrix multiplication
    modelViewMatrix = modelViewMatrix * translationMatrix;
    modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

    //send necessary matrics to shaders in respective uniforms
    glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

    oglDraw_I();

    if (b_I_Done)
    {
        //init above metrices to identity
        translationMatrix = vmath::mat4::identity();
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();

        //do necessary transformations here
        translationMatrix = vmath::translate(f_Translate_A, 0.0f, -3.0f);

        //do necessary matrix multiplication
        modelViewMatrix = modelViewMatrix * translationMatrix;
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        oglDraw_A();

        //init above metrices to identity
        translationMatrix = vmath::mat4::identity();
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();

        //do necessary transformations here
        translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
        translationMatrix = vmath::translate(6.8f, 0.0f, -20.0f);

        //do necessary matrix multiplication
        modelViewMatrix = modelViewMatrix * translationMatrix;
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        oglDraw_middleStrips();
    }

    if (b_A_Done)
    {
        //init above metrices to identity
        translationMatrix = vmath::mat4::identity();
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();

        //do necessary transformations here
        translationMatrix = vmath::translate(0.0f, f_Translate_N, -3.0f);

        //do necessary matrix multiplication
        modelViewMatrix = modelViewMatrix * translationMatrix;
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        oglDraw_N();
    }

    if (b_N_Done)
    {
        //init above metrices to identity
        translationMatrix = vmath::mat4::identity();
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();

        //do necessary transformations here
        translationMatrix = vmath::translate(0.0f, f_Translate_i, -3.0f);

        //do necessary matrix multiplication
        modelViewMatrix = modelViewMatrix * translationMatrix;
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        oglDraw_i();
    }
    
    if (b_i_Done)
    {
        //init above metrices to identity
        translationMatrix = vmath::mat4::identity();
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();

        //do necessary transformations here
        translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

        //do necessary matrix multiplication
        modelViewMatrix = modelViewMatrix * translationMatrix;
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        oglDraw_D();
    }

    if (b_D_Done)
    {
        /* TOP PLANE */

        //if (b_top_plane_smoke_done == false)
        //{
        //    translationMatrix = vmath::mat4::identity();
        //    //modelViewMatrix = vmath::mat4::identity();
        //    //modelViewProjectionMatrix = vmath::mat4::identity();
        //    translationMatrix = vmath::translate(0.2f, -0.45f, -20.0f);

        //    //do necessary matrix multiplication
        //    modelViewMatrix = modelViewMatrix * translationMatrix;
        //    modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //    //send necessary matrics to shaders in respective uniforms
        //    glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        //    oglDrawSmoke_top(1);

        //    if (b_start_incrementing == true)
        //    {
        //        oglDrawSmoke_top(2);
        //    }
        //}
        if (b_clip_top_plane == false)
        {
            //init above metrices to identity
            translationMatrix = vmath::mat4::identity();
            rotationMatrix = vmath::mat4::identity();
            modelViewMatrix = vmath::mat4::identity();
            modelViewProjectionMatrix = vmath::mat4::identity();
            
            //do necessary transformations here
            //translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
            translationMatrix = vmath::translate(top._x, top._y, -20.0f);
            rotationMatrix = vmath::rotate(top.rotation_angle, 0.0f, 0.0f, 1.0f);

            //do necessary matrix multiplication
            modelViewMatrix = modelViewMatrix * translationMatrix;
            modelViewMatrix = modelViewMatrix * rotationMatrix;
            modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

            //send necessary matrics to shaders in respective uniforms
            glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
            
            oglDraw_plane();

            //oglDrawSmoke();
        }


        //init above metrices to identity
        translationMatrix = vmath::mat4::identity();
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();

        //do necessary transformations here
        translationMatrix = vmath::translate(0.0f, 0.0f, -20.0f);

        //do necessary matrix multiplication
        modelViewMatrix = modelViewMatrix * translationMatrix;
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        //if (b_bottom_plane_smoke_done == false)
        //{
        //    translationMatrix = vmath::mat4::identity();
        //    //modelViewMatrix = vmath::mat4::identity();
        //    //modelViewProjectionMatrix = vmath::mat4::identity();
        //    translationMatrix = vmath::translate(0.0f, 0.0f, -20.0f);

        //    //do necessary matrix multiplication
        //    modelViewMatrix = modelViewMatrix * translationMatrix;
        //    modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //    //send necessary matrics to shaders in respective uniforms
        //    glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        //    oglDrawSmoke_bottom(1);

        //    if (b_start_decrementing == true)
        //    {
        //        oglDrawSmoke_bottom(2);
        //    }
        //}

        if (b_clip_bottom_plane == false)
        {
            //init above metrices to identity
            translationMatrix = vmath::mat4::identity();
            rotationMatrix = vmath::mat4::identity();
            modelViewMatrix = vmath::mat4::identity();
            modelViewProjectionMatrix = vmath::mat4::identity();

            //do necessary transformations here
            translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
            translationMatrix = vmath::translate(bottom._x, bottom._y, -20.0f);
            rotationMatrix = vmath::rotate(bottom.rotation_angle, 0.0f, 0.0f, 1.0f);

            //do necessary matrix multiplication
            modelViewMatrix = modelViewMatrix * translationMatrix;
            modelViewMatrix = modelViewMatrix * rotationMatrix;
            modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

            //send necessary matrics to shaders in respective uniforms
            glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

            oglDraw_plane();
        }

        //translationMatrix = vmath::mat4::identity();
        ////modelViewMatrix = vmath::mat4::identity();
        ////modelViewProjectionMatrix = vmath::mat4::identity();
        //translationMatrix = vmath::translate(0.0f, 0.0f, -20.0f);

        ////do necessary matrix multiplication
        //modelViewMatrix = modelViewMatrix * translationMatrix;
        //modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        ////send necessary matrics to shaders in respective uniforms
        //glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        //oglDrawSmoke_middle();

        //init above metrices to identity
        translationMatrix = vmath::mat4::identity();
        modelViewMatrix = vmath::mat4::identity();
        modelViewProjectionMatrix = vmath::mat4::identity();

        //do necessary transformations here
        translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
        translationMatrix = vmath::translate(x_plane_pos, 0.0f, -20.0f);

        //do necessary matrix multiplication
        modelViewMatrix = modelViewMatrix * translationMatrix;
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        oglDraw_plane();
    }

    
    //unuse program
    glUseProgram(0);
    
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext_ap presentRenderbuffer:GL_RENDERBUFFER];
    
    [self update];
}

- (void)update
{
    //init translation Of I
    if (f_Translate_I <= 0.0f)
    {
        f_Translate_I = f_Translate_I + 0.0045f;
        if (f_Translate_I > 0.0f)
        {
            b_I_Done = true;
        }
    }

    //init translation A
    if (b_I_Done)
    {
        if (f_Translate_A > 0.0f)
        {
            f_Translate_A = f_Translate_A - 0.0041f;
            if (f_Translate_A < 0.0f)
            {
                b_A_Done = true;
            }
        }
    }

    //init translation N
    if (b_A_Done == true)
    {
        if (f_Translate_N > 0.0f)
        {
            f_Translate_N = f_Translate_N - 0.006f;
            if (f_Translate_N < 0.0f)
            {
                b_N_Done = true;
            }
        }
    }

    //init translation i
    if (b_N_Done == true)
    {
        if (f_Translate_i < 0.0f)
        {
            f_Translate_i = f_Translate_i + 0.00455f;
            if (f_Translate_i > 0.0f)
            {
                b_i_Done = true;
            }
        }
    }

    //init color transition of D
    if (b_i_Done == true)
    {
        if ((f_DRedColor <= 1.0f) && (f_DGreenColor <= 0.5f))
        {
            f_DRedColor += 0.00032f;
            f_DGreenColor += 0.00016f;

            if ((f_DRedColor > 1.0f) && (f_DGreenColor > 0.5f))
            {
                b_D_Done = true;
            }
        }
    }

    //plane transition
    if (b_D_Done == true)
    {
        if (x_plane_pos <= 22.0f)
        {
            x_plane_pos = x_plane_pos + 0.0445f;

            /* top plane */
            if (top.angle <= (M_PI + M_PI_2))
            {
                top._x = top.radius * cosf(top.angle) - 8.0f;
                top._y = top.radius * sinf(top.angle) + 10.0f;
                top.angle += 0.005f;
                if (top.angle > (M_PI + M_PI_2))
                {
                    b_clip_top_plane = true;
                }
                if (top_angle_2 <= 4.71238f)
                {
                    //top_angle_2 = top_angle_2 + 0.00475f;
                    top_angle_2 = top_angle_2 + 0.00325f;
                }
            }
            //rotation angle calculation
            if (top.rotation_angle <= 0.0f)
            {
                top.rotation_angle += 0.21f;
            }

            /* bottom plane */
            if (bottom.angle >= M_PI_2)
            {
                bottom._x = bottom.radius * cosf(bottom.angle) - 8.0f;
                bottom._y = bottom.radius * sinf(bottom.angle) - 10.0f;
                bottom.angle -= 0.005f;
                if (bottom.angle <= M_PI_2)
                {
                    b_clip_bottom_plane = true;
                }
                if (bottom_angle_2 <= M_PI_2)
                {
                    bottom_angle_2 = bottom_angle_2 + 0.000475f;
                }
            }
            //rotation angle calculation
            if (bottom.rotation_angle >= 0.0f)
            {
                bottom.rotation_angle -= 0.21f;
            }

            if (x_plane_pos > 8.0f)
            {
                b_clip_top_plane = false;
                b_clip_bottom_plane = false;
                
                b_start_incrementing = true;
                b_start_decrementing = true;

                if (top.angle <= 2 * M_PI)
                {
                    top._x = top.radius * cosf(top.angle) + 8.0f;
                    top._y = top.radius * sinf(top.angle) + 10.0f;
                    top.angle += 0.005f;

                    if (top_angle_4 <= 6.28318f)
                    {
                        top_angle_4 += 0.00038f;
                    }

                    if (top.angle > 2 * M_PI)
                    {
                        b_clip_top_plane = true;
                    }
                    //angle related calculation
                    top.rotation_angle += 0.21f;
                }
                
                
                if (bottom.angle >= 0.0f)
                {
                    bottom._x = bottom.radius * cosf(bottom.angle) + 8.0f;
                    bottom._y = bottom.radius * sinf(bottom.angle) - 10.0f;
                    bottom.angle -= 0.005f;
                    if (bottom.angle <= 0.0f)
                    {
                        b_clip_bottom_plane = true;
                    }
                    if (bottom_angle_4 >= 0.0f)
                    {
                        bottom_angle_4 -= 0.00038f;
                    }
                    //angle related calculation
                    bottom.rotation_angle -= 0.21f;

                }
                if (x_plane_pos > 22.0f)
                {
                    b_PlaneTrue = true;
                }
            }
        }
    }

    if (b_PlaneTrue)
    {
        if ((f_ARedColor <= 1.0f) && (f_AGreenColor <= 0.5f) && (f_ABlueColor <= 1.0f) && (f_AWhiteColor <= 1.0f))
        {
            f_ARedColor = f_ARedColor + 0.002f;
            f_AGreenColor = f_AGreenColor + 0.001f;
            f_ABlueColor = f_ABlueColor + 0.002f;
            f_AWhiteColor = f_AWhiteColor + 0.002f;
            
            /*if ((r > 1.0f) && (g > 0.5f) && (b > 1.0f) && (White_A > 1.0f))
            {
                bColourDone = true;
            }*/
        }
    }
}

-(void)layoutSubviews
{
    //code
    GLint width;
    GLint height;
    
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext_ap renderbufferStorage:GL_RENDERBUFFER fromDrawable:(CAEAGLLayer *)self.layer];
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &width);
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &height);
    
    glGenRenderbuffers(1, &depthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
    
    glViewport(0, 0, width, height);
    
    GLfloat fwidth = (GLfloat)width;
    GLfloat fheight = (GLfloat)height;
    
    perspectiveProjectionMatrix = vmath::perspective(45.0f, ((GLfloat)fwidth / (GLfloat)fheight), 0.1f, 100.0f);

    
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        printf("Failed To Create Complete Framebuffer Object %x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
    }
    
    [self drawView:nil];    //repaint
}

-(void)startAnimation
{
    if(!isAnimating)
    {
        displayLink=[NSClassFromString(@"CADisplayLink")
                     displayLinkWithTarget:self selector:@selector(drawView:)];
        [displayLink setPreferredFramesPerSecond:animationFrameInterval];
        [displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
        
        isAnimating = YES;
    }
}

-(void)stopAnimation
{
    if(isAnimating)
    {
        [displayLink invalidate];
        displayLink=nil;
        
        isAnimating = NO;
    }
}

//to become first responder
- (BOOL)acceptsFirstResponder
{
    //code
    return (YES);
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
    //code
}

-(void)onSingleTap:(UITapGestureRecognizer *)gr
{
    //code
    
}

-(void)onDoubleTap:(UITapGestureRecognizer *)gr
{
    //code
    
}

-(void)onSwipe:(UISwipeGestureRecognizer *)gr
{
    //code
    [self release];
    exit(0);
}

-(void)onLongPress:(UILongPressGestureRecognizer *)gr
{
    //code
   
}

- (void)dealloc
{
    //code
    
    //code
    if (vbo_I_position)
    {
        glDeleteBuffers(1, &vbo_I_position);
        vbo_I_position = 0;
    }
    if (vbo_I_color)
    {
        glDeleteBuffers(1, &vbo_I_color);
        vbo_I_color = 0;
    }

    if (vbo_N_position)
    {
        glDeleteBuffers(1, &vbo_N_position);
        vbo_N_position = 0;
    }
    if (vbo_N_color)
    {
        glDeleteBuffers(1, &vbo_N_color);
        vbo_N_color = 0;
    }

    if (vbo_D_position)
    {
        glDeleteBuffers(1, &vbo_D_position);
        vbo_D_position = 0;
    }
    if (vbo_D_color)
    {
        glDeleteBuffers(1, &vbo_D_color);
        vbo_D_color = 0;
    }

    if (vbo_i_position)
    {
        glDeleteBuffers(1, &vbo_i_position);
        vbo_i_position = 0;
    }
    if (vbo_i_color)
    {
        glDeleteBuffers(1, &vbo_i_color);
        vbo_i_color = 0;
    }

    if (vbo_A_position)
    {
        glDeleteBuffers(1, &vbo_A_position);
        vbo_A_position = 0;
    }
    if (vbo_A_color)
    {
        glDeleteBuffers(1, &vbo_A_color);
        vbo_A_color = 0;
    }

    if (vbo_plane_position)
    {
        glDeleteBuffers(1, &vbo_plane_position);
        vbo_plane_position = 0;
    }
    if (vbo_plane_color)
    {
        glDeleteBuffers(1, &vbo_plane_color);
        vbo_plane_color = 0;
    }

    if (vao_I)
    {
        glDeleteVertexArrays(1, &vao_I);
        vao_I = 0;
    }

    if (vao_N)
    {
        glDeleteVertexArrays(1, &vao_N);
        vao_N = 0;
    }

    if (vao_D)
    {
        glDeleteVertexArrays(1, &vao_D);
        vao_D = 0;
    }

    if (vao_i)
    {
        glDeleteVertexArrays(1, &vao_i);
        vao_i = 0;
    }

    if (vao_A)
    {
        glDeleteVertexArrays(1, &vao_A);
        vao_A = 0;
    }

    if (vao_plane)
    {
        glDeleteVertexArrays(1, &vao_plane);
        vao_plane = 0;
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
    
    if(depthRenderbuffer)
    {
        glDeleteRenderbuffers(1, &depthRenderbuffer);
        depthRenderbuffer=0;
    }
    if(colorRenderbuffer)
    {
        glDeleteRenderbuffers(1, &colorRenderbuffer);
        colorRenderbuffer=0;
    }
    if(defaultFramebuffer)
    {
        glDeleteFramebuffers(1, &defaultFramebuffer);
        defaultFramebuffer=0;
    }
    
    if([EAGLContext currentContext]==eaglContext_ap)
    {
        [EAGLContext setCurrentContext:nil];
    }
    [EAGLContext release];
    eaglContext_ap=nil;
    
    [super dealloc];
}
@end
