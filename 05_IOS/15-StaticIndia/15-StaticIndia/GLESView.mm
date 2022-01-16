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
            
            //middle strips
            0.66f, -0.05f, 0.0f,
            0.84f, -0.05f, 0.0f,
            
            0.65f, -0.1f, 0.0f,
            0.85f, -0.1f, 0.0f,
            
            0.64f, -0.15f, 0.0f,
            0.86f, -0.15f, 0.0f,
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
            
            1.0f, 0.5f, 0.0f,
            1.0f, 0.5f, 0.0f,
            
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            
            0.0f, 0.5f, 0.0f,
            0.0f, 0.5f, 0.0f
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
    
    glLineWidth(3.0f);
    glDrawArrays(GL_LINES, 8, 2);
    glDrawArrays(GL_LINES, 10, 2);
    glDrawArrays(GL_LINES, 12, 2);
    
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
    
    //init above metrices to identity
    modelViewMatrix = vmath::mat4::identity();
    modelViewProjectionMatrix = vmath::mat4::identity();
    
    //do necessary transformations here
    modelViewMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
    
    //do necessary matrix multiplication
    modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
    
    //send necessary matrics to shaders in respective uniforms
    glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
    
    oglDraw_I();
    oglDraw_N();
    oglDraw_D();
    oglDraw_i();
    oglDraw_A();

    
    //unuse program
    glUseProgram(0);
    
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext_ap presentRenderbuffer:GL_RENDERBUFFER];
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
