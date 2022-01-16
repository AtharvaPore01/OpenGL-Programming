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

struct buffers
{
    GLuint vao;
    GLuint vbo_position;
    GLuint vbo_color;
}one, two, three, four, five, six;

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
        "in vec4 vCOlor;" \
        "uniform mat4 u_mvp_matrix;" \
        "out vec4 out_color;" \
        "void main(void)" \
        "{" \
        "gl_Position = u_mvp_matrix * vPosition;" \
        "gl_PointSize = 2.0;" \
        "out_color = vCOlor;" \
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
        
         /* vertices */
           const GLfloat firstDesign_vertices[] =
           {
               //First Row
               -1.7f, 0.9f, 0.0f,
               -1.5f, 0.9f, 0.0f,
               -1.3f, 0.9f, 0.0f,
               -1.1f, 0.9f, 0.0f,

               //Second Row
               -1.7f, 0.7f, 0.0f,
               -1.5f, 0.7f, 0.0f,
               -1.3f, 0.7f, 0.0f,
               -1.1f, 0.7f, 0.0f,

               //Third Row
               -1.7f, 0.5f, 0.0f,
               -1.5f, 0.5f, 0.0f,
               -1.3f, 0.5f, 0.0f,
               -1.1f, 0.5f, 0.0f,

               //Fourth Row
               -1.7f, 0.3f, 0.0f,
               -1.5f, 0.3f, 0.0f,
               -1.3f, 0.3f, 0.0f,
               -1.1f, 0.3f, 0.0f
           };

           const GLfloat secondDesign_vertice[] =
           {
               //1st Vertical Line
               -0.6f, 0.9f, 0.0f,
               -0.6f, 0.3f, 0.0f,
               //2nd Vertical Line
               -0.4f, 0.9f, 0.0f,
               -0.4f, 0.3f, 0.0f,
               //3rd Vertical Line
               -0.2f, 0.9f, 0.0f,
               -0.2f, 0.3f, 0.0f,
               
               //1st Horizontal Line
               -0.6f, 0.9f, 0.0f,
               -0.0f, 0.9f, 0.0f,

               //2nd Horizontal Line
               -0.6f, 0.7f, 0.0f,
               -0.0f, 0.7f, 0.0f,
               
               //3rd Horizontal Line
               -0.6f, 0.5f, 0.0f,
               -0.0f, 0.5f, 0.0f,

               //1st Olique Line
               -0.6f, 0.7f, 0.0f,
               -0.4f, 0.9, 0.0f,

               //2nd Olique Line
               -0.6f, 0.5f, 0.0f,
               -0.2f, 0.9, 0.0f,

               //3rd Olique Line
               -0.6f, 0.3f, 0.0f,
               -0.0f, 0.9f, 0.0f,

               //4th Olique Line
               -0.4f, 0.3f, 0.0f,
               -0.0f, 0.7f, 0.0f,

               -0.2f, 0.3f, 0.0f,
               -0.0f, 0.5f, 0.0f
           };

           const GLfloat thirdDesign_vertices[] =
           {
               //1st Vertical Line
               0.3f, 0.9f, 0.0f,
               0.3f, 0.3f, 0.0f,
               //2nd Vertical Line
               0.5f, 0.9f, 0.0f,
               0.5f, 0.3f, 0.0f,
               //3rd Vertical Line
               0.7f, 0.9f, 0.0f,
               0.7f, 0.3f, 0.0f,
               //4th Vertical Line
               0.9f, 0.9f, 0.0f,
               0.9f, 0.3f, 0.0f,
               //1st Horizontal Line
               0.3f, 0.9f, 0.0f,
               0.9f, 0.9f, 0.0f,
               //2nd Horizontal Line
               0.3f, 0.7f, 0.0f,
               0.9f, 0.7f, 0.0f,
               //3rd Horizontal Line
               0.3f, 0.5f, 0.0f,
               0.9f, 0.5f, 0.0f,
               //4th Horizontal Line
               0.3f, 0.3f, 0.0f,
               0.9f, 0.3f, 0.0f
           };

           const GLfloat fourthDesign_vertices[] =
           {
               //4th Row
               -1.7f, -0.9f, 0.0f,
               -1.1f, -0.9f, 0.0f,
               //3rd Row
               -1.7f, -0.7f, 0.0f,
               -1.1f, -0.7f, 0.0f,
               //2nd Row
               -1.7f, -0.5f, 0.0f,
               -1.1f, -0.5f, 0.0f,
               //1st Row
               -1.7f, -0.3f, 0.0f,
               -1.1f, -0.3f, 0.0f,

               //4th column
               -1.7f, -0.9f, 0.0f,
               -1.7f, -0.3f, 0.0f,
               //3rd Column
               -1.5f, -0.9f, 0.0f,
               -1.5f, -0.3f, 0.0f,
               //2nd Column
               -1.3f, -0.9f, 0.0f,
               -1.3f, -0.3f, 0.0f,
               //1st Column
               -1.1f, -0.9f, 0.0f,
               -1.1f, -0.3f, 0.0f,

               //1st Olique Line
               -1.7f, -0.5f, 0.0f,
               -1.5f, -0.3f, 0.0f,
               //2nd Olique Line
               -1.7f, -0.7f, 0.0f,
               -1.3f, -0.3f, 0.0f,
               //3rd Olique Line
               -1.7f, -0.9f, 0.0f,
               -1.1f, -0.3f, 0.0f,
               //4th Olique Line
               -1.5f, -0.9f, 0.0f,
               -1.1f, -0.5f, 0.0f,
               //5th Olique Line
               -1.3f, -0.9f, 0.0f,
               -1.1f, -0.7f, 0.0f
           };

           const GLfloat fifthDesign_vertices[] =
           {
               //4th Row
               -0.6f, -0.9f, 0.0f,
               -0.0f, -0.9f, 0.0f,
               //1st Row
               -0.6f, -0.3f, 0.0f,
               -0.0f, -0.3f, 0.0f,

               //4th column
               -0.6f, -0.9f, 0.0f,
               -0.6f, -0.3f, 0.0f,
               //1st Column
               0.0f, -0.9f, 0.0f,
               0.0f, -0.3f, 0.0f,

               //Ray
               -0.6f, -0.3f, 0.0f,
               0.0f, -0.5f, 0.0f,

               -0.6f, -0.3f, 0.0f,
               0.0f, -0.7f, 0.0f,

               -0.6f, -0.3f, 0.0f,
               0.0f, -0.9f, 0.0f,

               -0.6f, -0.3f, 0.0f,
               -0.4f, -0.9f, 0.0f,

               -0.6f, -0.3f, 0.0f,
               -0.2f, -0.9f, 0.0f
           };

           const GLfloat sixthDesign_vertices[] =
           {
               //first quad
               0.5f, -0.3f, 0.0f,
               0.3f, -0.3f, 0.0f,
               0.3f, -0.9f, 0.0f,
               0.5f, -0.9f, 0.0f,

               //second quad
               0.7f, -0.3f, 0.0f,
               0.5f, -0.3f, 0.0f,
               0.5f, -0.9f, 0.0f,
               0.7f, -0.9f, 0.0f,

               //third quad
               0.9f, -0.3f, 0.0f,
               0.7f, -0.3f, 0.0f,
               0.7f, -0.9f, 0.0f,
               0.9f, -0.9f, 0.0f,

               //vertical line 1
               0.5f, -0.3f, 0.0f,
               0.5f, -0.9f, 0.0f,

               //vertical line 2
               0.7f, -0.3f, 0.0f,
               0.7f, -0.9f, 0.0f,

               //Horizontal Line 1
               0.3f, -0.5f, 0.0f,
               0.9f, -0.5f, 0.0f,

               //Horizontal Line 1
               0.3f, -0.7f, 0.0f,
               0.9f, -0.7f, 0.0f
           };

           const GLfloat sixthDesign_color[] =
           {
               //first quad
               1.0f, 0.0f, 0.0f,
               1.0f, 0.0f, 0.0f,
               1.0f, 0.0f, 0.0f,
               1.0f, 0.0f, 0.0f,

               //second quad
               0.0f, 1.0f, 0.0f,
               0.0f, 1.0f, 0.0f,
               0.0f, 1.0f, 0.0f,
               0.0f, 1.0f, 0.0f,

               //third quad
               0.0f, 0.0f, 1.0f,
               0.0f, 0.0f, 1.0f,
               0.0f, 0.0f, 1.0f,
               0.0f, 0.0f, 1.0f,

               //vertical line 1
               1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f,

               //vertical line 2
               1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f,

               //Horizontal Line 1
               1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f,

               //Horizontal Line 1
               1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f
           };


           /* First Design */

           //generate and bind vao
           glGenVertexArrays(1, &one.vao);
           glBindVertexArray(one.vao);

           //generate and bind vbo
           glGenBuffers(1, &one.vbo_position);
           glBindBuffer(GL_ARRAY_BUFFER, one.vbo_position);

           glBufferData(GL_ARRAY_BUFFER, sizeof(firstDesign_vertices), firstDesign_vertices, GL_STATIC_DRAW);
           glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
           glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

           //unbind vbo_position
           glBindBuffer(GL_ARRAY_BUFFER, 0);

           glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

           //unbind vao
           glBindVertexArray(0);

           /* Second Design */

           //generate and bind vao
           glGenVertexArrays(1, &two.vao);
           glBindVertexArray(two.vao);

           //generate and bind vbo_position
           glGenBuffers(1, &two.vbo_position);
           glBindBuffer(GL_ARRAY_BUFFER, two.vbo_position);

           glBufferData(GL_ARRAY_BUFFER, sizeof(secondDesign_vertice), secondDesign_vertice, GL_STATIC_DRAW);
           glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
           glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

           //unbind vbo_position
           glBindBuffer(GL_ARRAY_BUFFER, 0);

           glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

           //unbind vao
           glBindVertexArray(0);

           /* Third Design */

           //generate and bind vao
           glGenVertexArrays(1, &three.vao);
           glBindVertexArray(three.vao);

           //generate and bind vbo_position
           glGenBuffers(1, &three.vbo_position);
           glBindBuffer(GL_ARRAY_BUFFER, three.vbo_position);

           glBufferData(GL_ARRAY_BUFFER, sizeof(thirdDesign_vertices), thirdDesign_vertices, GL_STATIC_DRAW);
           glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
           glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

           //unbind vbo_position
           glBindBuffer(GL_ARRAY_BUFFER, 0);

           glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

           //unbind vao
           glBindVertexArray(0);

           /* Fourth Design */

           //generate and bind vao
           glGenVertexArrays(1, &four.vao);
           glBindVertexArray(four.vao);

           //generate and bind vbo_position
           glGenBuffers(1, &four.vbo_position);
           glBindBuffer(GL_ARRAY_BUFFER, four.vbo_position);

           glBufferData(GL_ARRAY_BUFFER, sizeof(fourthDesign_vertices), fourthDesign_vertices, GL_STATIC_DRAW);
           glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
           glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

           //unbind vbo_position
           glBindBuffer(GL_ARRAY_BUFFER, 0);

           glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

           //unbind vao
           glBindVertexArray(0);

           /* Fifth Design */

           //generate and bind vao
           glGenVertexArrays(1, &five.vao);
           glBindVertexArray(five.vao);

           //generate and bind vbo_position
           glGenBuffers(1, &five.vbo_position);
           glBindBuffer(GL_ARRAY_BUFFER, five.vbo_position);

           glBufferData(GL_ARRAY_BUFFER, sizeof(fifthDesign_vertices), fifthDesign_vertices, GL_STATIC_DRAW);
           glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
           glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

           //unbind vbo_position
           glBindBuffer(GL_ARRAY_BUFFER, 0);

           glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

           //unbind vao
           glBindVertexArray(0);

           /* Sixth Design */

           //generate and bind vao
           glGenVertexArrays(1, &six.vao);
           glBindVertexArray(six.vao);

           //generate and bind vbo_position
           glGenBuffers(1, &six.vbo_position);
           glBindBuffer(GL_ARRAY_BUFFER, six.vbo_position);

           glBufferData(GL_ARRAY_BUFFER, sizeof(sixthDesign_vertices), sixthDesign_vertices, GL_STATIC_DRAW);
           glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
           glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

           //unbind vbo_position
           glBindBuffer(GL_ARRAY_BUFFER, 0);

           //generate and bind vbo_color
           glGenBuffers(1, &six.vbo_color);
           glBindBuffer(GL_ARRAY_BUFFER, six.vbo_color);

           glBufferData(GL_ARRAY_BUFFER, sizeof(sixthDesign_color), sixthDesign_color, GL_STATIC_DRAW);
           glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
           glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);

           //unbind vbo_color
           glBindBuffer(GL_ARRAY_BUFFER, 0);

           //unbind vao
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

-(void)drawView:(id)sender
{
    //Function declaration
    void DottedSquare(void);
    void Design_two(void);
    void Square(void);
    void SquareAndObliqueLine(void);
    void SquareAndRay(void);
    void RGB_Quads(void);

    //code
    [EAGLContext setCurrentContext:eaglContext_ap];
    
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glUseProgram(gShaderProgramObject);
    
     //declaration of metrices
       vmath::mat4 modelViewMatrix;
       vmath::mat4 modelViewProjectionMatrix;
       vmath::mat4 translationMatrix;

       //init above metrices to identity
       modelViewMatrix = vmath::mat4::identity();
       modelViewProjectionMatrix = vmath::mat4::identity();
       translationMatrix = vmath::mat4::identity();
       
       //do necessary transformations here
       modelViewMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

       //do necessary matrix multiplication
       modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

       //send necessary matrics to shaders in respective uniforms
       glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

       DottedSquare();
       SquareAndObliqueLine();

       //init above metrices to identity
       modelViewMatrix = vmath::mat4::identity();
       modelViewProjectionMatrix = vmath::mat4::identity();
       translationMatrix = vmath::mat4::identity();
       
       //do necessary transformations here
       modelViewMatrix = vmath::translate(0.2f, 0.0f, -3.0f);

       //do necessary matrix multiplication
       modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

       //send necessary matrics to shaders in respective uniforms
       glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

       Design_two();
       SquareAndRay();

       //init above metrices to identity
       modelViewMatrix = vmath::mat4::identity();
       modelViewProjectionMatrix = vmath::mat4::identity();
       translationMatrix = vmath::mat4::identity();
       
       //do necessary transformations here
       modelViewMatrix = vmath::translate(0.6f, 0.0f, -3.0f);

       //do necessary matrix multiplication
       modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

       //send necessary matrics to shaders in respective uniforms
       glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

       Square();
       RGB_Quads();


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

void DottedSquare(void)
{
    //glPointSize(2.0f);
    glBindVertexArray(one.vao);

    //First Row
    glDrawArrays(GL_POINTS, 0, 1);
    glDrawArrays(GL_POINTS, 1, 1);
    glDrawArrays(GL_POINTS, 2, 1);
    glDrawArrays(GL_POINTS, 3, 1);

    //Second Row
    glDrawArrays(GL_POINTS, 4, 1);
    glDrawArrays(GL_POINTS, 5, 1);
    glDrawArrays(GL_POINTS, 6, 1);
    glDrawArrays(GL_POINTS, 7, 1);

    //Third Row
    glDrawArrays(GL_POINTS, 8, 1);
    glDrawArrays(GL_POINTS, 9, 1);
    glDrawArrays(GL_POINTS, 10, 1);
    glDrawArrays(GL_POINTS, 11, 1);

    //Fourth Row
    glDrawArrays(GL_POINTS, 12, 1);
    glDrawArrays(GL_POINTS, 13, 1);
    glDrawArrays(GL_POINTS, 14, 1);
    glDrawArrays(GL_POINTS, 15, 1);

    glBindVertexArray(0);
}

void Design_two(void)
{

    glBindVertexArray(two.vao);
    
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
    
    

    glBindVertexArray(0);

}

void Square(void)
{


    glBindVertexArray(three.vao);

    //1st Vertical Line
    glDrawArrays(GL_LINES, 0, 2);
    //2nd Vertical Line
    glDrawArrays(GL_LINES, 2, 2);
    //3rd Vertical Line
    glDrawArrays(GL_LINES, 4, 2);
    //4th Vertical Line
    glDrawArrays(GL_LINES, 6, 2);

    //1st Horizontal Line
    glDrawArrays(GL_LINES, 8, 2);
    //2nd Horizontal Line
    glDrawArrays(GL_LINES, 10, 2);
    //3rd Horizontal Line
    glDrawArrays(GL_LINES, 12, 2);
    //4th Horizontal Line
    glDrawArrays(GL_LINES, 14, 2);

    glBindVertexArray(0);
}

void SquareAndObliqueLine(void)
{

    glBindVertexArray(four.vao);

    glDrawArrays(GL_LINES, 0, 2);//4th Row
    glDrawArrays(GL_LINES, 2, 2);//3rd Row
    glDrawArrays(GL_LINES, 4, 2);//2nd Row
    glDrawArrays(GL_LINES, 6, 2);//1st Row
    
    glDrawArrays(GL_LINES, 8, 2);//4th column
    glDrawArrays(GL_LINES, 10, 2);//3rd column
    glDrawArrays(GL_LINES, 12, 2);//2nd column
    glDrawArrays(GL_LINES, 14, 2);//1st column

    glDrawArrays(GL_LINES, 16, 2);//1st OliqueLine
    glDrawArrays(GL_LINES, 18, 2);//2nd OliqueLine
    glDrawArrays(GL_LINES, 20, 2);//3rd OliqueLine
    glDrawArrays(GL_LINES, 22, 2);//4th OliqueLine
    glDrawArrays(GL_LINES, 24, 2);//5th OliqueLine

    glBindVertexArray(0);
}

void SquareAndRay(void)
{
    glBindVertexArray(five.vao);

    glDrawArrays(GL_LINES, 0, 2);//4th Row
    glDrawArrays(GL_LINES, 2, 2);//1st Row
    glDrawArrays(GL_LINES, 4, 2);//4th column
    glDrawArrays(GL_LINES, 6, 2);//1st Column
    
    //ray
    glDrawArrays(GL_LINES, 8, 2);
    glDrawArrays(GL_LINES, 10, 2);
    glDrawArrays(GL_LINES, 12, 2);
    glDrawArrays(GL_LINES, 14, 2);
    glDrawArrays(GL_LINES, 16, 2);

    glBindVertexArray(0);
}

void RGB_Quads(void)
{
    glLineWidth(3.0f);
    glBindVertexArray(six.vao);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 8, 4);

    glDrawArrays(GL_LINES, 12, 2);//vertical line 1
    glDrawArrays(GL_LINES, 14, 2);//vertical line 2
    glDrawArrays(GL_LINES, 16, 2);//Horizontal Line 1
    glDrawArrays(GL_LINES, 18, 2);//Horizontal Line 2

    glBindVertexArray(0);
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
    
    if (one.vbo_position)
    {
        glDeleteBuffers(1, &one.vbo_position);
        one.vbo_position = 0;
    }
    if (one.vao)
    {
        glDeleteVertexArrays(1, &one.vao);
        one.vao = 0;
    }

    if (two.vbo_position)
    {
        glDeleteBuffers(1, &two.vbo_position);
        two.vbo_position = 0;
    }
    if (two.vao)
    {
        glDeleteVertexArrays(1, &two.vao);
        two.vao = 0;
    }

    if (three.vbo_position)
    {
        glDeleteBuffers(1, &three.vbo_position);
        three.vbo_position = 0;
    }
    if (three.vao)
    {
        glDeleteVertexArrays(1, &three.vao);
        three.vao = 0;
    }

    if (four.vbo_position)
    {
        glDeleteBuffers(1, &four.vbo_position);
        four.vbo_position = 0;
    }
    if (four.vao)
    {
        glDeleteVertexArrays(1, &four.vao);
        four.vao = 0;
    }

    if (five.vbo_position)
    {
        glDeleteBuffers(1, &five.vbo_position);
        five.vbo_position = 0;
    }
    if (five.vao)
    {
        glDeleteVertexArrays(1, &five.vao);
        five.vao = 0;
    }

    if (six.vbo_position)
    {
        glDeleteBuffers(1, &six.vbo_position);
        six.vbo_position = 0;
    }
    if (six.vbo_color)
    {
        glDeleteBuffers(1, &six.vbo_color);
        six.vbo_color = 0;
    }
    if (six.vao)
    {
        glDeleteVertexArrays(1, &six.vao);
        six.vao = 0;
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
