#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>

#import "vmath.h"
#import <math.h>
#import "GLESView.h"

enum
{
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_NORMAL,
    AMC_ATTRIBUTE_TEXCOODR_0
};


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
    
    GLuint vao_red;
    GLuint vao_green;
    GLuint vao_blue;
    GLuint vao_circumscribed_circle;
    GLuint vao_circumscribed_circle_triangle;
    
    GLuint vao_circumscribed_rectangle;
    GLuint vao_circumscribed_rectangle_circle;
    
    GLuint vbo_red_line_position;
    GLuint vbo_red_line_color;
    GLuint vbo_green_line_position;
    GLuint vbo_green_line_color;
    GLuint vbo_blue_line_position;
    
    GLuint vbo_circumscribed_circle_position_triangle;
    GLuint vbo_circumscribed_circle_position_circle;
    GLuint vbo_circumscribed_circle_color;
    GLuint vbo_circumscribed_circle_color_triangle;
    
    GLuint vbo_circumscribed_rectangle_position;
    GLuint vbo_circumscribed_rectangle_position_circle;
    GLuint vbo_circumscribed_rectangle_color;
    GLuint vbo_circumscribed_rectangle_color_circle;
    
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
        
        //triangle of circumscribed circle
        const GLfloat triangleVertices[] =
        {
            0.0f, 1.2f, 0.0f,
            -1.0f, -0.6f, 0.0f,
            
            -1.0f, -0.6f, 0.0f,
            1.0f, -0.6f, 0.0f,
            
            1.0f, -0.6f, 0.0f,
            0.0f, 1.2f, 0.0f
        };
        
        const GLfloat circumscribedCircleColor[] =
        {
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f
        };
        
        //circumscribed rectangle
        const GLfloat rectangleVertices[] =
        {
            1.0f, 0.58f, 0.0f,
            -1.0f, 0.58f, 0.0f,
            
            -1.0f, -0.6f, 0.0f,
            1.0f, -0.6f, 0.0f,
            
            1.0f, -0.6f, 0.0f,
            1.0f, 0.58f, 0.0f,
            
            -1.0f, 0.58f, 0.0f,
            -1.0f, -0.6f, 0.0f
        };
        
        const GLfloat circumscribedRectangleColor[] =
        {
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            
            1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f
            
        };
        
        //circumscribed circle
        //triangle
        glGenVertexArrays(1, &vao_circumscribed_circle_triangle);
        glBindVertexArray(vao_circumscribed_circle_triangle);
        
        glGenBuffers(1, &vbo_circumscribed_circle_position_triangle);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_position_triangle);
        glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glGenBuffers(1, &vbo_circumscribed_circle_color_triangle);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_color_triangle);
        glBufferData(GL_ARRAY_BUFFER, sizeof(circumscribedCircleColor), circumscribedCircleColor, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        //circle
        glGenVertexArrays(1, &vao_circumscribed_circle);
        glBindVertexArray(vao_circumscribed_circle);
        glGenBuffers(1, &vbo_circumscribed_circle_position_circle);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_position_circle);
        glBufferData(GL_ARRAY_BUFFER, 1 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glGenBuffers(1, &vbo_circumscribed_circle_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_color);
        glBufferData(GL_ARRAY_BUFFER, sizeof(circumscribedCircleColor), circumscribedCircleColor, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        //circumscribed rectangle
        //rectangle
        glGenVertexArrays(1, &vao_circumscribed_rectangle);
        glBindVertexArray(vao_circumscribed_rectangle);
        
        glGenBuffers(1, &vbo_circumscribed_rectangle_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_position);
        
        glBufferData(GL_ARRAY_BUFFER, sizeof(rectangleVertices), rectangleVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glGenBuffers(1, &vbo_circumscribed_rectangle_color);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_color);
        
        glBufferData(GL_ARRAY_BUFFER, sizeof(circumscribedRectangleColor), circumscribedRectangleColor, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        //circle
        glGenVertexArrays(1, &vao_circumscribed_rectangle_circle);
        glBindVertexArray(vao_circumscribed_rectangle_circle);
        
        glGenBuffers(1, &vbo_circumscribed_rectangle_position_circle);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_position_circle);
        
        glBufferData(GL_ARRAY_BUFFER, 1 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glGenBuffers(1, &vbo_circumscribed_rectangle_color_circle);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_color_circle);
        
        glBufferData(GL_ARRAY_BUFFER, sizeof(circumscribedRectangleColor), circumscribedRectangleColor, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        
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
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        
        //make orthograhic projection matrix a identity matrix
        perspectiveProjectionMatrix = vmath::mat4::identity();
        
        //clear color
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        
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

- (void)oglGenCircleInsideTriangle
{
    //OGLCircleInsideTriangle Variables
    GLfloat radius, a, b, c;
    GLfloat Perimeter;
    GLfloat Area_Of_Triangle, x_center, y_center;
    GLfloat circleVertices[3];
    
    //function declaration
    GLfloat findDistance(GLfloat, GLfloat, GLfloat, GLfloat);
    GLfloat findPerimeter(GLfloat, GLfloat, GLfloat);
    GLfloat findXCenter(GLfloat a, GLfloat b, GLfloat c);
    GLfloat findYCenter(GLfloat a, GLfloat b, GLfloat c);
    GLfloat findAreaOfTriangle(GLfloat Perimeter, GLfloat a, GLfloat b, GLfloat c);
    GLfloat findRadius(GLfloat AreaOfTrianlge, GLfloat Perimeter);
    
    //code
    //Distance Between Vertices Of The Triangle
    a = sqrtf((powf((-1.0f - 0.0f), 2) + powf((-0.6f - 1.2f), 2)));
    b = sqrtf((powf((1.0f - (-1.0f)), 2) + powf((-0.6f - (-0.6f)), 2)));
    c = sqrtf((powf((0.0f - 1.0f), 2) + powf((1.2f - (-0.6f)), 2)));
    
    //Semi Perimeter
    Perimeter = (a + b + c) / 2;
    
    //Area Of Trianle Using Heron's Formula
    Area_Of_Triangle = sqrtf(Perimeter * (Perimeter - a) * (Perimeter - b) * (Perimeter - c));
    
    //Radius Of Circle
    radius = Area_Of_Triangle / Perimeter;
    
    //Center Of The Circle
    x_center = ((a * 1.0f) + (b * (0.0f)) + (c * (-1.0f))) / (a + b + c);
    y_center = ((a * (-0.6f)) + (b * (1.2f)) + (c * (-0.6f))) / (a + b + c);
    
    //bind with vao
    glBindVertexArray(vao_circumscribed_circle_triangle);
    
    glDrawArrays(GL_LINES, 0, 2);
    glDrawArrays(GL_LINES, 2, 2);
    glDrawArrays(GL_LINES, 4, 2);
    
    //unbind vao
    glBindVertexArray(0);
    
    //bind with vao
    glBindVertexArray(vao_circumscribed_circle);
    for (GLfloat angle = 0.0f; angle < (2.0f * M_PI); angle = angle + 0.01f)
    {
        circleVertices[0] = ((cosf(angle) * radius) + x_center);
        circleVertices[1] = ((sinf(angle) * radius) + y_center);
        circleVertices[2] = 0.0f;
        
        //vertices
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_circle_position_circle);
        glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //draw scene
        glDrawArrays(GL_POINTS, 0, 1);
        //glDrawArrays(GL_LINE_LOOP, 0, 10);
    }
    
    //unbind vao
    glBindVertexArray(0);
}

-(void)oglGenCircumscribedRectangle
{
    //variable declaration
    //GLfloat Number_Of_Sides = 4.0f;
    GLfloat radius;
    GLfloat Top, Bottom, Left, Right, Area_of_Rectangle;
    GLfloat circleVertices[3];
    
    //code
    
    //Rectangle's Sides
    Top = sqrtf((powf((-1.0f - 1.0f), 2) + powf((0.58f - 0.58f), 2)));
    Bottom = sqrtf((powf((1.0f - (-1.0f)), 2) + powf((-0.6f - (-0.6f)), 2)));
    Left = sqrtf((powf((1.0f - 1.0f), 2) + powf((-0.58f - 0.6f), 2)));
    Right = sqrtf((powf((-1.0f - (-1.0f)), 2) + powf((-0.6f - 0.58f), 2)));
    
    //Area Of Rectangle
    Area_of_Rectangle = (Top + Bottom + Left + Right) / 2;
    
    //Radius
    radius = (sqrtf((pow(Bottom, 2)) + (pow(Right, 2))) / 2);
    
    //bind with vao
    glBindVertexArray(vao_circumscribed_rectangle);
    
    glDrawArrays(GL_LINES, 0, 2);
    glDrawArrays(GL_LINES, 2, 2);
    glDrawArrays(GL_LINES, 4, 2);
    glDrawArrays(GL_LINES, 6, 2);
    
    //unbind vao
    glBindVertexArray(0);
    
    //bind with vao
    glBindVertexArray(vao_circumscribed_rectangle_circle);
    for (GLfloat angle = 0.0f; angle < (2.0f * M_PI); angle = angle + 0.01f)
    {
        circleVertices[0] = ((cosf(angle) * radius));
        circleVertices[1] = ((sinf(angle) * radius));
        circleVertices[2] = 0.0f;
        
        //vertices
        glBindBuffer(GL_ARRAY_BUFFER, vbo_circumscribed_rectangle_position_circle);
        glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //draw scene
        glDrawArrays(GL_POINTS, 0, 1);
    }
    
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
    
    //circumscribed circle
    
    //init above metrices to identity
    modelViewMatrix = vmath::mat4::identity();
    modelViewProjectionMatrix = vmath::mat4::identity();
    translationMatrix = vmath::mat4::identity();
    
    //do necessary transformations here
    translationMatrix = vmath::translate(0.0f, 0.0f, -3.9f);
    
    //do necessary matrix multiplication
    modelViewMatrix *= translationMatrix;
    modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
    
    //send necessary matrics to shaders in respective uniforms
    glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
    
    [self oglGenCircleInsideTriangle];
    
    [self oglGenCircumscribedRectangle];
    
    //init above metrices to identity
    modelViewMatrix = vmath::mat4::identity();
    modelViewProjectionMatrix = vmath::mat4::identity();
    
    //do necessary transformations here
    modelViewMatrix = vmath::translate(0.0f, 0.0f, -1.2f);
    
    //do necessary matrix multiplication
    modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
    
    //send necessary matrics to shaders in respective uniforms
    glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
    
    //bind with vao red
    glBindVertexArray(vao_red);
    //draw scene
    glDrawArrays(GL_LINES, 0, 2);
    //unbind vao red
    glBindVertexArray(0);
    
    //bind with vao green
    glBindVertexArray(vao_green);
    //draw scene
    glDrawArrays(GL_LINES, 0, 2);
    //unbind vao green
    glBindVertexArray(0);
    
    //bind with vao blue
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
    
    //unbind vao blue
    glBindVertexArray(0);
    
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
    if (vbo_circumscribed_circle_color_triangle)
    {
        glDeleteBuffers(1, &vbo_circumscribed_circle_color_triangle);
        vbo_circumscribed_circle_color_triangle = 0;
    }
    if (vbo_circumscribed_circle_color)
    {
        glDeleteBuffers(1, &vbo_circumscribed_circle_color);
        vbo_circumscribed_circle_color = 0;
    }
    if (vbo_circumscribed_circle_position_triangle)
    {
        glDeleteBuffers(1, &vbo_circumscribed_circle_position_triangle);
        vbo_circumscribed_circle_position_triangle = 0;
    }
    if (vbo_circumscribed_circle_position_circle)
    {
        glDeleteBuffers(1, &vbo_circumscribed_circle_position_circle);
        vbo_circumscribed_circle_position_circle = 0;
    }
    if (vbo_circumscribed_rectangle_color)
    {
        glDeleteBuffers(1, &vbo_circumscribed_rectangle_color);
        vbo_circumscribed_rectangle_color = 0;
    }
    if (vbo_circumscribed_rectangle_color_circle)
    {
        glDeleteBuffers(1, &vbo_circumscribed_rectangle_color_circle);
        vbo_circumscribed_rectangle_color_circle = 0;
    }
    if (vbo_circumscribed_rectangle_position)
    {
        glDeleteBuffers(1, &vbo_circumscribed_rectangle_position);
        vbo_circumscribed_rectangle_position = 0;
    }
    if (vbo_circumscribed_rectangle_position_circle)
    {
        glDeleteBuffers(1, &vbo_circumscribed_rectangle_position_circle);
        vbo_circumscribed_rectangle_position_circle = 0;
    }
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
    if (vao_circumscribed_circle)
    {
        glDeleteVertexArrays(1, &vao_circumscribed_circle);
        vao_circumscribed_circle = 0;
    }
    if (vao_circumscribed_circle_triangle)
    {
        glDeleteVertexArrays(1, &vao_circumscribed_circle_triangle);
        vao_circumscribed_circle_triangle = 0;
    }
    if (vao_circumscribed_rectangle)
    {
        glDeleteVertexArrays(1, &vao_circumscribed_rectangle);
        vao_circumscribed_rectangle = 0;
    }
    if (vao_circumscribed_rectangle_circle)
    {
        glDeleteVertexArrays(1, &vao_circumscribed_rectangle_circle);
        vao_circumscribed_rectangle_circle = 0;
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
