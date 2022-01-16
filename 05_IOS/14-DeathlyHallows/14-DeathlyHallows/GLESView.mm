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
    
    GLuint vao_triangle;
    GLuint vao_circle;
    GLuint vao_line;
    
    GLuint vbo_triangle;
    GLuint vbo_circle;
    GLuint vbo_line;
    
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
        "void main(void)" \
        "{" \
        "FragColor = vec4(1.0, 1.0, 1.0, 1.0);" \
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
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        
        //make orthograhic projection matrix a identity matrix
        perspectiveProjectionMatrix = vmath::mat4::identity();
        
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
    
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext_ap presentRenderbuffer:GL_RENDERBUFFER];
    
    rotationAngle = rotationAngle + 1.0f;
    if (rotationAngle >= 360.0f)
    {
        rotationAngle = 0.0f;
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
