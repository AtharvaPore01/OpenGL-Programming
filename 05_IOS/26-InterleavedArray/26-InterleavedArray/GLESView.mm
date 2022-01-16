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

GLfloat rotation_angle_cube = 0.0f;
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
    
    GLuint vao_cube;
    GLuint vbo_cube;
    
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
        "precision mediump int;" \
        "in vec4 vPosition;" \
        "in vec4 vColor;" \
        "in vec3 vNormal;" \
        "in vec2 vTexcoord;" \
        
        "uniform mat4 u_model_matrix;" \
        "uniform mat4 u_view_matrix;" \
        "uniform mat4 u_projection_matrix;" \
        "uniform int u_LKeyPressed;" \
        "uniform vec4 u_light_position;" \
        
        "out vec4 out_color;" \
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
        "out_color = vColor;" \
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
        
        "vec3 phong_ads_light;" \
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
        "phong_ads_light = (ambient + diffuse + specular) * vec3(out_color * texture(u_sampler, out_texcoord));" \
        "}" \
        "else" \
        "{" \
        "phong_ads_light = vec3(out_color * texture(u_sampler, out_texcoord));" \
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
                    printf("Program Link Error : \n %s\n", szInfoLog);
                    free(szInfoLog);
                    [self release];
                }
            }
        }
        
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
        
        GLfloat cubeVCNT[] =
        {
            1.0f, 1.0f, -1.0f,    1.0f, 0.0f, 0.0f,    0.0f, 1.0f,    0.0f, 1.0f, 0.0f,
            -1.0f, 1.0f, -1.0f,    1.0f, 0.0f, 0.0f,    0.0f, 0.0f,    0.0f, 1.0f, 0.0f,
            -1.0f, 1.0f, 1.0f,    1.0f, 0.0f, 0.0f,    1.0f, 0.0f,    0.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f,    1.0f, 0.0f, 0.0f,    1.0f, 1.0f,    0.0f, 1.0f, 0.0f,
            
            1.0f, -1.0f, -1.0f,    0.0f, 1.0f, 0.0f,    1.0f, 1.0f,    0.0f, -1.0f, 0.0f,
            -1.0f, -1.0f, -1.0f,0.0f, 1.0f, 0.0f,    0.0f, 1.0f,    0.0f, -1.0f, 0.0f,
            -1.0f, -1.0f, 1.0f,    0.0f, 1.0f, 0.0f,    0.0f, 0.0f,    0.0f, -1.0f, 0.0f,
            1.0f, -1.0f, 1.0f,    0.0f, 1.0f, 0.0f,    1.0f, 0.0f,    0.0f, -1.0f, 0.0f,
            
            1.0f, 1.0f, 1.0f,    0.0f, 0.0f, 1.0f,    0.0f, 0.0f,    0.0f, 0.0f, 1.0f,
            -1.0f, 1.0f, 1.0f,    0.0f, 0.0f, 1.0f,    1.0f, 0.0f,    0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 1.0f,    0.0f, 0.0f, 1.0f,    1.0f, 1.0f,    0.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f,    0.0f, 0.0f, 1.0f,    0.0f, 1.0f,    0.0f, 0.0f, 1.0f,
            
            1.0f, 1.0f, -1.0f,    0.0f, 1.0f, 1.0f,    1.0f, 0.0f,    0.0f, 0.0f, -1.0f,
            -1.0f, 1.0f, -1.0f,    0.0f, 1.0f, 1.0f,    1.0f, 1.0f,    0.0f, 0.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,0.0f, 1.0f, 1.0f,    0.0f, 1.0f,    0.0f, 0.0f, -1.0f,
            1.0f, -1.0f, -1.0f,    0.0f, 1.0f, 1.0f,    0.0f, 0.0f,    0.0f, 0.0f, -1.0f,
            
            1.0f, 1.0f, -1.0f,    1.0f, 0.0f, 1.0f,    1.0f, 0.0f,    1.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f,    1.0f, 0.0f, 1.0f,    1.0f, 1.0f,    1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f,    1.0f, 0.0f, 1.0f,    0.0f, 1.0f,    1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, -1.0f,    1.0f, 0.0f, 1.0f,    0.0f, 0.0f,    1.0f, 0.0f, 0.0f,
            
            -1.0f, 1.0f, -1.0f,    1.0f, 1.0f, 0.0f,    0.0f, 0.0f,    -1.0f, 0.0f, 0.0f,
            -1.0f, 1.0f, 1.0f,    1.0f, 1.0f, 0.0f,    1.0f, 0.0f,    -1.0f, 0.0f, 0.0f,
            -1.0f, -1.0f, 1.0f,    1.0f, 1.0f, 0.0f,    1.0f, 1.0f,    -1.0f, 0.0f, 0.0f,
            -1.0f, -1.0f, -1.0f,1.0f, 1.0f, 0.0f,    0.0f, 1.0f,    -1.0f, 0.0f, 0.0f
        };
        
        glGenVertexArrays(1, &vao_cube);
        glBindVertexArray(vao_cube);
        
        glGenBuffers(1, &vbo_cube);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_cube);
        glBufferData(GL_ARRAY_BUFFER, 24 * 11 * sizeof(GLfloat), cubeVCNT, GL_STATIC_DRAW);
        
        //position
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        
        //color
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        
        //texcoord
        glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOODR_0, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
        glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOODR_0);
        
        //normal
        glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void*)(8 * sizeof(GLfloat)));
        glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);

        //depth
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        
        //texture
        glEnable(GL_TEXTURE_2D);
        marble_texture = [self loadTexture:@"marble" :@"bmp"];
        
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

- (GLuint)loadTexture:(NSString *)texFileName :(NSString *)extension
{
    NSString *textureFileNameWithPath = [[NSBundle mainBundle] pathForResource:texFileName ofType:extension];
    
    //convert the image in cocoa format
    UIImage *bmpImage = [[UIImage alloc]initWithContentsOfFile:textureFileNameWithPath];
    if(!bmpImage)
    {
        NSLog(@"can't find %@", textureFileNameWithPath);
        return(0);
    }
    
    CGImageRef cgImage = bmpImage.CGImage;
    
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
    //code
    [EAGLContext setCurrentContext:eaglContext_ap];
    
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glUseProgram(gShaderProgramObject);
    
    glUseProgram(gShaderProgramObject);
    
    //declaration of metrices
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
    translationMatrix = vmath::translate(0.0f, 0.0f, -6.0f);
    rotationMatrix = vmath::rotate(rotation_angle_cube, rotation_angle_cube, rotation_angle_cube);
    
    //do necessary matrix multiplication
    modelMatrix = modelMatrix * translationMatrix;
    modelMatrix = modelMatrix * rotationMatrix;
    projectionMatrix = perspectiveProjectionMatrix;
    
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
    
    //bind with vao
    glBindVertexArray(vao_cube);
    
    //draw scene
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
    
    //unbind vao
    glBindVertexArray(0);
    
    //unuse program
    glUseProgram(0);
    
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext_ap presentRenderbuffer:GL_RENDERBUFFER];
    
    if(bAnimate == YES)
    {
        rotation_angle_cube = rotation_angle_cube + 1.0f;
        if (rotation_angle_cube >= 360.0f)
        {
            rotation_angle_cube = 0.0f;
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
    if(bAnimate == NO)
    {
        bAnimate = YES;
    }
    else
    {
        bAnimate = NO;
    }
}

-(void)onDoubleTap:(UITapGestureRecognizer *)gr
{
    //code
    if(bLight == NO)
    {
        bLight = YES;
    }
    else
        bLight = NO;
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
    
    if (vbo_cube)
    {
        glDeleteBuffers(1, &vbo_cube);
        vbo_cube = 0;
    }
    if (vao_cube)
    {
        glDeleteVertexArrays(1, &vao_cube);
        vao_cube = 0;
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
