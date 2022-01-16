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

struct Light
{
    GLfloat Ambient[4];
    GLfloat Diffuse[4];
    GLfloat Specular[4];
    GLfloat Position[4];
};

Light Lights[2];

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 128.0f;                            //{128.0f};

//flags
BOOL bLight = NO;
BOOL bAnimate = NO;

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
    
    GLuint vao_pyramid;
    GLuint vbo_pyramid_position;
    GLuint vbo_pyramid_normal;
    
    GLuint ap_model_uniform;
    GLuint ap_view_uniform;
    GLuint ap_projection_uniform;
    
    GLuint ap_La_uniform_red;
    GLuint ap_La_uniform_blue;
    GLuint ap_Ld_uniform_red;
    GLuint ap_Ld_uniform_blue;
    GLuint ap_Ls_uniform_red;
    GLuint ap_Ls_uniform_blue;
    GLuint ap_lightPosition_uniform_red;
    GLuint ap_lightPosition_uniform_blue;
    
    GLuint ap_Ka_uniform;
    GLuint ap_Kd_uniform;
    GLuint ap_Ks_uniform;
    GLuint ap_shininess_uniform;
    GLuint ap_LKeyPressed_Uniform;

    //GLuint mvpUniform;
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
        
        //setting light properties
        Lights[0].Ambient[0] = 0.0f;
        Lights[0].Ambient[1] = 0.0f;
        Lights[0].Ambient[2] = 0.0f;
        Lights[0].Ambient[3] = 1.0f;
        
        Lights[0].Diffuse[0] = 1.0f;
        Lights[0].Diffuse[1] = 0.0f;
        Lights[0].Diffuse[2] = 0.0f;
        Lights[0].Diffuse[3] = 1.0f;
        
        Lights[0].Specular[0] = 1.0f;
        Lights[0].Specular[1] = 0.0f;
        Lights[0].Specular[2] = 0.0f;
        Lights[0].Specular[3] = 1.0f;
        
        Lights[0].Position[0] = -2.0f;
        Lights[0].Position[1] = 0.0f;
        Lights[0].Position[2] = 0.0f;
        Lights[0].Position[3] = 1.0f;
        
        Lights[1].Ambient[0] = 0.0f;
        Lights[1].Ambient[1] = 0.0f;
        Lights[1].Ambient[2] = 0.0f;
        Lights[1].Ambient[3] = 1.0f;
        
        Lights[1].Diffuse[0] = 0.0f;
        Lights[1].Diffuse[1] = 0.0f;
        Lights[1].Diffuse[2] = 1.0f;
        Lights[1].Diffuse[3] = 1.0f;
        
        Lights[1].Specular[0] = 0.0f;
        Lights[1].Specular[1] = 0.0f;
        Lights[1].Specular[2] = 1.0f;
        Lights[1].Specular[3] = 1.0f;
        
        Lights[1].Position[0] = 2.0f;
        Lights[1].Position[1] = 0.0f;
        Lights[1].Position[2] = 0.0f;
        Lights[1].Position[3] = 1.0f;
        
        /* Vertex Shader */
        //define vertex shader object
        gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
        
        //write vertex shader code
        const GLchar *vertexShaderSourceCode =
        "#version 300 es" \
        "\n" \
        "in vec4 vPosition;" \
        "in vec3 vNormal;" \
        "uniform mat4 u_model_matrix;" \
        "uniform mat4 u_view_matrix;" \
        "uniform mat4 u_projection_matrix;" \
        "uniform int u_LKeyPressed;" \
        "uniform vec3 u_La_red;" \
        "uniform vec3 u_La_blue;" \
        "uniform vec3 u_Ld_red;" \
        "uniform vec3 u_Ld_blue;" \
        "uniform vec3 u_Ls_red;" \
        "uniform vec3 u_Ls_blue;" \
        "uniform vec4 u_light_position_red;" \
        "uniform vec4 u_light_position_blue;" \
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
        "vec3 light_direction_red = normalize(vec3(u_light_position_red - eye_coordinates));" \
        "vec3 light_direction_blue = normalize(vec3(u_light_position_blue - eye_coordinates));" \
        "float tn_dot_LightDirection_red = max(dot(light_direction_red, transformed_normal), 0.0);" \
        "float tn_dot_LightDirection_blue = max(dot(light_direction_blue, transformed_normal), 0.0);" \
        "vec3 reflection_vector_red = reflect(-light_direction_red, transformed_normal);" \
        "vec3 reflection_vector_blue = reflect(-light_direction_blue, transformed_normal);" \
        "vec3 viewer_vector = normalize(vec3(-eye_coordinates.xyz));" \
        
        "vec3 ambient = (u_La_red * u_Ka) + (u_La_blue * u_Ka);" \
        "vec3 diffuse = (u_Ld_red * u_Kd * tn_dot_LightDirection_red) + ( u_Ld_blue * u_Kd * tn_dot_LightDirection_blue);" \
        "vec3 specular = (u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red, viewer_vector), 0.0), shininess)) + (u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue, viewer_vector), 0.0), shininess));" \
        
        "phong_ads_light = ambient + diffuse + specular;" \
        "}" \
        "else" \
        "{" \
        "phong_ads_light = vec3(1.0, 1.0, 1.0);" \
        "}" \
        "gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
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
        "in vec3 phong_ads_light;" \
        "out vec4 FragColor;" \
        "void main(void)" \
        "{" \
        "FragColor = vec4(phong_ads_light, 0.0);" \
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
        ap_model_uniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
        ap_view_uniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
        ap_projection_uniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
        ap_LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject, "u_LKeyPressed");
        
        ap_La_uniform_red = glGetUniformLocation(gShaderProgramObject, "u_La_red");
        ap_La_uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_La_blue");
        ap_Ld_uniform_red = glGetUniformLocation(gShaderProgramObject, "u_Ld_red");
        ap_Ld_uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_Ld_blue");
        ap_Ls_uniform_red = glGetUniformLocation(gShaderProgramObject, "u_Ls_red");
        ap_Ls_uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_Ls_blue");
        
        ap_Ka_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
        ap_Kd_uniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
        ap_Ks_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
        ap_shininess_uniform = glGetUniformLocation(gShaderProgramObject, "shininess");
        
        ap_lightPosition_uniform_red = glGetUniformLocation(gShaderProgramObject, "u_light_position_red");
        ap_lightPosition_uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_light_position_blue");

        
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
        const GLfloat pyramidNormals[] =
        {
            0.0f, 0.447214f, 0.894427f,
            0.0f, 0.447214f, 0.894427f,
            0.0f, 0.447214f, 0.894427f,
            
            0.89427f, 0.447214f, 0.0f,
            0.89427f, 0.447214f, 0.0f,
            0.89427f, 0.447214f, 0.0f,
            
            0.0f, 0.447214f, -0.894427f,
            0.0f, 0.447214f, -0.894427f,
            0.0f, 0.447214f, -0.894427f,
            
            -0.89427f, 0.447214f, 0.0f,
            -0.89427f, 0.447214f, 0.0f,
            -0.89427f, 0.447214f, 0.0f
            
        };
        
       
        //create vao and vbo
        
        //position
        glGenVertexArrays(1, &vao_pyramid);
        glBindVertexArray(vao_pyramid);
        glGenBuffers(1, &vbo_pyramid_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pyramid_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        //colour
        glGenBuffers(1, &vbo_pyramid_normal);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pyramid_normal);
        glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidNormals), pyramidNormals, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
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

-(void)drawView:(id)sender
{
    //variable
    static float rotation_angle_triangle = 0.0f;
    
    //code
    [EAGLContext setCurrentContext:eaglContext_ap];
    
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
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
    translationMatrix = vmath::translate(0.0f, 0.0f, -4.0f);
    rotationMatrix = vmath::rotate(rotation_angle_triangle, 0.0f, 1.0f, 0.0f);
    
    //do necessary matrix multiplication
    modelMatrix = modelMatrix * translationMatrix;
    modelMatrix = modelMatrix * rotationMatrix;
    projectionMatrix *= perspectiveProjectionMatrix;
    
    //send necessary matrics to shaders in respective uniforms
    glUniformMatrix4fv(ap_model_uniform, 1, GL_FALSE, modelMatrix);
    glUniformMatrix4fv(ap_view_uniform, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(ap_projection_uniform, 1, GL_FALSE, projectionMatrix);
    
    //if light is enabled
    if (bLight == YES)
    {
        //notify shader that we pressed the "L" key
        glUniform1i(ap_LKeyPressed_Uniform, 1);
        //send red light intensity
        glUniform3fv(ap_La_uniform_red, 1, Lights[0].Ambient);
        glUniform3fv(ap_Ld_uniform_red, 1, Lights[0].Diffuse);
        glUniform3fv(ap_Ls_uniform_red, 1, Lights[0].Specular);
        //send blue light intensity
        glUniform3fv(ap_La_uniform_blue, 1, Lights[1].Ambient);
        glUniform3fv(ap_Ld_uniform_blue, 1, Lights[1].Diffuse);
        glUniform3fv(ap_Ls_uniform_blue, 1, Lights[1].Specular);
        //send coeff. of material's reflectivity
        glUniform3fv(ap_Ka_uniform, 1, MaterialAmbient);
        glUniform3fv(ap_Kd_uniform, 1, MaterialDiffuse);
        glUniform3fv(ap_Ks_uniform, 1, MaterialSpecular);
        //shininess
        glUniform1f(ap_shininess_uniform, MaterialShininess);
        //send light position
        glUniform4fv(ap_lightPosition_uniform_red, 1, Lights[0].Position);
        glUniform4fv(ap_lightPosition_uniform_blue, 1, Lights[1].Position);
    }
    else
    {
        //notify shader that we aren't pressed the "L" key
        glUniform1i(ap_LKeyPressed_Uniform, 0);
    }
    
    //bind with vao
    glBindVertexArray(vao_pyramid);
    
    //draw scene
    glDrawArrays(GL_TRIANGLES, 0, 12);
    
    //unbind vao
    glBindVertexArray(0);

    
    //unuse program
    glUseProgram(0);
    
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext_ap presentRenderbuffer:GL_RENDERBUFFER];
    
    rotation_angle_triangle = rotation_angle_triangle + 1.0f;
    if (rotation_angle_triangle >= 360.0f)
    {
        rotation_angle_triangle = 0.0f;
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
    if (bLight == NO)
    {
        bLight = YES;
    }
    else
    {
        bLight = NO;
    }
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
    
    if (vbo_pyramid_position)
    {
        glDeleteBuffers(1, &vbo_pyramid_position);
        vbo_pyramid_position = 0;
    }
    if (vbo_pyramid_normal)
    {
        glDeleteBuffers(1, &vbo_pyramid_normal);
        vbo_pyramid_normal = 0;
    }
    if (vao_pyramid)
    {
        glDeleteVertexArrays(1, &vao_pyramid);
        vao_pyramid = 0;
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
