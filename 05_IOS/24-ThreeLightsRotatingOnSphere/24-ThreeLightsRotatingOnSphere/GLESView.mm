#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>

#import "vmath.h"

#import "GLESView.h"
#import "sphere.h"

enum
{
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_NORMAL,
    AMC_ATTRIBUTE_TEXCOODR_0
};

//sphere related variables
float ap_sphere_vertices[1146];
float ap_sphere_normals[1146];
float ap_sphere_texture[764];
short ap_sphere_elements[2280];
unsigned int ap_gNumVertices;
unsigned int ap_gNumElements;

//light related variables
GLuint model_uniform;
GLuint view_uniform;
GLuint projection_uniform;

GLuint La_uniform_red;
GLuint La_uniform_blue;
GLuint La_uniform_green;
GLuint Ld_uniform_red;
GLuint Ld_uniform_green;
GLuint Ld_uniform_blue;
GLuint Ls_uniform_red;
GLuint Ls_uniform_green;
GLuint Ls_uniform_blue;
GLuint lightPosition_uniform_red;
GLuint lightPosition_uniform_green;
GLuint lightPosition_uniform_blue;

GLuint Ka_uniform;
GLuint Kd_uniform;
GLuint Ks_uniform;
GLuint shininess_uniform;

GLuint LKeyPressed_Uniform;

//for rotation
float x_red = 0.0f;
float y_red = 0.0f;
float z_red = 0.0f;

float x_green = 0.0f;
float y_green = 0.0f;
float z_green = 0.0f;

float x_blue = 0.0f;
float y_blue = 0.0f;
float z_blue = 0.0f;

//light values
//Red
float LightAmbient_red[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
float LightSpecular_red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
float LightPosition_red[4] = { x_red, y_red, z_red, 1.0f };
//float LightPosition_red[4];
//green
float LightAmbient_green[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
float LightSpecular_green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
float LightPosition_green[4] = { x_green, y_green, z_green, 1.0f };
//float LightPosition_green[4];
//blue
float LightAmbient_blue[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
float LightDiffuse_blue[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
float LightSpecular_blue[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
float LightPosition_blue[4] = { x_blue, y_blue, z_blue, 1.0f };
//float LightPosition_blue[4];

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 128.0f;

float LightAngle_red = 0.0f;
float LightAngle_green = 0.0f;
float LightAngle_blue = 0.0f;


//flags
BOOL bLight = NO;

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
    
    GLuint vao_sphere;
    GLuint vbo_sphere_position;
    GLuint vbo_sphere_normal;
    GLuint vbo_sphere_element;
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
        "in vec3 vNormal;" \
        "uniform mat4 u_model_matrix;" \
        "uniform mat4 u_view_matrix;" \
        "uniform mat4 u_projection_matrix;" \
        "uniform int u_LKeyPressed;" \
        
        "uniform vec4 u_light_position_red;" \
        "uniform vec4 u_light_position_green;" \
        "uniform vec4 u_light_position_blue;" \
        
        "out vec3 t_norm;" \
        
        "out vec3 light_direction_red;" \
        "out vec3 light_direction_green;" \
        "out vec3 light_direction_blue;" \
        
        "out vec3 viewer_vector;" \
        "void main(void)" \
        "{" \
        "if (u_LKeyPressed == 1)" \
        "{" \
        "vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" \
        "mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" \
        "t_norm = normal_matrix * vNormal;" \
        
        "light_direction_red = vec3(u_light_position_red - eye_coordinates);" \
        "light_direction_green = vec3(u_light_position_green - eye_coordinates);" \
        "light_direction_blue = vec3(u_light_position_blue - eye_coordinates);" \
        
        "viewer_vector = vec3(-eye_coordinates);" \
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
        "in vec3 t_norm;" \
        
        "in vec3 light_direction_red;" \
        "in vec3 light_direction_green;" \
        "in vec3 light_direction_blue;" \
        
        "in vec3 viewer_vector;" \
        
        "uniform int u_LKeyPressed;" \
        
        "uniform vec3 u_La_red;" \
        "uniform vec3 u_La_green;" \
        "uniform vec3 u_La_blue;" \
        
        "uniform vec3 u_Ld_red;" \
        "uniform vec3 u_Ld_green;" \
        "uniform vec3 u_Ld_blue;" \
        
        "uniform vec3 u_Ls_red;" \
        "uniform vec3 u_Ls_green;" \
        "uniform vec3 u_Ls_blue;" \
        
        "uniform vec3 u_Ka;" \
        "uniform vec3 u_Kd;" \
        "uniform vec3 u_Ks;" \
        "uniform float shininess;" \
        
        "vec3 phong_ads_light;" \
        "out vec4 FragColor;" \
        
        "void main(void)" \
        "{" \
        "if(u_LKeyPressed == 1)" \
        "{" \
        "vec3 normalised_transformed_normal = normalize(t_norm);" \
        
        "vec3 normalised_light_direction_red = normalize(light_direction_red);" \
        "vec3 normalised_light_direction_green = normalize(light_direction_green);" \
        "vec3 normalised_light_direction_blue = normalize(light_direction_blue);" \
        
        "vec3 normalised_viewer_vector = normalize(viewer_vector);" \
        
        "vec3 reflection_vector_red = reflect(-normalised_light_direction_red, normalised_transformed_normal);" \
        "vec3 reflection_vector_green = reflect(-normalised_light_direction_green, normalised_transformed_normal);" \
        "vec3 reflection_vector_blue = reflect(-normalised_light_direction_blue, normalised_transformed_normal);" \
        
        "float tn_dot_LightDirection_red = max(dot(normalised_light_direction_red, normalised_transformed_normal), 0.0);" \
        "float tn_dot_LightDirection_green = max(dot(normalised_light_direction_green, normalised_transformed_normal), 0.0);" \
        "float tn_dot_LightDirection_blue = max(dot(normalised_light_direction_blue, normalised_transformed_normal), 0.0);" \
        
        "vec3 ambient = (u_La_red * u_Ka) + (u_La_green * u_Ka) + (u_La_blue * u_Ka);" \
        "vec3 diffuse = (u_Ld_red * u_Kd * tn_dot_LightDirection_red) + (u_Ld_green * u_Kd * tn_dot_LightDirection_green) + (u_Ld_blue * u_Kd * tn_dot_LightDirection_blue);" \
        "vec3 specular = (u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red, normalised_viewer_vector), 0.0), shininess)) + (u_Ls_green * u_Ks * pow(max(dot(reflection_vector_green, normalised_viewer_vector), 0.0), shininess)) + (u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue, normalised_viewer_vector), 0.0), shininess));" \
        
        "phong_ads_light = ambient + diffuse + specular;" \
        "}" \
        "else" \
        "{" \
        "phong_ads_light = vec3(1.0, 1.0, 1.0);" \
        "}" \
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
        LKeyPressed_Uniform = glGetUniformLocation(gShaderProgramObject, "u_LKeyPressed");
        
        model_uniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
        view_uniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
        projection_uniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
        
        La_uniform_red = glGetUniformLocation(gShaderProgramObject, "u_La_red");
        La_uniform_green = glGetUniformLocation(gShaderProgramObject, "u_La_green");
        La_uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_La_blue");
        
        Ld_uniform_red = glGetUniformLocation(gShaderProgramObject, "u_Ld_red");
        Ld_uniform_green = glGetUniformLocation(gShaderProgramObject, "u_Ld_green");
        Ld_uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_Ld_blue");
        
        Ls_uniform_red = glGetUniformLocation(gShaderProgramObject, "u_Ls_red");
        Ls_uniform_green = glGetUniformLocation(gShaderProgramObject, "u_Ls_green");
        Ls_uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_Ls_blue");
        
        Ka_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
        Kd_uniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
        Ks_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
        
        shininess_uniform = glGetUniformLocation(gShaderProgramObject, "shininess");
        
        lightPosition_uniform_red = glGetUniformLocation(gShaderProgramObject, "u_light_position_red");
        lightPosition_uniform_green = glGetUniformLocation(gShaderProgramObject, "u_light_position_green");
        lightPosition_uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_light_position_blue");
        
        
        //sphere vertices
        getSphereVertexData(ap_sphere_vertices, ap_sphere_normals, ap_sphere_texture, ap_sphere_elements);
        ap_gNumVertices = getNumberOfSphereVertices();
        ap_gNumElements = getNumberOfSphereElements();
        
        glGenVertexArrays(1, &vao_sphere);
        glBindVertexArray(vao_sphere);
        //position
        glGenBuffers(1, &vbo_sphere_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(ap_sphere_vertices), ap_sphere_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //normal
        glGenBuffers(1, &vbo_sphere_normal);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_normal);
        glBufferData(GL_ARRAY_BUFFER, sizeof(ap_sphere_normals), ap_sphere_normals, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        //elements
        glGenBuffers(1, &vbo_sphere_element);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ap_sphere_elements), ap_sphere_elements, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        
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
    
    vmath::mat4 rotationMatrixRed;
    vmath::mat4 rotationMatrixGreen;
    vmath::mat4 rotationMatrixBlue;
    
    //init above metrices to identity
    modelMatrix = vmath::mat4::identity();
    viewMatrix = vmath::mat4::identity();
    projectionMatrix = vmath::mat4::identity();
    translationMatrix = vmath::mat4::identity();
    
    //do necessary transformations here
    translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
    
    //do necessary matrix multiplication
    modelMatrix = modelMatrix * translationMatrix;
    projectionMatrix = perspectiveProjectionMatrix;
    
    //send necessary matrics to shaders in respective uniforms
    glUniformMatrix4fv(model_uniform, 1, GL_FALSE, modelMatrix);
    glUniformMatrix4fv(view_uniform, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(projection_uniform, 1, GL_FALSE, projectionMatrix);
    
    if (bLight)
    {
        //notify shader that we pressed the "L" key
        glUniform1i(LKeyPressed_Uniform, 1);
        //send light intensityx
        glUniform3fv(La_uniform_red, 1, LightAmbient_red);
        glUniform3fv(La_uniform_green, 1, LightAmbient_green);
        glUniform3fv(La_uniform_blue, 1, LightAmbient_blue);
        
        glUniform3fv(Ld_uniform_red, 1, LightDiffuse_red);
        glUniform3fv(Ld_uniform_green, 1, LightDiffuse_green);
        glUniform3fv(Ld_uniform_blue, 1, LightDiffuse_blue);
        
        glUniform3fv(Ls_uniform_red, 1, LightSpecular_red);
        glUniform3fv(Ls_uniform_green, 1, LightSpecular_green);
        glUniform3fv(Ls_uniform_blue, 1, LightSpecular_blue);
        
        //send coeff. of material's reflectivity
        glUniform3fv(Ka_uniform, 1, MaterialAmbient);
        glUniform3fv(Kd_uniform, 1, MaterialDiffuse);
        glUniform3fv(Ks_uniform, 1, MaterialSpecular);
        //shininess
        glUniform1f(shininess_uniform, MaterialShininess);
        //send light position
        
        LightPosition_red[0] = 0.0f;
        LightPosition_red[1] = 100.0f * cosf(LightAngle_red);
        LightPosition_red[2] = 100.0f * sinf(LightAngle_red);
        LightPosition_red[3] = 1.0f;
        glUniform4fv(lightPosition_uniform_red, 1, LightPosition_red);
        
        LightPosition_green[0] = 100.0f * cosf(LightAngle_green);
        LightPosition_green[1] = 0.0f;
        LightPosition_green[2] = 100.0f * sinf(LightAngle_green);
        LightPosition_green[3] = 1.0f;
        glUniform4fv(lightPosition_uniform_green, 1, LightPosition_green);
        
        LightPosition_blue[0] = 100.0f * cosf(LightAngle_blue);
        LightPosition_blue[1] = 100.0f * sinf(LightAngle_blue);
        LightPosition_blue[2] = 0.0f;
        LightPosition_blue[3] = 1.0f;
        glUniform4fv(lightPosition_uniform_blue, 1, LightPosition_blue);
    }
    else
    {
        //notify shader that we aren't pressed the "L" key
        glUniform1i(LKeyPressed_Uniform, 0);
    }
    
    //bind with vao
    glBindVertexArray(vao_sphere);
    
    //draw scene
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element);
    glDrawElements(GL_TRIANGLES, ap_gNumElements, GL_UNSIGNED_SHORT, 0);
    
    //unbind vao
    glBindVertexArray(0);
    
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext_ap presentRenderbuffer:GL_RENDERBUFFER];
    
    LightAngle_red = LightAngle_red + 0.02f;
    if (LightAngle_red >= 360)
    {
        LightAngle_red = 0.0f;
    }
    
    LightAngle_green = LightAngle_green + 0.02f;
    if (LightAngle_green >= 360)
    {
        LightAngle_green = 0.0f;
    }
    
    LightAngle_blue = LightAngle_blue + 0.02f;
    if (LightAngle_blue >= 360)
    {
        LightAngle_blue = 0.0f;
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
    if (bLight == FALSE)
    {
        bLight = TRUE;
    }
    else
    {
        bLight = FALSE;
    }
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
