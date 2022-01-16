#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>

#import "vmath.h"

#import "GLESView.h"
#import "sphere.h"

#define RADIUS 100.0f

enum
{
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_NORMAL,
    AMC_ATTRIBUTE_TEXCOODR_0
};

//global variable
FILE *gpFile = NULL;

GLuint ap_model_uniform;
GLuint ap_view_uniform;
GLuint ap_projection_uniform;

GLuint ap_La_uniform;
GLuint ap_Ld_uniform;
GLuint ap_Ls_uniform;
GLuint ap_lightPosition_uniform;

GLuint ap_Ka_uniform;
GLuint ap_Kd_uniform;
GLuint ap_Ks_uniform;
GLuint ap_shininess_uniform;
GLuint ap_LKeyPressed_Uniform;


//sphere related variables
float ap_sphere_vertices[1146];
float ap_sphere_normals[1146];
float ap_sphere_texture[764];
short ap_sphere_elements[2280];
unsigned int ap_gNumVertices;
unsigned int ap_gNumElements;

//light values
float LightAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float LightDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightPosition[4] = { 100.0f, 100.0f, 100.0f, 1.0f };            //{ 1.0f, 1.0f, 1.0f, 1.0f };

//material values
float MaterialAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float MaterialDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float MaterialShininess = 128.0f;                            //{128.0f};

struct material_array
{
    float MaterialAmbient[4];
    float MaterialDiffuse[4];
    float MaterialSpecular[4];
    float MaterialShininess;
};

material_array mat_arr[24];
int iCount = 0;
float lightAngle = 0.0f;

//flags
BOOL bLight = NO;

int giWindowWidth = 0;
int giWindowHeight = 0;

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
        ap_La_uniform = glGetUniformLocation(gShaderProgramObject, "u_La");
        ap_Ld_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
        ap_Ls_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ls");
        ap_Ka_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
        ap_Kd_uniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
        ap_Ks_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
        ap_shininess_uniform = glGetUniformLocation(gShaderProgramObject, "shininess");
        ap_lightPosition_uniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");

        
        
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
        
        //material call
        [self oglInitMaterial];
        
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

- (void)oglDraw24Spheres
{
    //variable declaration
    int i = 0;
    float _x = -7.0f;
    float _y = 3.0f;
    
    //declaration of metrices
    vmath::mat4 modelMatrix;
    vmath::mat4 viewMatrix;
    vmath::mat4 projectionMatrix;
    vmath::mat4 translationMatrix;
    
    for (i = 0; i < 24; i++)
    {
        glViewport((i % 6) * giWindowWidth / 6, giWindowHeight - (i / 6 + 1) * giWindowHeight / 4, (GLsizei)giWindowWidth / 6, (GLsizei)giWindowHeight / 4);
        
        perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)(giWindowWidth / 6) / (GLfloat)(giWindowHeight / 4), 0.1f, 100.0f);
        //init above metrices to identity
        modelMatrix = vmath::mat4::identity();
        viewMatrix = vmath::mat4::identity();
        projectionMatrix = vmath::mat4::identity();
        translationMatrix = vmath::mat4::identity();
        
        //do necessary transformations here
        if (_x <= 5.0f)
        {
            _x = _x + 2.0f;
        }
        if (_x > 5.0f)
        {
            _x = -5.0f;
            _y = _y - 2.0f;
        }
        translationMatrix = vmath::translate(0.0f, 0.0f, -3.0f);
        
        //do necessary matrix multiplication
        modelMatrix = modelMatrix * translationMatrix;
        projectionMatrix *= perspectiveProjectionMatrix;
        
        //send necessary matrics to shaders in respective uniforms
        glUniformMatrix4fv(ap_model_uniform, 1, GL_FALSE, modelMatrix);
        glUniformMatrix4fv(ap_view_uniform, 1, GL_FALSE, viewMatrix);
        glUniformMatrix4fv(ap_projection_uniform, 1, GL_FALSE, projectionMatrix);
        
        //if light is enabled
        if (bLight)
        {
            //notify shader that we pressed the "L" key
            glUniform1i(ap_LKeyPressed_Uniform, 1);
            //send light intensityx
            glUniform3fv(ap_La_uniform, 1, LightAmbient);
            glUniform3fv(ap_Ld_uniform, 1, LightDiffuse);
            glUniform3fv(ap_Ls_uniform, 1, LightSpecular);
            //send coeff. of material's reflectivity
            glUniform3fv(ap_Ka_uniform, 1, mat_arr[i].MaterialAmbient);
            glUniform3fv(ap_Kd_uniform, 1, mat_arr[i].MaterialDiffuse);
            glUniform3fv(ap_Ks_uniform, 1, mat_arr[i].MaterialSpecular);
            //shininess
            glUniform1f(ap_shininess_uniform, mat_arr[i].MaterialShininess);
            //send light position
            if (iCount == 1)
            {
                LightPosition[0] = 0.0f;
                LightPosition[1] = cosf(lightAngle) * RADIUS;
                LightPosition[2] = sinf(lightAngle) * RADIUS;
                LightPosition[3] = 1.0f;
            }
            
            if (iCount == 2)
            {
                LightPosition[0] = cosf(lightAngle) * RADIUS;
                LightPosition[1] = 0.0f;
                LightPosition[2] = sinf(lightAngle) * RADIUS;
                LightPosition[3] = 1.0f;
            }
            
            if (iCount == 3)
            {
                LightPosition[0] = cosf(lightAngle) * RADIUS;
                LightPosition[1] = sinf(lightAngle) * RADIUS;
                LightPosition[2] = 0.0f;
                LightPosition[3] = 1.0f;
            }
            
            glUniform4fv(ap_lightPosition_uniform, 1, LightPosition);
        }
        else
        {
            //notify shader that we aren't pressed the "L" key
            glUniform1i(ap_LKeyPressed_Uniform, 0);
        }
        
        //bind with vao
        glBindVertexArray(vao_sphere);
        
        //draw scene
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element);
        glDrawElements(GL_TRIANGLES, ap_gNumElements, GL_UNSIGNED_SHORT, 0);
        
        //unbind vao
        glBindVertexArray(0);
    }
}

- (void)oglInitMaterial
{
    //code
    //emrald
    mat_arr[0].MaterialAmbient[0] = 0.0215f;
    mat_arr[0].MaterialAmbient[1] = 0.1745f;
    mat_arr[0].MaterialAmbient[2] = 0.0215f;
    mat_arr[0].MaterialAmbient[3] = 1.0f;
    mat_arr[0].MaterialDiffuse[0] = 0.07568f;
    mat_arr[0].MaterialDiffuse[1] = 0.61424f;
    mat_arr[0].MaterialDiffuse[2] = 0.07568f;
    mat_arr[0].MaterialDiffuse[3] = 1.0f;
    mat_arr[0].MaterialSpecular[0] = 0.633f;
    mat_arr[0].MaterialSpecular[1] = 0.727811f;
    mat_arr[0].MaterialSpecular[2] = 0.633f;
    mat_arr[0].MaterialSpecular[3] = 1.0f;
    mat_arr[0].MaterialShininess = 0.6f * 128.0f;
    
    //jade
    mat_arr[1].MaterialAmbient[0] = 0.135f;
    mat_arr[1].MaterialAmbient[1] = 0.2225f;
    mat_arr[1].MaterialAmbient[2] = 0.1575f;
    mat_arr[1].MaterialAmbient[3] = 1.0f;
    mat_arr[1].MaterialDiffuse[0] = 0.54f;
    mat_arr[1].MaterialDiffuse[1] = 0.89f;
    mat_arr[1].MaterialDiffuse[2] = 0.63f;
    mat_arr[1].MaterialDiffuse[3] = 1.0f;
    mat_arr[1].MaterialSpecular[0] = 0.316228f;
    mat_arr[1].MaterialSpecular[1] = 0.316228f;
    mat_arr[1].MaterialSpecular[2] = 0.316228f;
    mat_arr[1].MaterialSpecular[3] = 1.0f;
    mat_arr[1].MaterialShininess = 0.1f * 128.0f;
    
    //obsidian
    mat_arr[2].MaterialAmbient[0] = 0.05375f;
    mat_arr[2].MaterialAmbient[1] = 0.05f;
    mat_arr[2].MaterialAmbient[2] = 0.06625f;
    mat_arr[2].MaterialAmbient[3] = 1.0f;
    mat_arr[2].MaterialDiffuse[0] = 0.18275f;
    mat_arr[2].MaterialDiffuse[1] = 0.17f;
    mat_arr[2].MaterialDiffuse[2] = 0.22525f;
    mat_arr[2].MaterialDiffuse[3] = 1.0f;
    mat_arr[2].MaterialSpecular[0] = 0.332741f;
    mat_arr[2].MaterialSpecular[1] = 0.328634f;
    mat_arr[2].MaterialSpecular[2] = 0.346435f;
    mat_arr[2].MaterialSpecular[3] = 1.0f;
    mat_arr[2].MaterialShininess = 0.3f * 128.0f;
    
    //pearl
    mat_arr[3].MaterialAmbient[0] = 0.25f;
    mat_arr[3].MaterialAmbient[1] = 0.20725f;
    mat_arr[3].MaterialAmbient[2] = 0.20725f;
    mat_arr[3].MaterialAmbient[3] = 1.0f;
    mat_arr[3].MaterialDiffuse[0] = 1.0f;
    mat_arr[3].MaterialDiffuse[1] = 0.829f;
    mat_arr[3].MaterialDiffuse[2] = 0.829f;
    mat_arr[3].MaterialDiffuse[3] = 1.0f;
    mat_arr[3].MaterialSpecular[0] = 0.296648f;
    mat_arr[3].MaterialSpecular[1] = 0.296648f;
    mat_arr[3].MaterialSpecular[2] = 0.296648f;
    mat_arr[3].MaterialSpecular[3] = 1.0f;
    mat_arr[3].MaterialShininess = 0.088f * 128.0f;
    
    //ruby
    mat_arr[4].MaterialAmbient[0] = 0.1745f;
    mat_arr[4].MaterialAmbient[1] = 0.01175f;
    mat_arr[4].MaterialAmbient[2] = 0.01175f;
    mat_arr[4].MaterialAmbient[3] = 1.0f;
    mat_arr[4].MaterialDiffuse[0] = 0.61424f;
    mat_arr[4].MaterialDiffuse[1] = 0.04136f;
    mat_arr[4].MaterialDiffuse[2] = 0.04136f;
    mat_arr[4].MaterialDiffuse[3] = 1.0f;
    mat_arr[4].MaterialSpecular[0] = 0.727811f;
    mat_arr[4].MaterialSpecular[1] = 0.626959f;
    mat_arr[4].MaterialSpecular[2] = 0.626959f;
    mat_arr[4].MaterialSpecular[3] = 1.0f;
    mat_arr[4].MaterialShininess = 0.6f * 128.0f;
    
    //Turquoise
    mat_arr[5].MaterialAmbient[0] = 0.1f;
    mat_arr[5].MaterialAmbient[1] = 0.18725f;
    mat_arr[5].MaterialAmbient[2] = 0.1745f;
    mat_arr[5].MaterialAmbient[3] = 1.0f;
    mat_arr[5].MaterialDiffuse[0] = 0.396f;
    mat_arr[5].MaterialDiffuse[1] = 0.74151f;
    mat_arr[5].MaterialDiffuse[2] = 0.69102f;
    mat_arr[5].MaterialDiffuse[3] = 1.0f;
    mat_arr[5].MaterialSpecular[0] = 0.297254f;
    mat_arr[5].MaterialSpecular[1] = 0.30829f;
    mat_arr[5].MaterialSpecular[2] = 0.306678f;
    mat_arr[5].MaterialSpecular[3] = 1.0f;
    mat_arr[5].MaterialShininess = 0.1f * 128.0f;
    
    //brass
    mat_arr[6].MaterialAmbient[0] = 0.329412f;
    mat_arr[6].MaterialAmbient[1] = 0.223529f;
    mat_arr[6].MaterialAmbient[2] = 0.027451f;
    mat_arr[6].MaterialAmbient[3] = 1.0f;
    mat_arr[6].MaterialDiffuse[0] = 0.782392f;
    mat_arr[6].MaterialDiffuse[1] = 0.568627f;
    mat_arr[6].MaterialDiffuse[2] = 0.113725f;
    mat_arr[6].MaterialDiffuse[3] = 1.0f;
    mat_arr[6].MaterialSpecular[0] = 0.992157f;
    mat_arr[6].MaterialSpecular[1] = 0.941176f;
    mat_arr[6].MaterialSpecular[2] = 0.807843f;
    mat_arr[6].MaterialSpecular[3] = 1.0f;
    mat_arr[6].MaterialShininess = 0.21794872f * 128.0f;
    
    //bronze
    mat_arr[7].MaterialAmbient[0] = 0.2125f;
    mat_arr[7].MaterialAmbient[1] = 0.1275f;
    mat_arr[7].MaterialAmbient[2] = 0.054f;
    mat_arr[7].MaterialAmbient[3] = 1.0f;
    mat_arr[7].MaterialDiffuse[0] = 0.714f;
    mat_arr[7].MaterialDiffuse[1] = 0.4284f;
    mat_arr[7].MaterialDiffuse[2] = 0.18144f;
    mat_arr[7].MaterialDiffuse[3] = 1.0f;
    mat_arr[7].MaterialSpecular[0] = 0.393548f;
    mat_arr[7].MaterialSpecular[1] = 0.271906f;
    mat_arr[7].MaterialSpecular[2] = 0.166721f;
    mat_arr[7].MaterialSpecular[3] = 1.0f;
    mat_arr[7].MaterialShininess = 0.2f * 128.0f;
    
    //chrome
    mat_arr[8].MaterialAmbient[0] = 0.25f;
    mat_arr[8].MaterialAmbient[1] = 0.25f;
    mat_arr[8].MaterialAmbient[2] = 0.25f;
    mat_arr[8].MaterialAmbient[3] = 1.0f;
    mat_arr[8].MaterialDiffuse[0] = 0.4f;
    mat_arr[8].MaterialDiffuse[1] = 0.4f;
    mat_arr[8].MaterialDiffuse[2] = 0.4f;
    mat_arr[8].MaterialDiffuse[3] = 1.0f;
    mat_arr[8].MaterialSpecular[0] = 0.774597f;
    mat_arr[8].MaterialSpecular[1] = 0.774597f;
    mat_arr[8].MaterialSpecular[2] = 0.774597f;
    mat_arr[8].MaterialSpecular[3] = 1.0f;
    mat_arr[8].MaterialShininess = 0.6f * 128.0f;
    
    //copper
    mat_arr[9].MaterialAmbient[0] = 0.19125f;
    mat_arr[9].MaterialAmbient[1] = 0.0735f;
    mat_arr[9].MaterialAmbient[2] = 0.0225f;
    mat_arr[9].MaterialAmbient[3] = 1.0f;
    mat_arr[9].MaterialDiffuse[0] = 0.7038f;
    mat_arr[9].MaterialDiffuse[1] = 0.27048f;
    mat_arr[9].MaterialDiffuse[2] = 0.0828f;
    mat_arr[9].MaterialDiffuse[3] = 1.0f;
    mat_arr[9].MaterialSpecular[0] = 0.256777f;
    mat_arr[9].MaterialSpecular[1] = 0.137622f;
    mat_arr[9].MaterialSpecular[2] = 0.086014f;
    mat_arr[9].MaterialSpecular[3] = 1.0f;
    mat_arr[9].MaterialShininess = 0.1f * 128.0f;
    
    //gold
    mat_arr[10].MaterialAmbient[0] = 0.24725f;
    mat_arr[10].MaterialAmbient[1] = 0.1995f;
    mat_arr[10].MaterialAmbient[2] = 0.0745f;
    mat_arr[10].MaterialAmbient[3] = 1.0f;
    mat_arr[10].MaterialDiffuse[0] = 0.75164f;
    mat_arr[10].MaterialDiffuse[1] = 0.60648f;
    mat_arr[10].MaterialDiffuse[2] = 0.22648f;
    mat_arr[10].MaterialDiffuse[3] = 1.0f;
    mat_arr[10].MaterialSpecular[0] = 0.628281f;
    mat_arr[10].MaterialSpecular[1] = 0.555802f;
    mat_arr[10].MaterialSpecular[2] = 0.366065f;
    mat_arr[10].MaterialSpecular[3] = 1.0f;
    mat_arr[10].MaterialShininess = 0.4f * 128.0f;
    
    //silver
    mat_arr[11].MaterialAmbient[0] = 0.19225f;
    mat_arr[11].MaterialAmbient[1] = 0.19225f;
    mat_arr[11].MaterialAmbient[2] = 0.19225f;
    mat_arr[11].MaterialAmbient[3] = 1.0f;
    mat_arr[11].MaterialDiffuse[0] = 0.50754f;
    mat_arr[11].MaterialDiffuse[1] = 0.50754f;
    mat_arr[11].MaterialDiffuse[2] = 0.50754f;
    mat_arr[11].MaterialDiffuse[3] = 1.0f;
    mat_arr[11].MaterialSpecular[0] = 0.508273f;
    mat_arr[11].MaterialSpecular[1] = 0.508273f;
    mat_arr[11].MaterialSpecular[2] = 0.508273f;
    mat_arr[11].MaterialSpecular[3] = 1.0f;
    mat_arr[11].MaterialShininess = 0.4f * 128.0f;
    
    //Black Plastic
    mat_arr[12].MaterialAmbient[0] = 0.0f;
    mat_arr[12].MaterialAmbient[1] = 0.0f;
    mat_arr[12].MaterialAmbient[2] = 0.0f;
    mat_arr[12].MaterialAmbient[3] = 1.0f;
    mat_arr[12].MaterialDiffuse[0] = 0.01f;
    mat_arr[12].MaterialDiffuse[1] = 0.01f;
    mat_arr[12].MaterialDiffuse[2] = 0.01f;
    mat_arr[12].MaterialDiffuse[3] = 1.0f;
    mat_arr[12].MaterialSpecular[0] = 0.50f;
    mat_arr[12].MaterialSpecular[1] = 0.50f;
    mat_arr[12].MaterialSpecular[2] = 0.50f;
    mat_arr[12].MaterialSpecular[3] = 1.0f;
    mat_arr[12].MaterialShininess = 0.25f * 128.0f;
    //Cyan Plastic
    mat_arr[13].MaterialAmbient[0] = 0.0f;
    mat_arr[13].MaterialAmbient[1] = 0.1f;
    mat_arr[13].MaterialAmbient[2] = 0.06f;
    mat_arr[13].MaterialAmbient[3] = 1.0f;
    mat_arr[13].MaterialDiffuse[0] = 0.01f;
    mat_arr[13].MaterialDiffuse[1] = 0.50980392f;
    mat_arr[13].MaterialDiffuse[2] = 0.50980392f;
    mat_arr[13].MaterialDiffuse[3] = 1.0f;
    mat_arr[13].MaterialSpecular[0] = 0.50196078f;
    mat_arr[13].MaterialSpecular[1] = 0.50196078f;
    mat_arr[13].MaterialSpecular[2] = 0.50196078f;
    mat_arr[13].MaterialSpecular[3] = 1.0f;
    mat_arr[13].MaterialShininess = 0.25f * 128.0f;
    //Green Plastic
    mat_arr[14].MaterialAmbient[0] = 0.0f;
    mat_arr[14].MaterialAmbient[1] = 0.0f;
    mat_arr[14].MaterialAmbient[2] = 0.0f;
    mat_arr[14].MaterialAmbient[3] = 1.0f;
    mat_arr[14].MaterialDiffuse[0] = 0.1f;
    mat_arr[14].MaterialDiffuse[1] = 0.35f;
    mat_arr[14].MaterialDiffuse[2] = 0.1f;
    mat_arr[14].MaterialDiffuse[3] = 1.0f;
    mat_arr[14].MaterialSpecular[0] = 0.45f;
    mat_arr[14].MaterialSpecular[1] = 0.55f;
    mat_arr[14].MaterialSpecular[2] = 0.45f;
    mat_arr[14].MaterialSpecular[3] = 1.0f;
    mat_arr[14].MaterialShininess = 0.25f * 128.0f;
    //Red Plastic
    mat_arr[15].MaterialAmbient[0] = 0.0f;
    mat_arr[15].MaterialAmbient[1] = 0.0f;
    mat_arr[15].MaterialAmbient[2] = 0.0f;
    mat_arr[15].MaterialAmbient[3] = 1.0f;
    mat_arr[15].MaterialDiffuse[0] = 0.5f;
    mat_arr[15].MaterialDiffuse[1] = 0.0f;
    mat_arr[15].MaterialDiffuse[2] = 0.0f;
    mat_arr[15].MaterialDiffuse[3] = 1.0f;
    mat_arr[15].MaterialSpecular[0] = 0.7f;
    mat_arr[15].MaterialSpecular[1] = 0.6f;
    mat_arr[15].MaterialSpecular[2] = 0.6f;
    mat_arr[15].MaterialSpecular[3] = 1.0f;
    mat_arr[15].MaterialShininess = 0.25f * 128.0f;
    //White Plastic
    mat_arr[16].MaterialAmbient[0] = 0.0f;
    mat_arr[16].MaterialAmbient[1] = 0.0f;
    mat_arr[16].MaterialAmbient[2] = 0.0f;
    mat_arr[16].MaterialAmbient[3] = 1.0f;
    mat_arr[16].MaterialDiffuse[0] = 0.55f;
    mat_arr[16].MaterialDiffuse[1] = 0.55f;
    mat_arr[16].MaterialDiffuse[2] = 0.55f;
    mat_arr[16].MaterialDiffuse[3] = 1.0f;
    mat_arr[16].MaterialSpecular[0] = 0.70f;
    mat_arr[16].MaterialSpecular[1] = 0.70f;
    mat_arr[16].MaterialSpecular[2] = 0.70f;
    mat_arr[16].MaterialSpecular[3] = 1.0f;
    mat_arr[16].MaterialShininess = 0.25f * 128.0f;
    //yellow Plastic
    mat_arr[17].MaterialAmbient[0] = 0.0f;
    mat_arr[17].MaterialAmbient[1] = 0.0f;
    mat_arr[17].MaterialAmbient[2] = 0.0f;
    mat_arr[17].MaterialAmbient[3] = 1.0f;
    mat_arr[17].MaterialDiffuse[0] = 0.5f;
    mat_arr[17].MaterialDiffuse[1] = 0.5f;
    mat_arr[17].MaterialDiffuse[2] = 0.0f;
    mat_arr[17].MaterialDiffuse[3] = 1.0f;
    mat_arr[17].MaterialSpecular[0] = 0.60f;
    mat_arr[17].MaterialSpecular[1] = 0.60f;
    mat_arr[17].MaterialSpecular[2] = 0.50f;
    mat_arr[17].MaterialSpecular[3] = 1.0f;
    mat_arr[17].MaterialShininess = 0.25f * 128.0f;
    
    //Black Rubber
    mat_arr[18].MaterialAmbient[0] = 0.02f;
    mat_arr[18].MaterialAmbient[1] = 0.02f;
    mat_arr[18].MaterialAmbient[2] = 0.02f;
    mat_arr[18].MaterialAmbient[3] = 1.0f;
    mat_arr[18].MaterialDiffuse[0] = 0.01f;
    mat_arr[18].MaterialDiffuse[1] = 0.01f;
    mat_arr[18].MaterialDiffuse[2] = 0.01f;
    mat_arr[18].MaterialDiffuse[3] = 1.0f;
    mat_arr[18].MaterialSpecular[0] = 0.4f;
    mat_arr[18].MaterialSpecular[1] = 0.4f;
    mat_arr[18].MaterialSpecular[2] = 0.4f;
    mat_arr[18].MaterialSpecular[3] = 1.0f;
    mat_arr[18].MaterialShininess = 0.078125f * 128.0f;
    
    //Cyan Rubber
    mat_arr[19].MaterialAmbient[0] = 0.0f;
    mat_arr[19].MaterialAmbient[1] = 0.05f;
    mat_arr[19].MaterialAmbient[2] = 0.05f;
    mat_arr[19].MaterialAmbient[3] = 1.0f;
    mat_arr[19].MaterialDiffuse[0] = 0.4f;
    mat_arr[19].MaterialDiffuse[1] = 0.5f;
    mat_arr[19].MaterialDiffuse[2] = 0.5f;
    mat_arr[19].MaterialDiffuse[3] = 1.0f;
    mat_arr[19].MaterialSpecular[0] = 0.04f;
    mat_arr[19].MaterialSpecular[1] = 0.7f;
    mat_arr[19].MaterialSpecular[2] = 0.7f;
    mat_arr[19].MaterialSpecular[3] = 1.0f;
    mat_arr[19].MaterialShininess = 0.078125f * 128.0f;
    
    //Green Rubber
    mat_arr[20].MaterialAmbient[0] = 0.0f;
    mat_arr[20].MaterialAmbient[1] = 0.05f;
    mat_arr[20].MaterialAmbient[2] = 0.0f;
    mat_arr[20].MaterialAmbient[3] = 1.0f;
    mat_arr[20].MaterialDiffuse[0] = 0.4f;
    mat_arr[20].MaterialDiffuse[1] = 0.5f;
    mat_arr[20].MaterialDiffuse[2] = 0.4f;
    mat_arr[20].MaterialDiffuse[3] = 1.0f;
    mat_arr[20].MaterialSpecular[0] = 0.04f;
    mat_arr[20].MaterialSpecular[1] = 0.7f;
    mat_arr[20].MaterialSpecular[2] = 0.04f;
    mat_arr[20].MaterialSpecular[3] = 1.0f;
    mat_arr[20].MaterialShininess = 0.078125f * 128.0f;
    
    //Red Rubber
    mat_arr[21].MaterialAmbient[0] = 0.05f;
    mat_arr[21].MaterialAmbient[1] = 0.0f;
    mat_arr[21].MaterialAmbient[2] = 0.0f;
    mat_arr[21].MaterialAmbient[3] = 1.0f;
    mat_arr[21].MaterialDiffuse[0] = 0.5f;
    mat_arr[21].MaterialDiffuse[1] = 0.4f;
    mat_arr[21].MaterialDiffuse[2] = 0.4f;
    mat_arr[21].MaterialDiffuse[3] = 1.0f;
    mat_arr[21].MaterialSpecular[0] = 0.7f;
    mat_arr[21].MaterialSpecular[1] = 0.04f;
    mat_arr[21].MaterialSpecular[2] = 0.04f;
    mat_arr[21].MaterialSpecular[3] = 1.0f;
    mat_arr[21].MaterialShininess = 0.078125f * 128.0f;
    
    //White Rubber
    mat_arr[22].MaterialAmbient[0] = 0.05f;
    mat_arr[22].MaterialAmbient[1] = 0.05f;
    mat_arr[22].MaterialAmbient[2] = 0.05f;
    mat_arr[22].MaterialAmbient[3] = 1.0f;
    mat_arr[22].MaterialDiffuse[0] = 0.5f;
    mat_arr[22].MaterialDiffuse[1] = 0.5f;
    mat_arr[22].MaterialDiffuse[2] = 0.5f;
    mat_arr[22].MaterialDiffuse[3] = 1.0f;
    mat_arr[22].MaterialSpecular[0] = 0.7f;
    mat_arr[22].MaterialSpecular[1] = 0.7f;
    mat_arr[22].MaterialSpecular[2] = 0.7f;
    mat_arr[22].MaterialSpecular[3] = 1.0f;
    mat_arr[22].MaterialShininess = 0.078125f * 128.0f;
    
    //Yellow Rubber
    mat_arr[23].MaterialAmbient[0] = 0.05f;
    mat_arr[23].MaterialAmbient[1] = 0.05f;
    mat_arr[23].MaterialAmbient[2] = 0.0f;
    mat_arr[23].MaterialAmbient[3] = 1.0f;
    mat_arr[23].MaterialDiffuse[0] = 0.5f;
    mat_arr[23].MaterialDiffuse[1] = 0.5f;
    mat_arr[23].MaterialDiffuse[2] = 0.4f;
    mat_arr[23].MaterialDiffuse[3] = 1.0f;
    mat_arr[23].MaterialSpecular[0] = 0.7f;
    mat_arr[23].MaterialSpecular[1] = 0.7f;
    mat_arr[23].MaterialSpecular[2] = 0.04f;
    mat_arr[23].MaterialSpecular[3] = 1.0f;
    mat_arr[23].MaterialShininess = 0.078125f * 128.0f;
}


-(void)drawView:(id)sender
{
    //code
    [EAGLContext setCurrentContext:eaglContext_ap];
    
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glUseProgram(gShaderProgramObject);
    
    [self oglDraw24Spheres];
    
    //unuse program
    glUseProgram(0);

    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext_ap presentRenderbuffer:GL_RENDERBUFFER];
    
    lightAngle = lightAngle + 0.005f;
    if (lightAngle >= 360)
    {
        lightAngle = 0.0f;
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
    
    giWindowWidth = width;
    giWindowHeight = height;
    
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
    iCount = iCount + 1;
    if(iCount > 3)
    {
        iCount = 0;
    }
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
