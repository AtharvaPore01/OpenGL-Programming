package com.Atharva_Pore.two_lights_on_rotating_pyramid;

//programmable related (OpenGL Related) packages
import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import javax.microedition.khronos.opengles.GL10;			//for basic features of openGL-ES
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;									//nio = Non Blocking I/O OR Native I/O,	For Opengl Buffers.
import java.nio.ByteOrder;									//for arranginf byte order of the buffer in native byte order(Little Indian / Big Indian)
import java.nio.FloatBuffer;								//to create float type buffer.

import android.opengl.Matrix;								//for matrix mathematics.

//standard packages
import android.content.Context; 							//for Context drawingContext class
import android.graphics.Color;								//for Color class
import android.view.Gravity;								//for Gravity class	
import android.view.MotionEvent;							//for MotionEvent			
import android.view.GestureDetector;						//for GestureDetector
import android.view.GestureDetector.OnGestureListener;		//for OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener;	//for OnDoubleTapListener

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener
{
	private final Context context;
	private GestureDetector gestureDetector;

	//shader related variables
	//java doesn't have unsigned int so ther is no GLuint So we are using int(in Windows and XWindows we did GLuint).
	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	//java don't have addresses to send so we are sending arrays name as its base address
	private int[] vao_pyramid = new int[1];
	private int[] vbo_pyramid_position = new int[1];
	private int[] vbo_pyramid_normal = new int[1];

	//private int mvpUniform;

	private int model_uniform;
	private int view_uniform;
	private int projection_uniform;
	private int La_uniform_red;
	private int La_uniform_blue;
	private int Ld_uniform_red;
	private int Ld_uniform_blue;
	private int Ls_uniform_red;
	private int Ls_uniform_blue;
	private int lightPosition_uniform_red;
	private int lightPosition_uniform_blue;
	private int Ka_uniform;
	private int Kd_uniform;
	private int Ks_uniform;
	private int shininess_uniform;
	private int singleTap_Uniform;

	private float[] Ambient_red = new float[] { 0.0f, 0.0f, 0.0f, 1.0f };
	private float[] Diffuse_red = new float[] { 1.0f, 0.0f, 0.0f, 1.0f };
	private float[] Specular_red = new float[] { 1.0f, 0.0f, 0.0f, 1.0f };
	private float[] Position_red = new float[] { -2.0f, 0.0f, 0.0f, 1.0f };

	private float[] Ambient_blue = new float[] { 0.0f, 0.0f, 0.0f, 1.0f };
	private float[] Diffuse_blue = new float[] { 0.0f, 0.0f, 1.0f, 1.0f };
	private float[] Specular_blue = new float[] { 0.0f, 0.0f, 1.0f, 1.0f };
	private float[] Position_blue = new float[] { 2.0f, 0.0f, 0.0f, 1.0f };

	//material values
	private float[] MaterialAmbient = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
	private float[] MaterialDiffuse = new float[] { 1.0f, 1.0f, 1.0f, 1.0f };
	private float[] MaterialSpecular = new float[] { 1.0f, 1.0f, 1.0f, 1.0f };
	private float MaterialShininess = 128.0f;

	private float[] perspectiveProjectionMatrix = new float[16];		//16 because it is 4 x 4 matrix.

	//Rotation variables
	float rotation_angle_pyramid = 0.0f;
	
	//flags
	boolean bLight = false;
	boolean bAnimate = false;

	GLESView(Context drawingContext)
	{
		super(drawingContext);
		context = drawingContext;

		//tell egl that the incoming version of opengl-es is 3 onwards.
		setEGLContextClientVersion(3);

		setRenderer(this);

		//tell opengl-es to repaint the render mode when it will be dirty.
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

		gestureDetector = new GestureDetector(drawingContext, this, null, false);
		/*	GestureDetector Asks 
		*	Parameter 1 :- Give Me The Global Environment.
		*	Parameter 2 :- Who Will Listen? We Tell Him Me So this is given.
		*	Parameter 3 :- Does There Anyone Who Will Handle This? We Say No So null.
		*	Parameter 4 :- unused(internally used by android) so STRICTLY false.
		*/ 
	}

	// Handling 'OnTouchEvent' Is The Most IMPORTANT
	// Because It Triggers All Gesture And Tap Event.

	@Override
	public boolean onTouchEvent(MotionEvent event)
	{
		//code
		int eventaction = event.getAction();	//this is not requiered in any OpenGL Code.
		if(!gestureDetector.onTouchEvent(event))
		{
			super.onTouchEvent(event);
		}
		return(true);
	} 

	// abstract method for onDoubleTapEvent so must be implementd
	@Override
	public boolean onDoubleTap(MotionEvent event)
	{
		if(bAnimate == false)
		{
			bAnimate = true;
		}
		else
		{
			bAnimate = false;
		}
		return(true);
	}

	// abstract method for onDoubleTapEvent so must be implementd
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//No Code Because We Already Written In 'onDoubleTap'
		return(true);
	}

	// abstract method for onDoubleTapEvent so must be implementd
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		if(bLight == false)
		{
			bLight = true;
		}
		else
		{
			bLight = false;
		}
		return(true);
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public boolean onDown(MotionEvent e)
	{
		//No Code Because We Already Written In onSingleTapConfirmed
		return(true);
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocity_x, float velocity_y)
	{
		return(true);
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public void onLongPress(MotionEvent e)
	{

	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public boolean onScroll(MotionEvent e1, MotionEvent e2, float distance_x, float distance_y)
	{
		oglUninitialise();
		System.exit(0);
		return(true);
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public void onShowPress(MotionEvent e)
	{

	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return(true);
	}

	//implement render class method
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config)
	{
		String version = gl.glGetString(GL10.GL_VERSION);
		String shadingLanguageVersion = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String renderer = gl.glGetString(GLES32.GL_RENDERER);
		String vendor = gl.glGetString(GLES32.GL_VENDOR);

		System.out.println("RTR : version : "+version);
		System.out.println("RTR : Shading Language Version : "+shadingLanguageVersion);
		System.out.println("RTR : renderer : "+renderer);
		System.out.println("RTR : vendor : "+vendor);

		oglInitialise();
	}

	@Override
	public void onSurfaceChanged(GL10 unused, int iWidth, int iHeight)
	{
		oglResize(iWidth, iHeight);
	}

	@Override
	public void onDrawFrame(GL10 unused)
	{
		if(bAnimate == true)
		{
			oglupdate();
		}
		oglDisplay();
	}

	//our custom methods

	private void oglInitialise()
	{
		/* vertex shader code */
		
		//define shader object
		vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		//write shader source code
		final String vertexShaderSourceCode = 
			String.format
			(	
				"#version 320 es" +
				"\n" +
				"precision mediump int;" +
				"in vec4 vPosition;" +
				"in vec3 vNormal;" +
				"uniform mat4 u_model_matrix;" +
				"uniform mat4 u_view_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				"uniform int u_singleTap;" +
				"uniform vec3 u_La_red;" +
				"uniform vec3 u_La_blue;" +
				"uniform vec3 u_Ld_red;" +
				"uniform vec3 u_Ld_blue;" +
				"uniform vec3 u_Ls_red;" +
				"uniform vec3 u_Ls_blue;" +
				"uniform vec4 u_light_position_red;" +
				"uniform vec4 u_light_position_blue;" +
				"uniform vec3 u_Ka;" +
				"uniform vec3 u_Kd;" +
				"uniform vec3 u_Ks;" +
				"uniform float shininess;" +
				"out vec3 phong_ads_light;" +
				"void main(void)" +
				"{" +
					"if (u_singleTap == 1)" +
					"{" +
						"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" +
						"mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" +
						"vec3 transformed_normal = normalize(normal_matrix * vNormal);" +
						"vec3 light_direction_red = normalize(vec3(u_light_position_red - eye_coordinates));" +
						"vec3 light_direction_blue = normalize(vec3(u_light_position_blue - eye_coordinates));" +
						"float tn_dot_LightDirection_red = max(dot(light_direction_red, transformed_normal), 0.0);" +
						"float tn_dot_LightDirection_blue = max(dot(light_direction_blue, transformed_normal), 0.0);" +
						"vec3 reflection_vector_red = reflect(-light_direction_red, transformed_normal);" +
						"vec3 reflection_vector_blue = reflect(-light_direction_blue, transformed_normal);" +
						"vec3 viewer_vector = normalize(vec3(-eye_coordinates.xyz));" +

						"vec3 ambient = (u_La_red * u_Ka) + (u_La_blue * u_Ka);" +
						"vec3 diffuse = (u_Ld_red * u_Kd * tn_dot_LightDirection_red) + ( u_Ld_blue * u_Kd * tn_dot_LightDirection_blue);" +
						"vec3 specular = (u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red, viewer_vector), 0.0), shininess)) + (u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue, viewer_vector), 0.0), shininess));" +

						"phong_ads_light = ambient + diffuse + specular;" +
					"}" +
					"else" +
					"{" +
						"phong_ads_light = vec3(1.0, 1.0, 1.0);" +
					"}" +
					"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
				"}"
			);

		//specify above source code to shader object
		GLES32.glShaderSource(	vertexShaderObject,
								vertexShaderSourceCode);

		//compile the vertex shader
		GLES32.glCompileShader(vertexShaderObject);

		//error checking
		/***Steps For Error Checking***/
		/*
			1.	Call glGetShaderiv(), and get the compile status of that object.
			2.	check that compile status, if it is false then shader has compilation error.
			3.	if(false) call again the glGetShaderiv() function and get the
				infoLogLength.
			4.	if(infoLogLength > 0) then call glGetShaderInfoLog() function to get the error
				information.
			5.	Print that obtained logs in file.
		*/
		int[] iShaderCompileStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null;
		
		GLES32.glGetShaderiv(	vertexShaderObject,
								GLES32.GL_COMPILE_STATUS,
								iShaderCompileStatus, 0);
		
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE);
		{
			GLES32.glGetShaderiv(	vertexShaderObject,
									GLES32.GL_INFO_LOG_LENGTH,
									iInfoLogLength, 0);
			
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("RTR : Vertex Shader Compilation error : "+szInfoLog);
				oglUninitialise();
				System.exit(0);
			}

		}

		/* fragment shader code */

		//define shader object
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		//write shader source code
		final String fragmentShaderSourceCode = 
			String.format
			(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"in vec3 phong_ads_light;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = vec4(phong_ads_light, 0.0);" +
				"}"
			);
		//specify above source code to shader object
		GLES32.glShaderSource(	fragmentShaderObject,
								fragmentShaderSourceCode);

		//compile the vertex shader
		GLES32.glCompileShader(fragmentShaderObject);

		//error checking
		/***Steps For Error Checking***/
		/*
			1.	Call glGetShaderiv(), and get the compile status of that object.
			2.	check that compile status, if it is false then shader has compilation error.
			3.	if(false) call again the glGetShaderiv() function and get the
				infoLogLength.
			4.	if(infoLogLength > 0) then call glGetShaderInfoLog() function to get the error
				information.
			5.	Print that obtained logs in file.
		*/
		iShaderCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(	fragmentShaderObject,
								GLES32.GL_COMPILE_STATUS,
								iShaderCompileStatus, 0);

		if(iShaderCompileStatus[0] != 0)
		{
			GLES32.glGetShaderiv(	fragmentShaderObject,
									GLES32.GL_INFO_LOG_LENGTH,
									iInfoLogLength, 0);

			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("RTR : Fragment Shader Compilation error : "+szInfoLog);
				oglUninitialise();
				System.exit(0);
			}
		}

		/* shader program code */

		//create shader program object
		shaderProgramObject = GLES32.glCreateProgram();

		//Attach Vertex shader
		GLES32.glAttachShader(	shaderProgramObject,
								vertexShaderObject);

		//Attach Fragment Shader
		GLES32.glAttachShader(	shaderProgramObject,
								fragmentShaderObject);

		//pre linking bonding to vertex attributes
		GLES32.glBindAttribLocation(	shaderProgramObject,
										GLESMacros.AMC_ATTRIBUTE_POSITION,
										"vPosition");
		GLES32.glBindAttribLocation(	shaderProgramObject,
										GLESMacros.AMC_ATTRIBUTE_NORMAL,
										"vNormal");

		//link the shader porgram
		GLES32.glLinkProgram(shaderProgramObject);

		//error checking
		/***Steps For Error Checking***/
		/*
			1.	Call glGetProgramiv(), and get the compile status of that object.
			2.	check that link status, if it is false then shader has compilation error.
			3.	if(false) call again the glGetProgramiv() function and get the
				infoLogLength.
			4.	if(infoLogLength > 0) then call glGetProgramInfoLog() function to get the error
				information.
			5.	Print that obtained logs in file.
		*/
		int[] iProgramLinkStatus = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetProgramiv(	shaderProgramObject,
								GLES32.GL_LINK_STATUS,
								iProgramLinkStatus, 0);

		if(iProgramLinkStatus[0] != 0)
		{
			GLES32.glGetProgramiv(	shaderProgramObject,
									GLES32.GL_INFO_LOG_LENGTH,
									iInfoLogLength, 0);

			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetProgramInfoLog(	shaderProgramObject);
				System.out.println("RTR : Shader program link error : "+szInfoLog);
				oglUninitialise();
				System.exit(0);
			}
		}

		//post linking retriving uniform location
		model_uniform 				= GLES32.glGetUniformLocation(shaderProgramObject, "u_model_matrix");
		view_uniform 				= GLES32.glGetUniformLocation(shaderProgramObject, "u_view_matrix");
		projection_uniform 			= GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");
		singleTap_Uniform	 		= GLES32.glGetUniformLocation(shaderProgramObject, "u_singleTap");
		
		La_uniform_red 				= GLES32.glGetUniformLocation(shaderProgramObject, "u_La_red");
		La_uniform_blue				= GLES32.glGetUniformLocation(shaderProgramObject, "u_La_blue");
		Ld_uniform_red 				= GLES32.glGetUniformLocation(shaderProgramObject, "u_Ld_red");
		Ld_uniform_blue 			= GLES32.glGetUniformLocation(shaderProgramObject, "u_Ld_blue");
		Ls_uniform_red 				= GLES32.glGetUniformLocation(shaderProgramObject, "u_Ls_red");
		Ls_uniform_blue 			= GLES32.glGetUniformLocation(shaderProgramObject, "u_Ls_blue");
		
		Ka_uniform 					= GLES32.glGetUniformLocation(shaderProgramObject, "u_Ka");
		Kd_uniform 					= GLES32.glGetUniformLocation(shaderProgramObject, "u_Kd");
		Ks_uniform 					= GLES32.glGetUniformLocation(shaderProgramObject, "u_Ks");
		shininess_uniform 			= GLES32.glGetUniformLocation(shaderProgramObject, "shininess");

		lightPosition_uniform_red 	= GLES32.glGetUniformLocation(shaderProgramObject, "u_light_position_red");
		lightPosition_uniform_blue 	= GLES32.glGetUniformLocation(shaderProgramObject, "u_light_position_blue");

		//triangle vertices declaration
		final float[] pyramidVertices = new float[]
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

		final float[] pyramidNormals = new float[]
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

		/* Triangle */

		/* Position */

		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_pyramid, 0);
		GLES32.glBindVertexArray(vao_pyramid[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_pyramid_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_pyramid_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_pyramid_position = ByteBuffer.allocateDirect(pyramidVertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_pyramid_position.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_pyramid = byteBuffer_pyramid_position.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_pyramid.put(pyramidVertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_pyramid.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								pyramidVertices.length * 4,
								positionBuffer_pyramid,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		/* Color */
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_pyramid_normal, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_pyramid_normal[0]);

		//steps to make triangleColours array as buffer compatible for glBufferData().

		//step 1:	allocate buffer from native memory directly
		ByteBuffer byteBuffer_pyramid_normal = ByteBuffer.allocateDirect(pyramidNormals.length * 4);

		//step 2:	set the byte order as native byte order
		byteBuffer_pyramid_normal.order(ByteOrder.nativeOrder());

		//step 3:	create a float type buffer of our byte type buffer
		FloatBuffer normalBuffer_pyramid = byteBuffer_pyramid_normal.asFloatBuffer();

		//step 4:	now put the array in COOKED buffer
		normalBuffer_pyramid.put(pyramidNormals);

		//step 5:	set the starting position as 0th position
		normalBuffer_pyramid.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								pyramidNormals.length * 4,
								normalBuffer_pyramid,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_NORMAL,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		//clear
		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		//depth
		//GLES32.glClearDepth(1.0f);
		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		//make orthograhic projection matrix a identity matrix
		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
	}

	private void oglResize(int iWidth, int iHeight)
	{
		if(iHeight <= 0)
		{
			iHeight = 1;
		}

		GLES32.glViewport(0, 0, iWidth, iHeight);

		Matrix.perspectiveM(	perspectiveProjectionMatrix,
								0,
								45.0f,
								((float)iWidth / (float)iHeight),
								0.1f,
								100.0f);
	}

	private void oglupdate()
	{
		rotation_angle_pyramid = rotation_angle_pyramid + 1.0f;
		if (rotation_angle_pyramid >= 360.0f)
		{
			rotation_angle_pyramid = 0.0f;
		}
	}

	private void oglDisplay()
	{
		//code
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);
		GLES32.glUseProgram(shaderProgramObject);

		//declaration of metrices
		float[] modelMatrix = new float[16];
		float[] viewMatrix = new float[16];
		float[] projectionMatrix = new float[16];
		float[] translationMatrix = new float[16];
		float[] rotationMatrix = new float[16];

		/* Triangle */

		//init above metrices to identity
		Matrix.setIdentityM(modelMatrix, 0);
		Matrix.setIdentityM(viewMatrix, 0);
		Matrix.setIdentityM(projectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.setIdentityM(rotationMatrix, 0);

		//do necessary matrix multiplication
		Matrix.translateM(	translationMatrix, 0, 
							0.0f,
							0.0f,
							-4.0f);
		//static void rotateM(float[] m, int mOffset, float a, float x, float y, float z)
		Matrix.setRotateM(	rotationMatrix, 0,
							rotation_angle_pyramid,
							0.0f,
							1.0f,
							0.0f);
		
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							translationMatrix, 0);

		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							rotationMatrix, 0);

		Matrix.multiplyMM(	projectionMatrix, 0,
							projectionMatrix, 0,
							perspectiveProjectionMatrix, 0);

		//send this data to shader
		GLES32.glUniformMatrix4fv(	model_uniform,
									1,
									false,
									modelMatrix, 0);
		GLES32.glUniformMatrix4fv(	view_uniform,
									1,
									false,
									viewMatrix, 0);
		GLES32.glUniformMatrix4fv(	projection_uniform,
									1,
									false,
									projectionMatrix, 0);

		if (bLight)
		{
			//notify shader that we pressed the "L" key
			GLES32.glUniform1i(singleTap_Uniform, 1);
			//send red light intensity
			GLES32.glUniform3fv(La_uniform_red, 1, Ambient_red, 0);
			GLES32.glUniform3fv(Ld_uniform_red, 1, Diffuse_red, 0);
			GLES32.glUniform3fv(Ls_uniform_red, 1, Specular_red, 0);
			//send blue light intensity
			GLES32.glUniform3fv(La_uniform_blue, 1, Ambient_blue, 0);
			GLES32.glUniform3fv(Ld_uniform_blue, 1, Diffuse_blue, 0);
			GLES32.glUniform3fv(Ls_uniform_blue, 1, Specular_blue, 0);
			//send coeff. of material's reflectivity
			GLES32.glUniform3fv(Ka_uniform, 1, MaterialAmbient, 0);
			GLES32.glUniform3fv(Kd_uniform, 1, MaterialDiffuse, 0);
			GLES32.glUniform3fv(Ks_uniform, 1, MaterialSpecular, 0);
			//shininess
			GLES32.glUniform1f(shininess_uniform, MaterialShininess);
			//send light position
			GLES32.glUniform4fv(lightPosition_uniform_red, 1, Position_red, 0);
			GLES32.glUniform4fv(lightPosition_uniform_blue, 1, Position_blue, 0);
		}
		else
		{
			//notify shader that we pressed the "L" key
			GLES32.glUniform1i(singleTap_Uniform, 0);
		}

		//bind with vao
		GLES32.glBindVertexArray(vao_pyramid[0]);

		//draw scene
		GLES32.glDrawArrays(	GLES32.GL_TRIANGLES,
								0,
								12);
		
		//unbind vao
		GLES32.glBindVertexArray(0);
		
		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private void oglUninitialise()
	{
		//code
		if(vao_pyramid[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_pyramid, 0);
			vao_pyramid[0] = 0;
		}

		if(vbo_pyramid_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_pyramid_position, 0);
			vbo_pyramid_position[0] = 0;
		}

		if(vbo_pyramid_normal[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_pyramid_normal, 0);
			vbo_pyramid_normal[0] = 0;
		}

		if(shaderProgramObject != 0)
		{
			int[] shaderCount = new int[1];
			int shaderNumber;

			GLES32.glUseProgram(shaderProgramObject);

			//ask program how many shaders are attached
			GLES32.glGetProgramiv(	shaderProgramObject,
									GLES32.GL_ATTACHED_SHADERS,
									shaderCount, 0);

			int[] shaders = new int[shaderCount[0]];

			if(shaderCount[0] != 0)
			{
				GLES32.glGetAttachedShaders(	shaderProgramObject,
												shaderCount[0],
												shaderCount, 0,
												shaders, 0);

				for(shaderNumber = 0; shaderNumber < shaderCount[0]; shaderNumber++)
				{
					//detach shaders
					GLES32.glDetachShader(	shaderProgramObject,
											shaders[shaderNumber]);

					//delete shaders
					GLES32.glDeleteShader(shaders[shaderNumber]);

					shaders[shaderNumber] = 0;
				}
			} 
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject = 0;
			GLES32.glUseProgram(0);
		}
	}
}
