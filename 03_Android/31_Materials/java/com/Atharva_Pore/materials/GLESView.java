package com.Atharva_Pore.materials;

//programmable related (OpenGL Related) packages
import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import javax.microedition.khronos.opengles.GL10;			//for basic features of openGL-ES
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;									//nio = Non Blocking I/O OR Native I/O,	For Opengl Buffers.
import java.nio.ByteOrder;									//for arranginf byte order of the buffer in native byte order(Little Indian / Big Indian)
import java.nio.FloatBuffer;								//to create float type buffer.
import java.nio.ShortBuffer;								//for sphere.

import android.opengl.Matrix;								//for matrix mathematics.

//standard packages
import android.content.Context; 							//for Context drawingContext class
import android.graphics.Color;								//for Color class
import android.view.Gravity;								//for Gravity class	
import android.view.MotionEvent;							//for MotionEvent			
import android.view.GestureDetector;						//for GestureDetector
import android.view.GestureDetector.OnGestureListener;		//for OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener;	//for OnDoubleTapListener

import java.lang.Math;										//for maths (math.h)

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
	private int[] vao_sphere = new int[1];
    private int[] vbo_sphere_position = new int[1];
    private int[] vbo_sphere_normal = new int[1];
    private int[] vbo_sphere_element = new int[1];

	private int model_uniform;
	private int view_uniform;
	private int projection_uniform;
	
	private int La_uniform;
	private int Ld_uniform;
	private int Ls_uniform;
	private int lightPosition_uniform;

	private int Ka_uniform;
	private int Kd_uniform;
	private int Ks_uniform;
	private int shininess_uniform;
	private int singleTap_Uniform;

	private float[] perspectiveProjectionMatrix = new float[16];		//16 because it is 4 x 4 matrix.

	//sphere related variables
	private int numVertices;
	private int numElements;

	//light values
	final float[] LightAmbient	 	= new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
	final float[] LightDiffuse	 	= new float[] { 0.5f, 0.2f, 0.7f, 1.0f };
	final float[] LightSpecular	 	= new float[] { 0.7f, 0.7f, 0.7f, 1.0f };
	final float[] LightPosition	 	= new float[] { 100.0f, 100.0f, 100.0f, 1.0f };	
	
	private class material_array
	{
		float[] MaterialAmbient = new float[4];
		float[] MaterialDiffuse = new float[4];
		float[] MaterialSpecular = new float[4];
		float MaterialShininess;
	}

	material_array[] mat_arr = new material_array[24];

	private int iCount = 0;

	private float LightAngle = 0.0f;
	private int giWindowWidth = 0;
	private int giWindowHeight = 0;

	//flags
	boolean bLight = false;

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
		
		iCount = iCount + 1;
		if(iCount > 3)
		{
			iCount = 0;
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
		oglupdate();
		oglDisplay();
	}

	//our custom methods

	private void oglInitialise()
	{
		
		//sphere related declaration
		float[] sphere_vertices	=	new float[1146];
        float[] sphere_normals	=	new float[1146];
        float[] sphere_textures	=	new float[764];
        short[] sphere_elements	=	new short[2280];



        //sphere object creation
		Sphere sphere = new Sphere();

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
				"uniform mat4 u_mvp_matrix;" +
				"uniform int u_singleTap;" +
				"uniform vec4 u_light_position;" +
				"out vec3 t_norm;" +
				"out vec3 light_direction;" +
				"out vec3 viewer_vector;" +
				"void main(void)" +
				"{" +
					"if (u_singleTap == 1)" +
					"{" +
						"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" +
						"mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" +
						"t_norm = normal_matrix * vNormal;" +
						"light_direction = vec3(u_light_position - eye_coordinates);" +
						"viewer_vector = vec3(-eye_coordinates);" +
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
				"in vec3 t_norm;" +
				"in vec3 light_direction;" +
				"in vec3 viewer_vector;" +
				"uniform int u_singleTap;" +
				"uniform vec3 u_La;" +
				"uniform vec3 u_Ld;" +
				"uniform vec3 u_Ls;" +
				"uniform vec3 u_Ka;" +
				"uniform vec3 u_Kd;" +
				"uniform vec3 u_Ks;" +
				"uniform float shininess;" +

				"out vec3 phong_ads_light;" +
				"out vec4 FragColor;" +

				"void main(void)" +
				"{" +
					"if(u_singleTap == 1)" +
					"{" +
						"vec3 normalised_transformed_normal = normalize(t_norm);" +

						"vec3 normalised_light_direction = normalize(light_direction);" +

						"vec3 normalised_viewer_vector = normalize(viewer_vector);" +

						"vec3 reflection_vector = reflect(-normalised_light_direction, normalised_transformed_normal);" +

						"float tn_dot_LightDirection= max(dot(normalised_light_direction, normalised_transformed_normal), 0.0);" +

						"vec3 ambient = (u_La * u_Ka);" +
						"vec3 diffuse = (u_Ld * u_Kd * tn_dot_LightDirection);" +
						"vec3 specular = (u_Ls * u_Ks * pow(max(dot(reflection_vector, normalised_viewer_vector), 0.0), shininess));" +
						
						"phong_ads_light = ambient + diffuse + specular;" +
					"}" +
					"else" +
					"{" +
						"phong_ads_light = vec3(1.0, 1.0, 1.0);" +
					"}" +
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
		
		model_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_model_matrix");
		view_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_view_matrix");
		projection_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");
		singleTap_Uniform 		= GLES32.glGetUniformLocation(shaderProgramObject, "u_singleTap");

		La_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_La");
		Ld_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_Ld");
		Ls_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_Ls");

		Ka_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_Ka");
		Kd_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_Kd");
		Ks_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_Ks");

		shininess_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "shininess");
		
		lightPosition_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_light_position");

		sphere.getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);
        numVertices = sphere.getNumberOfSphereVertices();
        numElements = sphere.getNumberOfSphereElements();

		/* Position */
		//create and bind vao
        GLES32.glGenVertexArrays(1, vao_sphere, 0);
        GLES32.glBindVertexArray(vao_sphere[0]);
        
        //create and bind vbo
        GLES32.glGenBuffers(1, vbo_sphere_position, 0);
        GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_sphere_position[0]);

		//now from here below we will do 5 steps to change our rectangleVertices array in buffer which is compatible to give to glBufferData().

		//step 1:	allocate buffer directly from native memory.
		ByteBuffer byteBuffer_vertices = ByteBuffer.allocateDirect(sphere_vertices.length * 4);

		//step 2: 	change the byte order to native byte order.
		byteBuffer_vertices.order(ByteOrder.nativeOrder());

		//step 3:	Create float type buffer and convert our byte type buffer in float
		FloatBuffer verticesBuffer = byteBuffer_vertices.asFloatBuffer();

		//step 4:	now to the rectangleVertices array in this COOKED Buffer
		verticesBuffer.put(sphere_vertices);

		//step 5:	set the array at the 0th position
		verticesBuffer.position(0);

		 GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
                            sphere_vertices.length * 4,
                            verticesBuffer,
                            GLES32.GL_STATIC_DRAW);
        
        GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
                                     3,
                                     GLES32.GL_FLOAT,
                                     false, 
                                     0, 
                                     0);

		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		/* Normal */
		// normal vbo
        GLES32.glGenBuffers(1,vbo_sphere_normal,0);
        GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_sphere_normal[0]);

		//convert the array in glBufferData() compatibile buffer.

		//step 1:	allocate buffer directly from native memory.
		ByteBuffer byteBuffer_normal = ByteBuffer.allocateDirect(sphere_normals.length * 4);

		//step 2: 	change the byte order to native byte order.
		byteBuffer_normal.order(ByteOrder.nativeOrder());

		//step 3:	Create float type buffer and convert our byte type buffer in float
		FloatBuffer normalsBuffer = byteBuffer_normal.asFloatBuffer();

		//step 4:	now to the rectangleVertices array in this COOKED Buffer
		normalsBuffer.put(sphere_normals);

		//step 5:	set the array at the 0th position
		normalsBuffer.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
                            	sphere_normals.length * 4,
                            	normalsBuffer,
                            	GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_NORMAL,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);

		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		// element vbo
        GLES32.glGenBuffers(1,vbo_sphere_element,0);
        GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,vbo_sphere_element[0]);

        //convert the array in glBufferData() compatibile buffer.

		//step 1:	allocate buffer directly from native memory.
		ByteBuffer byteBuffer_element = ByteBuffer.allocateDirect(sphere_elements.length * 4);

		//step 2: 	change the byte order to native byte order.
		byteBuffer_element.order(ByteOrder.nativeOrder());

		//step 3:	Create float type buffer and convert our byte type buffer in float
		ShortBuffer elementsBuffer = byteBuffer_element.asShortBuffer();

		//step 4:	now to the rectangleVertices array in this COOKED Buffer
		elementsBuffer.put(sphere_elements);

		//step 5:	set the array at the 0th position
		elementsBuffer.position(0);

		GLES32.glBufferData(	GLES32.GL_ELEMENT_ARRAY_BUFFER,
                            	sphere_elements.length * 2,
                           	 	elementsBuffer,
                           	 	GLES32.GL_STATIC_DRAW);

		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,0);

		GLES32.glBindVertexArray(0);

		//init material
		oglInitMaterial();

		//clear
		GLES32.glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

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

		giWindowWidth = iWidth;
		giWindowHeight = iHeight;

		Matrix.perspectiveM(	perspectiveProjectionMatrix,
								0,
								45.0f,
								((float)iWidth / (float)iHeight),
								0.1f,
								100.0f);

	}

	private void oglupdate()
	{
		//code
		LightAngle = LightAngle + 0.005f;
		if (LightAngle >= 360)
		{
			LightAngle = 0.0f;
		}
	}

	private void oglDisplay()
	{
		//code
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);
		GLES32.glUseProgram(shaderProgramObject);

		oglDraw24Spheres();

		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private void oglUninitialise()
	{
		//code
		 // destroy vao
        if(vao_sphere[0] != 0)
        {
            GLES32.glDeleteVertexArrays(1, vao_sphere, 0);
            vao_sphere[0]=0;
        }
        
        // destroy position vbo
        if(vbo_sphere_position[0] != 0)
        {
            GLES32.glDeleteBuffers(1, vbo_sphere_position, 0);
            vbo_sphere_position[0]=0;
        }
        
        // destroy normal vbo
        if(vbo_sphere_normal[0] != 0)
        {
            GLES32.glDeleteBuffers(1, vbo_sphere_normal, 0);
            vbo_sphere_normal[0]=0;
        }
        
        // destroy element vbo
        if(vbo_sphere_element[0] != 0)
        {
            GLES32.glDeleteBuffers(1, vbo_sphere_element, 0);
            vbo_sphere_element[0]=0;
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

	private void oglDraw24Spheres()
	{
		//variable declaration
		int i = 0;
		//declaration of metrices
		float[] modelMatrix = new float[16];
		float[] viewMatrix = new float[16];
		float[] projectionMatrix = new float[16];
		float[] translationMatrix = new float[16];
	
		for (i = 0; i < 24; i++)
		{
			GLES32.glViewport((i % 6) * giWindowWidth / 6, giWindowHeight - (i / 6 + 1) * giWindowHeight / 4, (int)giWindowWidth / 6, (int)giWindowHeight / 4);

			//perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)(giWindowWidth / 6) / (GLfloat)(giWindowHeight / 4), 0.1f, 100.0f);
			
			Matrix.perspectiveM(	perspectiveProjectionMatrix,
									0,
									45.0f,
									(float)(giWindowWidth / 6) / (float)(giWindowHeight / 4),
									0.1f,
									100.0f);


			//make all matrices indentity.
			Matrix.setIdentityM(modelMatrix, 0);
			Matrix.setIdentityM(viewMatrix, 0);
			Matrix.setIdentityM(projectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			
			Matrix.translateM(	translationMatrix, 0,
							0.0f, 
							0.0f, 
							-3.0f);

			Matrix.multiplyMM(	modelMatrix, 0,
								modelMatrix, 0,
								translationMatrix, 0);

			Matrix.multiplyMM(	projectionMatrix, 0,
								projectionMatrix, 0,
								perspectiveProjectionMatrix, 0);


			//do necessary matrix multiplication
			//modelMatrix = modelMatrix * translationMatrix;
			//projectionMatrix *= perspectiveProjectionMatrix;

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

			if(bLight == true)
			{
				//send the message to shader that "L" key pressed
				GLES32.glUniform1i(singleTap_Uniform, 1);
				//send light intensity
				GLES32.glUniform3fv(La_uniform, 1, LightAmbient, 0);
				GLES32.glUniform3fv(Ld_uniform, 1, LightDiffuse, 0);
				GLES32.glUniform3fv(Ls_uniform, 1, LightSpecular, 0);

				//send coeff. of material's reflectivity
				GLES32.glUniform3fv(Ka_uniform, 1, mat_arr[i].MaterialAmbient, 0);
				GLES32.glUniform3fv(Kd_uniform, 1, mat_arr[i].MaterialDiffuse, 0);
				GLES32.glUniform3fv(Ks_uniform, 1, mat_arr[i].MaterialSpecular, 0);
				//shininess
				GLES32.glUniform1f(shininess_uniform, mat_arr[i].MaterialShininess);
				//send light position
				if (iCount == 1)
				{
					LightPosition[0] = 0.0f;
					LightPosition[1] = GLESMacros.RADIUS * (float)(Math.cos((double)LightAngle));
					LightPosition[2] = GLESMacros.RADIUS * (float)(Math.sin((double)LightAngle));
					LightPosition[3] = 1.0f;
				}

				if (iCount == 2)
				{
					LightPosition[0] = GLESMacros.RADIUS * (float)(Math.cos((double)LightAngle));
					LightPosition[1] = 0.0f;
					LightPosition[2] = GLESMacros.RADIUS * (float)(Math.sin((double)LightAngle));
					LightPosition[3] = 1.0f;
				}

				if (iCount == 3)
				{
					LightPosition[0] = GLESMacros.RADIUS * (float)(Math.cos((double)LightAngle));
					LightPosition[1] = GLESMacros.RADIUS * (float)(Math.sin((double)LightAngle));
					LightPosition[2] = 0.0f;
					LightPosition[3] = 1.0f;
				}
				
				GLES32.glUniform4fv(lightPosition_uniform, 1, LightPosition, 0);
			}
			else
			{
				//send the message to shader that "L" key isn't pressed
				GLES32.glUniform1i(singleTap_Uniform, 0);
			}

			// bind vao
	        GLES32.glBindVertexArray(vao_sphere[0]);
	        
	        // *** draw, either by glDrawTriangles() or glDrawArrays() or glDrawElements()
	        GLES32.glBindBuffer(	GLES32.GL_ELEMENT_ARRAY_BUFFER, 
	        						vbo_sphere_element[0]);
	        GLES32.glDrawElements(	GLES32.GL_TRIANGLES, 
	        						numElements, 
	        						GLES32.GL_UNSIGNED_SHORT, 
	        						0);
	        // unbind vao
	        GLES32.glBindVertexArray(0);
		}
	}

	private void oglInitMaterial()
	{
		for(int i = 0; i < 24; i++)
		{
			mat_arr[i] = new material_array();
		}
		//code
		//emrald
		//mat_arr[0] = new material_array();
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
		//mat_arr[1] = new material_array();
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
		//mat_arr[2] = new material_array();
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
		//mat_arr[3] = new material_array();
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
		//mat_arr[4] = new material_array();
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
		//mat_arr[5] = new material_array();
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
		//mat_arr[6] = new material_array();
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
		//mat_arr[7] = new material_array();
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
		//mat_arr[8] = new material_array();
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
		//mat_arr[9] = new material_array();
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
	//	mat_arr[10] = new material_array();
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
		//mat_arr[11] = new material_array();
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
		//mat_arr[12] = new material_array();
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
		//mat_arr[13] = new material_array();
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
		//mat_arr[14] = new material_array();
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
		//mat_arr[15] = new material_array();
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
		//mat_arr[16] = new material_array();
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
		//mat_arr[17] = new material_array();
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
		//mat_arr[18] = new material_array();
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
		//mat_arr[19] = new material_array();
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
		//mat_arr[20] = new material_array();
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
		//mat_arr[21] = new material_array();
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
		//mat_arr[22] = new material_array();
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
		//mat_arr[23] = new material_array();
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
}
