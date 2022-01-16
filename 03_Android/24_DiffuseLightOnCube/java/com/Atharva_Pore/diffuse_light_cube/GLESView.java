package com.Atharva_Pore.diffuse_light_cube;

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
	private int[] vao_cube = new int[1];
	private int[] vbo_cube_position = new int[1];
	private int[] vbo_cube_normal = new int[1];

	private int mvpUniform;
	private int modelViewUniform;
	private int projectionUniform;
	private int Ld_Uniform;
	private int Kd_Uniform;
	private int lightPosition_Uniform;
	private int singleTap_Uniform;

	private float[] perspectiveProjectionMatrix = new float[16];		//16 because it is 4 x 4 matrix.

	//Rotation variables
	float rotation_angle_cube = 0.0f;

	//flags
	boolean bAnimate = false;
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
		/*
			setTextColor(Color.rgb(255, 0, 0));
			setGravity(Gravity.CENTER);
			setText("Double Tap");
		*/
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
				"uniform mat4 u_mv_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				"uniform int u_singleTap;" +
				"uniform vec3 u_Ld;" +
				"uniform vec3 u_Kd;" +
				"uniform vec4 u_light_position;" +
				"out vec3 diffuse_color;" +
				"void main(void)" +
				"{" +
					"if(u_singleTap == 1)" +
					"{" +
						"vec4 eye_coordinates = u_mv_matrix * vPosition;" +
						"mat3 normal_matrix = mat3(transpose(inverse(u_mv_matrix)));" +
						"vec3 transformed_matrix = normalize(normal_matrix * vNormal);" +
						"vec3 s = normalize(vec3(u_light_position - eye_coordinates));" +
						"diffuse_color = u_Ld * u_Kd * max(dot(s, transformed_matrix), 0.0);" +
					"}" +
					"gl_Position = u_projection_matrix * u_mv_matrix * vPosition;" +
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
				"in vec3 diffuse_color;" +
				"out vec4 FragColor;" +
				"uniform int u_singleTap;" +
				"void main(void)" +
				"{" +
					"if(u_singleTap == 1)" + 
					"{" +
						"FragColor = vec4(diffuse_color, 1.0);" +
					"}" +
					"else" +
					"{" +
						"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" +
					"}" +
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
		modelViewUniform 		= GLES32.glGetUniformLocation(shaderProgramObject, "u_mv_matrix");
		projectionUniform 		= GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");
		singleTap_Uniform 		= GLES32.glGetUniformLocation(shaderProgramObject, "u_singleTap");
		Ld_Uniform 				= GLES32.glGetUniformLocation(shaderProgramObject, "u_Ld");
		Kd_Uniform 				= GLES32.glGetUniformLocation(shaderProgramObject, "u_Kd");
		lightPosition_Uniform 	= GLES32.glGetUniformLocation(shaderProgramObject, "u_light_position");

		System.out.println("RTR : Post Linking Done");
		final float[] cubeVertices = new float[]
		{
			1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
	
			1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
	
			1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f, 
			-1.0f, -1.0f, 1.0f, 
			1.0f, -1.0f, 1.0f,
	
			1.0f, 1.0f, -1.0f, 
			-1.0f, 1.0f, -1.0f,
			- 1.0f, -1.0f, -1.0f,
			1.0f, -1.0f, -1.0f,
	
			1.0f, 1.0f, -1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, -1.0f,
	
			-1.0f, 1.0f, -1.0f, 
			-1.0f, 1.0f, 1.0f, 
			-1.0f, -1.0f, 1.0f,
			-1.0f, -1.0f, -1.0f
		};

		final float[] cubeNormal = new float[]
		{
			
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,

			0.0f, -1.0f, 0.0f,
			0.0f, -1.0f, 0.0f,
			0.0f, -1.0f, 0.0f,
			0.0f, -1.0f, 0.0f,

			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,

			0.0f, 0.0f, -1.0f,
			0.0f, 0.0f, -1.0f,
			0.0f, 0.0f, -1.0f,
			0.0f, 0.0f, -1.0f,

			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,

			-1.0f, 0.0f, 0.0f,
			-1.0f, 0.0f, 0.0f,
			-1.0f, 0.0f, 0.0f,
			-1.0f, 0.0f, 0.0f
		
		};

		/* Cube */

		/* Position */
		for (int i = 0; i < 72; i++)
		{
			if (cubeVertices[i] == -1.0f)
			{
				cubeVertices[i] = cubeVertices[i] + 0.25f;
			}
			else if (cubeVertices[i] == 1.0f)
			{
				cubeVertices[i] = cubeVertices[i] - 0.25f;
			}
		}
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_cube, 0);
		GLES32.glBindVertexArray(vao_cube[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_cube_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_cube_position[0]);

		//now from here below we will do 5 steps to change our rectangleVertices array in buffer which is compatible to give to glBufferData().

		//step 1:	allocate buffer directly from native memory.
		ByteBuffer byteBuffer_cube_position = ByteBuffer.allocateDirect(cubeVertices.length * 4);

		//step 2: 	change the byte order to native byte order.
		byteBuffer_cube_position.order(ByteOrder.nativeOrder());

		//step 3:	Create float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_cube = byteBuffer_cube_position.asFloatBuffer();

		//step 4:	now to the rectangleVertices array in this COOKED Buffer
		positionBuffer_cube.put(cubeVertices);

		//step 5:	set the array at the 0th position
		positionBuffer_cube.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								cubeVertices.length * 4,
								positionBuffer_cube,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);

		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		/* Normal */
		//create and bind buffer
		GLES32.glGenBuffers(1, vbo_cube_normal, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_cube_normal[0]);

		//convert the array in glBufferData() compatibile buffer.

		//step 1: allocate the buffer directly from native memory
		ByteBuffer byteBuffer_cube_normal = ByteBuffer.allocateDirect(cubeNormal.length * 4);

		//step 2: change the byte order to the native byte order
		byteBuffer_cube_normal.order(ByteOrder.nativeOrder());

		//step 3: make the byte array as float array
		FloatBuffer normalBuffer_cube = byteBuffer_cube_normal.asFloatBuffer();

		//step 4: put our array in that COOKED buffer.
		normalBuffer_cube.put(cubeNormal);

		//step 5: set the array position as 0th position
		normalBuffer_cube.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								cubeNormal.length * 4,
								normalBuffer_cube,
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
		//System.out.println("RTR : Done With vbo_cube_normal and vbo_cube_position");
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
		rotation_angle_cube = rotation_angle_cube + 1.0f;
		if (rotation_angle_cube >= 360.0f)
		{
			rotation_angle_cube = 0.0f;
		}
	}

	private void oglDisplay()
	{
		//code
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);
		GLES32.glUseProgram(shaderProgramObject);

		//declaration of metrices
		float[] modelViewMatrix = new float[16];
		float[] projectionMatrix = new float[16];
		float[] translationMatrix = new float[16];
		float[] rotationMatrix_x = new float[16];
		float[] rotationMatrix_y = new float[16];
		float[] rotationMatrix_z = new float[16];

		/* Rectangle */
		
		//make all matrices indentity.
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(projectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.setIdentityM(rotationMatrix_x, 0);
		Matrix.setIdentityM(rotationMatrix_y, 0);
		Matrix.setIdentityM(rotationMatrix_z, 0);

		//do matrix multiplication
		Matrix.translateM(	translationMatrix, 0,
							0.0f, 
							0.0f, 
							-4.0f);

		//static void setRotateM(float[] rm, int rmOffset, float a, float x, float y, float z)
		Matrix.setRotateM(	rotationMatrix_x, 0,
							rotation_angle_cube,
							1.0f,							//X
							0.0f,
							0.0f);

		Matrix.setRotateM(	rotationMatrix_y, 0,
							rotation_angle_cube,
							0.0f,
							1.0f,							//Y
							0.0f);

		Matrix.setRotateM(	rotationMatrix_z, 0,
							rotation_angle_cube,
							0.0f,
							0.0f,
							1.0f);							//Z

		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							translationMatrix, 0);

		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							rotationMatrix_x, 0);

		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							rotationMatrix_y, 0);

		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							rotationMatrix_z, 0);

		Matrix.multiplyMM(	projectionMatrix, 0,
							projectionMatrix, 0,
							perspectiveProjectionMatrix, 0);

		//send this data to shader
		GLES32.glUniformMatrix4fv(	modelViewUniform,
									1,
									false,
									modelViewMatrix, 0);
		GLES32.glUniformMatrix4fv(	projectionUniform,
									1,
									false,
									projectionMatrix, 0);

		//if lighting is enabled then do following steps
		if(bLight == true)
		{
			//send the message to shader that "L" key pressed
			GLES32.glUniform1i(singleTap_Uniform, 1);
			//send intensity(L) of diffuse light to shader
			GLES32.glUniform3f(Ld_Uniform, 1.0f, 1.0f, 1.0f);
			//send coefficient of material difuse reflectance
			GLES32.glUniform3f(Kd_Uniform, 0.5f, 0.5f, 0.5f);
			//send light position
			GLES32.glUniform4f(lightPosition_Uniform, 0.0f, 0.0f, 2.0f, 1.0f);
		}
		else
		{
			//send the message to shader that "L" key isn't pressed
			GLES32.glUniform1i(singleTap_Uniform, 0);
		}

		//bind vao 
		GLES32.glBindVertexArray(vao_cube[0]);

		//draw scene
		GLES32.glDrawArrays(	GLES32.GL_TRIANGLE_FAN,
								0,
								4);
		GLES32.glDrawArrays(	GLES32.GL_TRIANGLE_FAN,
								4,
								4);
		GLES32.glDrawArrays(	GLES32.GL_TRIANGLE_FAN,
								8,
								4);
		GLES32.glDrawArrays(	GLES32.GL_TRIANGLE_FAN,
								12,
								4);
		GLES32.glDrawArrays(	GLES32.GL_TRIANGLE_FAN,
								16,
								4);
		GLES32.glDrawArrays(	GLES32.GL_TRIANGLE_FAN,
								20,
								4);
		//unbind vao 
		GLES32.glBindVertexArray(0);
		
		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private void oglUninitialise()
	{
		//code
		if(vao_cube[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_cube, 0);
			vao_cube[0] = 0;
		}

		if(vbo_cube_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_cube_position, 0);
			vbo_cube_position[0] = 0;
		}

		if(vbo_cube_normal[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_cube_normal, 0);
			vbo_cube_normal[0] = 0;
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
