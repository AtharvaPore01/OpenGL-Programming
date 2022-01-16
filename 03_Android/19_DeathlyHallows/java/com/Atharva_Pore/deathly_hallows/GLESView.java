package com.Atharva_Pore.deathly_hallows;

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
	private int[] vao_triangle = new int[1];
	private int[] vao_circle = new int[1];
	private int[] vao_line = new int[1];

	private int[] vbo_triangle = new int[1];
	private int[] vbo_circle = new int[1];
	private int[] vbo_line = new int[1];

	private int mvpUniform;

	private float[] perspectiveProjectionMatrix = new float[16];		//16 because it is 4 x 4 matrix.

	//for distance finding and semi-perimeter
	double a = 0.0f, b = 0.0f, c = 0.0f;
	double Perimeter = 0.0f;
	double x1 = 0.0f;
	double x2 = -1.0f;
	double x3 = 1.0f;
	double y1 = 1.0f;
	double y2 = -1.0f;
	double y3 = -1.0f;

	//for area of triangle
	double AreaOfTriangle = 0.0f;
	//for circle
	float x_center = 0.0f;
	float y_center = 0.0f;
	float radius = 0.0f;

	//initial position of triangle, circle, line
	float x_triangle = 3.0f;
	float y_triangle = -3.0f;
	float x_circle = -3.0f;
	float y_circle = -3.0f;
	float y_line = 3.0f;
	float rotationAngle;
	boolean bCircle = false;
	boolean bLine = false;

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
		oglUpdate();
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
				"in vec4 vPosition;" +
				"uniform mat4 u_mvp_matirx;" +
				"void main(void)" +
				"{" +
				"gl_Position = u_mvp_matirx * vPosition;" +
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
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
				"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" +
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
		mvpUniform = GLES32.glGetUniformLocation(	shaderProgramObject,
													"u_mvp_matirx");

		//triangle vertices declaration
		final float[] triangleVertices = new float[]
		{
			0.0f, 1.0f, 0.0f,
			-1.0f, -1.0f, 0.0f,
			-1.0f, -1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,
			0.0f, 1.0f, 0.0f
		};

		final float[] lineVertices = new float[]
		{
			0.0f, 1.0f, 0.0f,
			0.0f, -1.0f, 0.0f
		};

		//****triangle
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_triangle, 0);
		GLES32.glBindVertexArray(vao_triangle[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_triangle, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_triangle[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_triangle = ByteBuffer.allocateDirect(triangleVertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_triangle.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_triangle = byteBuffer_triangle.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_triangle.put(triangleVertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_triangle.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								triangleVertices.length * 4,
								positionBuffer_triangle,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);
		GLES32.glBindVertexArray(0);

		//****line
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_line, 0);
		GLES32.glBindVertexArray(vao_line[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_line, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_line[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_line = ByteBuffer.allocateDirect(lineVertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_line.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_line = byteBuffer_line.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_line.put(lineVertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_line.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								lineVertices.length * 4,
								positionBuffer_line,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);
		GLES32.glBindVertexArray(0);

		//****circle
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_circle, 0);
		GLES32.glBindVertexArray(vao_circle[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_circle, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_circle[0]);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								1 * 3 * 4,
								null,
								GLES32.GL_DYNAMIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
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

	private void oglDisplay()
	{
		//code
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);
		GLES32.glUseProgram(shaderProgramObject);

		//declaration of metrices
		float[] modelViewMatrix = new float[16];
		float[] modelViewProjectionMatrix = new float[16];
		float[] translationMatrix_line = new float[16];
		float[] translationMatrix_circle = new float[16];
		float[] translationMatrix_triangle = new float[16];
		float[] translationMatrix = new float[16];
		float[] rotationMatrix = new float[16];

		/* Triangle */
		//init above metrices to identity
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.setIdentityM(translationMatrix_line, 0);
		Matrix.setIdentityM(translationMatrix_triangle, 0);
		Matrix.setIdentityM(translationMatrix_circle, 0);
		Matrix.setIdentityM(rotationMatrix, 0);

		//do necessary matrix multiplication
		Matrix.translateM(	translationMatrix, 0, 
							0.0f,
							0.0f,
							-6.0f);
		Matrix.translateM(	translationMatrix_triangle, 0, 
							x_triangle,
							y_triangle,
							0.0f);
		Matrix.setRotateM(	rotationMatrix, 0,
							rotationAngle,
							0.0f,
							1.0f,
							0.0f);
		
		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							translationMatrix, 0);
		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							translationMatrix_triangle, 0);
		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							rotationMatrix, 0);

		Matrix.multiplyMM(	modelViewProjectionMatrix, 0,
							perspectiveProjectionMatrix, 0,
							modelViewMatrix, 0);

		//send necessary matrics to shaders in respective uniforms
		GLES32.glUniformMatrix4fv(	mvpUniform,
									1,
									false,
									modelViewProjectionMatrix, 0);

		deathlyHallowTriangle();
		if (x_triangle >= 0.0f && y_triangle <= 0.0f)
		{
			y_triangle = y_triangle + 0.01f;
			x_triangle = x_triangle - 0.01f;
			if (y_triangle > 0.0f)
			{
				bCircle = true;
			}
		}
		
		if(bCircle == true)
		{
			//init above metrices to identity
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			//Matrix.setIdentityM(translationMatrix_circle, 0);
			//Matrix.setIdentityM(rotationMatrix, 0);

			//do necessary matrix multiplication
			Matrix.translateM(	translationMatrix, 0, 
								0.0f,
								0.0f,
								-6.0f);
			Matrix.translateM(	translationMatrix_circle, 0, 
								x_circle,
								y_circle,
								0.0f);
			Matrix.setRotateM(	rotationMatrix, 0,
								rotationAngle,
								0.0f,
								1.0f,
								0.0f);
			
			Matrix.multiplyMM(	modelViewMatrix, 0,
								modelViewMatrix, 0,
								translationMatrix, 0);
			Matrix.multiplyMM(	modelViewMatrix, 0,
								modelViewMatrix, 0,
								translationMatrix_circle, 0);
			Matrix.multiplyMM(	modelViewMatrix, 0,
								modelViewMatrix, 0,
								rotationMatrix, 0);

			Matrix.multiplyMM(	modelViewProjectionMatrix, 0,
								perspectiveProjectionMatrix, 0,
								modelViewMatrix, 0);

			//send necessary matrics to shaders in respective uniforms
			GLES32.glUniformMatrix4fv(	mvpUniform,
										1,
										false,
										modelViewProjectionMatrix, 0);

			deathlyHallowsCircle();
			if ((x_circle <= 0.0f && y_circle <= 0.0f))
			{
				y_circle = y_circle + 0.01f;
				x_circle = x_circle + 0.01f;
				if (x_circle > 0.0f)
				{
					bLine = true;
				}
			}
		}

		if(bLine == true)
		{
			//init above metrices to identity
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			//Matrix.setIdentityM(translationMatrix_line, 0);
			//Matrix.setIdentityM(rotationMatrix, 0);

			//do necessary matrix multiplication
			Matrix.translateM(	translationMatrix, 0, 
								0.0f,
								0.0f,
								-6.0f);
			Matrix.translateM(	translationMatrix_line, 0, 
								0.0f,
								y_line,
								0.0f);
			Matrix.setRotateM(	rotationMatrix, 0,
								rotationAngle,
								0.0f,
								1.0f,
								0.0f);
			
			Matrix.multiplyMM(	modelViewMatrix, 0,
								modelViewMatrix, 0,
								translationMatrix, 0);
			Matrix.multiplyMM(	modelViewMatrix, 0,
								modelViewMatrix, 0,
								translationMatrix_line, 0);
			Matrix.multiplyMM(	modelViewMatrix, 0,
								modelViewMatrix, 0,
								rotationMatrix, 0);

			Matrix.multiplyMM(	modelViewProjectionMatrix, 0,
								perspectiveProjectionMatrix, 0,
								modelViewMatrix, 0);

			//send necessary matrics to shaders in respective uniforms
			GLES32.glUniformMatrix4fv(	mvpUniform,
										1,
										false,
										modelViewProjectionMatrix, 0);
			deathlyHallowsLine();
			if ((y_line >= 0.0f))
			{
				y_line = y_line - 0.01f;
			}
		}

		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private void oglUpdate()
	{
		//code
		rotationAngle = rotationAngle + 1.0f;
		if (rotationAngle >= 360.0f)
		{
			rotationAngle = 0.0f;
		}
	}

	private void calculateSemiPerimeter()
	{
		//code
		a = Math.sqrt((Math.pow((x2 - x1), 2) + Math.pow((y2 - y1), 2)));
		b = Math.sqrt((Math.pow((x3 - x2), 2) + Math.pow((y3 - y2), 2)));
		c = Math.sqrt((Math.pow((x1 - x3), 2) + Math.pow((y1 - y3), 2)));
		
		//Semi Perimeter
		Perimeter = (a + b + c) / 2;
	}

	private void calculateAreaOfTriangle()
	{
		//code
		AreaOfTriangle = Math.sqrt(Perimeter * (Perimeter - a) * (Perimeter - b) * (Perimeter - c));
	}

	private void calculateRadius()
	{
		//code
		radius = (float)(AreaOfTriangle / Perimeter);
	}

	private void calculateCenterOfTheCircle()
	{
		//code
		x_center = (float)(((a * x3) + (b * x1) + (c * x2)) / (a + b + c));
		y_center = (float)(((a * (y3)) + (b * (y1)) + (c * (y2))) / (a + b + c));
	}

	private void deathlyHallowsCircle()
	{
		float[] circleVertices = new float[3];

		//code
		//bind with vao
		GLES32.glBindVertexArray(vao_circle[0]);
		for (float angle = 0.0f; angle < (2.0f * Math.PI); angle = angle + 0.01f)
		{
			circleVertices[0] = (float)((Math.cos(angle) * radius) + x_center);
			circleVertices[1] = (float)((Math.sin(angle) * radius) + y_center);
			circleVertices[2] = 0.0f;

			//vertices
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_circle[0]);
			//step 1: allocate the buffer directly from native memory(unmanaged memory)
			ByteBuffer byteBuffer_circle = ByteBuffer.allocateDirect(circleVertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

			//step 2: arrange the byte order of buffer in native byte order
			byteBuffer_circle.order(ByteOrder.nativeOrder());

			//step 3: create the float type buffer and convert our byte type buffer in float
			FloatBuffer positionBuffer_circle = byteBuffer_circle.asFloatBuffer();

			//step 4: now put your array in this COOKED buffer.
			positionBuffer_circle.put(circleVertices);

			//step 5: set the array at 0th position of the buffer.
			positionBuffer_circle.position(0);
			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, circleVertices.length * 4, positionBuffer_circle, GLES32.GL_DYNAMIC_DRAW);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			//draw scene
			//glPointSize(1.5f);
			GLES32.glDrawArrays(GLES32.GL_POINTS, 0, 1);
			//glDrawArrays(GL_LINE_LOOP, 0, 10);
		}

		//unbind vao
		GLES32.glBindVertexArray(0);
	}

	private void deathlyHallowsLine()
	{
		//bind with vao
		GLES32.glBindVertexArray(vao_line[0]);

		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 2);

		GLES32.glBindVertexArray(0);
	}

	private void deathlyHallowTriangle()
	{
		//code
		calculateSemiPerimeter();
		calculateAreaOfTriangle();
		calculateRadius();
		calculateCenterOfTheCircle();

		//bind with vao
		GLES32.glBindVertexArray(vao_triangle[0]);

		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 2, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 4, 2);

		//unbind vao
		GLES32.glBindVertexArray(0);
	}

	private void oglUninitialise()
	{
		//code
		if(vao_line[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_line, 0);
			vao_line[0] = 0;
		}
		if(vao_circle[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_circle, 0);
			vao_circle[0] = 0;
		}
		if(vao_triangle[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_triangle, 0);
			vao_triangle[0] = 0;
		}

		if(vbo_line[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_line, 0);
			vbo_line[0] = 0;
		}
		if(vbo_circle[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_circle, 0);
			vbo_circle[0] = 0;
		}
		if(vbo_triangle[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_triangle, 0);
			vbo_triangle[0] = 0;
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
