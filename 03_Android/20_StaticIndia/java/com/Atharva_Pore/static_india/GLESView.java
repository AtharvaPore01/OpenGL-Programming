package com.Atharva_Pore.static_india;

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
	private int[] vao_I = new int[1];
	private int[] vao_N = new int[1];
	private int[] vao_D = new int[1];
	private int[] vao_i = new int[1];
	private int[] vao_A = new int[1];

	private int[] vbo_I_position	=	new int[1];	
	private int[] vbo_I_color		=	new int[1];	
	private int[] vbo_N_position	=	new int[1];	
	private int[] vbo_N_color		=	new int[1];	
	private int[] vbo_D_position	=	new int[1];	
	private int[] vbo_D_color		=	new int[1];	
	private int[] vbo_i_position	=	new int[1];	
	private int[] vbo_i_color		=	new int[1];	
	private int[] vbo_A_position	=	new int[1];	
	private int[] vbo_A_color		=	new int[1];	

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
				"in vec4 vColor;" +
				"uniform mat4 u_mvp_matirx;" +
				"out vec4 out_color;" +
				"void main(void)" +
				"{" +
				"gl_Position = u_mvp_matirx * vPosition;" +
				"out_color = vColor;" +
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
				"in vec4 out_color;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
				"FragColor = out_color;" +
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
										GLESMacros.AMC_ATTRIBUTE_COLOR,
										"vColor");

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
		final float[] I_vertices = new float[]
		{
			-1.15f, 0.7f, 0.0f,
			-1.25f, 0.7f, 0.0f,
			-1.25f, -0.7f, 0.0f,
			-1.15f, -0.7f, 0.0f
		};

		final float[] N_vertices = new float[]
		{
			-0.95f, 0.7f, 0.0f,
			-1.05f, 0.7f, 0.0f,
			-1.05f, -0.7f, 0.0f,
			-0.95f, -0.7f, 0.0f,

			-0.55f, 0.7f, 0.0f,
			-0.65f, 0.7f, 0.0f,
			-0.65f, -0.7f, 0.0f,
			-0.55f, -0.7f, 0.0f,

			-0.95f, 0.7f, 0.0f,
			-0.95f, 0.5f, 0.0f,
			-0.65f, -0.7f, 0.0f,
			-0.65f, -0.5f, 0.0f
		};

		final float[] D_vertices = new float[]
		{
			//top
			0.15f, 0.7f, 0.0f,
			-0.45f, 0.7f, 0.0f,
			-0.45f, 0.6f, 0.0f,
			0.15f, 0.6f, 0.0f,

			//bottom
			0.15f, -0.7f, 0.0f,
			-0.45f, -0.7f, 0.0f,
			-0.45f, -0.6f, 0.0f,
			0.15f, -0.6f, 0.0f,

			//left
			0.15f, 0.7f, 0.0f,
			0.05f, 0.7f, 0.0f,
			0.05f, -0.7f, 0.0f,
			0.15f, -0.7f, 0.0f,

			//right
			-0.25f, 0.6f, 0.0f,
			-0.35f, 0.6f, 0.0f,
			-0.35f, -0.6f, 0.0f,
			-0.25f, -0.6f, 0.0f
		};

		final float[] i_vertices = new float[]
		{
			0.35f, 0.7f, 0.0f,
			0.25f, 0.7f, 0.0f,
			0.25f, -0.7f, 0.0f,
			0.35f, -0.7f, 0.0f
		};

		final float[] A_vertices = new float[]
		{
			//left
			0.75f, 0.7f, 0.0f,
			0.75f, 0.5f, 0.0f,
			0.55f, -0.7f, 0.0f,
			0.45f, -0.7f, 0.0f,
			//right
			0.75f, 0.7f, 0.0f,
			0.75f, 0.5f, 0.0f,
			0.95f, -0.7f, 0.0f,
			1.05f, -0.7f, 0.0f,

			//middle strips
			0.66f, -0.05f, 0.0f,
			0.84f, -0.05f, 0.0f,

			0.65f, -0.1f, 0.0f,
			0.85f, -0.1f, 0.0f,

			0.64f, -0.15f, 0.0f,
			0.86f, -0.15f, 0.0f,
		};

		//color declaration
		final float[] I_color = new float[]
		{
			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f
		};

		final float[] N_color = new float[]
		{
			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,

			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,

			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
		};

		final float[] D_color = new float[]
		{
			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,

			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,

			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,

			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
		};

		final float[] i_color = new float[]
		{
			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
		};

		final float[] A_color = new float[]
		{
			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,

			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,

			1.0f, 0.5f, 0.0f,
			1.0f, 0.5f, 0.0f,

			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,

			0.0f, 0.5f, 0.0f,
			0.0f, 0.5f, 0.0f
		};

		//I

		//position
	
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_I, 0);
		GLES32.glBindVertexArray(vao_I[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_I_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_I_position[0]);

		//now from here below we are doing 5 steps to convert the I_vertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_I_position = ByteBuffer.allocateDirect(I_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_I_position.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_I = byteBuffer_I_position.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_I.put(I_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_I.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								I_vertices.length * 4,
								positionBuffer_I,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		//color
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_I_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_I_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_I_color = ByteBuffer.allocateDirect(I_color.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_I_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_I = byteBuffer_I_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_I.put(I_color);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_I.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								I_color.length * 4,
								colorBuffer_I,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_COLOR,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		//N

		//position
	
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_N, 0);
		GLES32.glBindVertexArray(vao_N[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_N_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_N_position[0]);

		//now from here below we are doing 5 steps to convert the I_vertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_N_position = ByteBuffer.allocateDirect(N_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_N_position.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_N = byteBuffer_N_position.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_N.put(N_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_N.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								N_vertices.length * 4,
								positionBuffer_N,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		//color
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_N_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_N_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_N_color = ByteBuffer.allocateDirect(N_color.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_N_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_N = byteBuffer_N_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_N.put(N_color);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_N.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								N_color.length * 4,
								colorBuffer_N,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_COLOR,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		//D

		//position
	
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_D, 0);
		GLES32.glBindVertexArray(vao_D[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_D_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_D_position[0]);

		//now from here below we are doing 5 steps to convert the I_vertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_D_position = ByteBuffer.allocateDirect(D_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_D_position.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_D = byteBuffer_D_position.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_D.put(D_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_D.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								D_vertices.length * 4,
								positionBuffer_D,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		//color
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_D_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_D_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_D_color = ByteBuffer.allocateDirect(D_color.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_D_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_D = byteBuffer_D_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_D.put(D_color);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_D.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								D_color.length * 4,
								colorBuffer_D,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_COLOR,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		//i

		//position
	
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_i, 0);
		GLES32.glBindVertexArray(vao_i[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_i_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_i_position[0]);

		//now from here below we are doing 5 steps to convert the I_vertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_i_position = ByteBuffer.allocateDirect(i_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_i_position.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_i = byteBuffer_i_position.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_i.put(i_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_i.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								i_vertices.length * 4,
								positionBuffer_i,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		//color
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_i_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_i_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_i_color = ByteBuffer.allocateDirect(i_color.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_i_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_i = byteBuffer_i_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_i.put(i_color);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_i.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								i_color.length * 4,
								colorBuffer_i,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_COLOR,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		//A

		//position
	
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_A, 0);
		GLES32.glBindVertexArray(vao_A[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_A_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_A_position[0]);

		//now from here below we are doing 5 steps to convert the I_vertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_A_position = ByteBuffer.allocateDirect(A_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_A_position.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_A = byteBuffer_A_position.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_A.put(A_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_A.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								A_vertices.length * 4,
								positionBuffer_A,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		//color
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_A_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_A_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_A_color = ByteBuffer.allocateDirect(A_color.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_A_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_A = byteBuffer_A_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_A.put(A_color);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_A.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								A_color.length * 4,
								colorBuffer_A,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_COLOR,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		//clear
		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		//depth
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
		float[] translationMatrix = new float[16];

		//init above metrices to identity
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);
		
		//do necessary matrix multiplication
		Matrix.translateM(	translationMatrix, 0, 
							0.0f,
							0.0f,
							-3.0f);
		
		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							translationMatrix, 0);

		Matrix.multiplyMM(	modelViewProjectionMatrix, 0,
							perspectiveProjectionMatrix, 0,
							modelViewMatrix, 0);
		//send necessary data to shader
		GLES32.glUniformMatrix4fv(	mvpUniform,
									1,
									false,
									modelViewProjectionMatrix, 0);

		oglDraw_I();
		oglDraw_N();
		oglDraw_D();
		oglDraw_i();
		oglDraw_A();

		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private void oglUpdate()
	{
		//code
	}

	private void oglDraw_I()
	{
		//code
		GLES32.glBindVertexArray(vao_I[0]);

		//draw scene
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);

		//unbind vao
		GLES32.glBindVertexArray(0);
	}

	private void oglDraw_N()
	{
		//code
		GLES32.glBindVertexArray(vao_N[0]);
		
		//draw scene
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 4);

		GLES32.glLineWidth(20.0f);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 8, 4);

		//unbind vao
		GLES32.glBindVertexArray(0);
	}

	private void oglDraw_D()
	{
		//code
		GLES32.glBindVertexArray(vao_D[0]);

		//draw scene
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 8, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 12, 4);

		//unbind vao
		GLES32.glBindVertexArray(0);
	}

	private void oglDraw_i()
	{
		//code
		GLES32.glBindVertexArray(vao_i[0]);

		//draw scene
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);

		//unbind vao
		GLES32.glBindVertexArray(0);
	}

	private void oglDraw_A()
	{
		//code
		GLES32.glBindVertexArray(vao_A[0]);

		//draw scene
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 4);

		GLES32.glLineWidth(3.0f);
		GLES32.glDrawArrays(GLES32.GL_LINES, 8, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 10, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 12, 2);

		//unbind vao
		GLES32.glBindVertexArray(0);
	}

	private void oglUninitialise()
	{
		//code
		if(vao_I[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_I, 0);
			vao_I[0] = 0;
		}
		if(vao_N[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_N, 0);
			vao_N[0] = 0;
		}
		if(vao_D[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_D, 0);
			vao_D[0] = 0;
		}
		if(vao_i[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_i, 0);
			vao_i[0] = 0;
		}
		if(vao_A[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_A, 0);
			vao_A[0] = 0;
		}

		if(vbo_I_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_I_position, 0);
			vbo_I_position[0] = 0;
		}
		if(vbo_I_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_I_color, 0);
			vbo_I_color[0] = 0;
		}

		if(vbo_N_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_N_position, 0);
			vbo_N_position[0] = 0;
		}
		if(vbo_N_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_N_color, 0);
			vbo_N_color[0] = 0;
		}

		if(vbo_D_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_D_position, 0);
			vbo_D_position[0] = 0;
		}
		if(vbo_D_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_D_color, 0);
			vbo_D_color[0] = 0;
		}

		if(vbo_i_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_i_position, 0);
			vbo_i_position[0] = 0;
		}
		if(vbo_i_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_i_color, 0);
			vbo_i_color[0] = 0;
		}

		if(vbo_A_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_A_position, 0);
			vbo_A_position[0] = 0;
		}
		if(vbo_A_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_A_color, 0);
			vbo_A_color[0] = 0;
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
