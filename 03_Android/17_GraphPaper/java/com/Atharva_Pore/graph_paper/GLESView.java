package com.Atharva_Pore.graph_paper;

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
	private int[] vao_red = new int[1];
	private int[] vao_green = new int[1];
	private int[] vao_blue = new int[1];
	private int[] vbo_red_line_position = new int[1];
	private int[] vbo_red_line_color = new int[1];
	private int[] vbo_green_line_position = new int[1];
	private int[] vbo_green_line_color = new int[1];
	private int[] vbo_blue_line_position = new int[1];
	private int[] vbo_blue_line_color = new int[1];

	private int mvpUniform;

	private float[] perspectiveProjectionMatrix = new float[16];		//16 because it is 4 x 4 matrix.

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
				"out vec4 out_color;" +
				"uniform mat4 u_mvp_matirx;" +
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
		final float[] blueLines = new float[]
		{
			-0.95f, 1.0f, 0.0f,
			-0.95f, -1.0f, 0.0f,

			-0.90f, 1.0f, 0.0f,
			-0.90f, -1.0f, 0.0f,

			-0.85f, 1.0f, 0.0f,
			-0.85f, -1.0f, 0.0f,

			-0.80f, 1.0f, 0.0f,
			-0.80f, -1.0f, 0.0f,

			-0.75f, 1.0f, 0.0f,
			-0.75f, -1.0f, 0.0f,

			-0.70f, 1.0f, 0.0f,
			-0.70f, -1.0f, 0.0f,

			-0.65f, 1.0f, 0.0f,
			-0.65f, -1.0f, 0.0f,

			-0.60f, 1.0f, 0.0f,
			-0.60f, -1.0f, 0.0f,

			-0.55f, 1.0f, 0.0f,
			-0.55f, -1.0f, 0.0f,

			-0.50f, 1.0f, 0.0f,
			-0.50f, -1.0f, 0.0f,

			-0.45f, 1.0f, 0.0f,
			-0.45f, -1.0f, 0.0f,

			-0.40f, 1.0f, 0.0f,
			-0.40f, -1.0f, 0.0f,

			-0.35f, 1.0f, 0.0f,
			-0.35f, -1.0f, 0.0f,

			-0.30f, 1.0f, 0.0f,
			-0.30f, -1.0f, 0.0f,

			-0.25f, 1.0f, 0.0f,
			-0.25f, -1.0f, 0.0f,

			-0.20f, 1.0f, 0.0f,
			-0.20f, -1.0f, 0.0f,

			-0.15f, 1.0f, 0.0f,
			-0.15f, -1.0f, 0.0f,

			-0.10f, 1.0f, 0.0f,
			-0.10f, -1.0f, 0.0f,

			-0.05f, 1.0f, 0.0f,
			-0.05f, -1.0f, 0.0f,

			0.95f, 1.0f, 0.0f,
			0.95f, -1.0f, 0.0f,

			0.90f, 1.0f, 0.0f,
			0.90f, -1.0f, 0.0f,

			0.85f, 1.0f, 0.0f,
			0.85f, -1.0f, 0.0f,

			0.80f, 1.0f, 0.0f,
			0.80f, -1.0f, 0.0f,

			0.75f, 1.0f, 0.0f,
			0.75f, -1.0f, 0.0f,

			0.70f, 1.0f, 0.0f,
			0.70f, -1.0f, 0.0f,

			0.65f, 1.0f, 0.0f,
			0.65f, -1.0f, 0.0f,

			0.60f, 1.0f, 0.0f,
			0.60f, -1.0f, 0.0f,

			0.55f, 1.0f, 0.0f,
			0.55f, -1.0f, 0.0f,

			0.50f, 1.0f, 0.0f,
			0.50f, -1.0f, 0.0f,

			0.45f, 1.0f, 0.0f,
			0.45f, -1.0f, 0.0f,

			0.40f, 1.0f, 0.0f,
			0.40f, -1.0f, 0.0f,

			0.35f, 1.0f, 0.0f,
			0.35f, -1.0f, 0.0f,

			0.30f, 1.0f, 0.0f,
			0.30f, -1.0f, 0.0f,

			0.25f, 1.0f, 0.0f,
			0.25f, -1.0f, 0.0f,

			0.20f, 1.0f, 0.0f,
			0.20f, -1.0f, 0.0f,

			0.15f, 1.0f, 0.0f,
			0.15f, -1.0f, 0.0f,

			0.10f, 1.0f, 0.0f,
			0.10f, -1.0f, 0.0f,

			0.05f, 1.0f, 0.0f,
			0.05f, -1.0f, 0.0f,

			1.0f, -0.95f, 0.0f,
			-1.0f, -0.95f, 0.0f,

			1.0f, -0.90f, 0.0f,
			-1.0f, -0.90f, 0.0f,

			1.0f, -0.85f, 0.0f,
			-1.0f, -0.85f, 0.0f,

			1.0f, -0.80f, 0.0f,
			-1.0f, -0.80f, 0.0f,

			1.0f, -0.75f, 0.0f,
			-1.0f, -0.75f, 0.0f,

			1.0f, -0.70f, 0.0f,
			-1.0f, -0.70f, 0.0f,

			1.0f, -0.65f, 0.0f,
			-1.0f, -0.65f, 0.0f,

			1.0f, -0.60f, 0.0f,
			-1.0f, -0.60f, 0.0f,

			1.0f, -0.55f, 0.0f,
			-1.0f, -0.55f, 0.0f,

			1.0f, -0.50f, 0.0f,
			-1.0f, -0.50f, 0.0f,

			1.0f, -0.45f, 0.0f,
			-1.0f, -0.45f, 0.0f,

			1.0f, -0.40f, 0.0f,
			-1.0f, -0.40f, 0.0f,

			1.0f, -0.35f, 0.0f,
			-1.0f, -0.35f, 0.0f,

			1.0f, -0.30f, 0.0f,
			-1.0f, -0.30f, 0.0f,

			1.0f, -0.25f, 0.0f,
			-1.0f, -0.25f, 0.0f,

			1.0f, -0.20f, 0.0f,
			-1.0f, -0.20f, 0.0f,

			1.0f, -0.15f, 0.0f,
			-1.0f, -0.15f, 0.0f,

			1.0f, -0.10f, 0.0f,
			-1.0f, -0.10f, 0.0f,

			1.0f, -0.05f, 0.0f,
			-1.0f, -0.05f, 0.0f,

			1.0f, 0.95f, 0.0f,
			-1.0f, 0.95f, 0.0f,

			1.0f, 0.90f, 0.0f,
			-1.0f, 0.90f, 0.0f,

			1.0f, 0.85f, 0.0f,
			-1.0f, 0.85f, 0.0f,

			1.0f, 0.80f, 0.0f,
			-1.0f, 0.80f, 0.0f,

			1.0f, 0.75f, 0.0f,
			-1.0f, 0.75f, 0.0f,

			1.0f, 0.70f, 0.0f,
			-1.0f, 0.70f, 0.0f,

			1.0f, 0.65f, 0.0f,
			-1.0f, 0.65f, 0.0f,

			1.0f, 0.60f, 0.0f,
			-1.0f, 0.60f, 0.0f,

			1.0f, 0.55f, 0.0f,
			-1.0f, 0.55f, 0.0f,

			1.0f, 0.50f, 0.0f,
			-1.0f, 0.50f, 0.0f,

			1.0f, 0.45f, 0.0f,
			-1.0f, 0.45f, 0.0f,

			1.0f, 0.40f, 0.0f,
			-1.0f, 0.40f, 0.0f,

			1.0f, 0.35f, 0.0f,
			-1.0f, 0.35f, 0.0f,

			1.0f, 0.30f, 0.0f,
			-1.0f, 0.30f, 0.0f,

			1.0f, 0.25f, 0.0f,
			-1.0f, 0.25f, 0.0f,

			1.0f, 0.20f, 0.0f,
			-1.0f, 0.20f, 0.0f,

			1.0f, 0.15f, 0.0f,
			-1.0f, 0.15f, 0.0f,

			1.0f, 0.10f, 0.0f,
			-1.0f, 0.10f, 0.0f,

			1.0f, 0.05f, 0.0f,
			-1.0f, 0.05f, 0.0f
		};

		final float[] redLines = new float[]
		{
			1.0f, 0.0f, 0.0f,
			-1.0f, 0.0f, 0.0f
		};

		final float[] greenLines = new float[]
		{
			0.0f, 1.0f, 0.0f,
			0.0f, -1.0f, 0.0f
		};

		//color
		final float[] redColor = new float[]
		{
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f
		};

		final float[] greenColor = new float[]
		{
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f
		};

		//GREEN LINES

		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_green, 0);
		GLES32.glBindVertexArray(vao_green[0]);

		//vertices

		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_green_line_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_green_line_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_green_vertex = ByteBuffer.allocateDirect(greenLines.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_green_vertex.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_green = byteBuffer_green_vertex.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_green.put(greenLines);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_green.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								greenLines.length * 4,
								positionBuffer_green,
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
		GLES32.glGenBuffers(1, vbo_green_line_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_green_line_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_green_color = ByteBuffer.allocateDirect(greenColor.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_green_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_green = byteBuffer_green_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_green.put(greenColor);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_green.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								greenColor.length * 4,
								colorBuffer_green,
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

		//RED LINE

		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_red, 0);
		GLES32.glBindVertexArray(vao_red[0]);

		//vertices

		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_red_line_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_red_line_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_red_vertex = ByteBuffer.allocateDirect(redLines.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_red_vertex.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_red = byteBuffer_red_vertex.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_red.put(redLines);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_red.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								redLines.length * 4,
								positionBuffer_red,
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
		GLES32.glGenBuffers(1, vbo_red_line_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_red_line_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_red_color = ByteBuffer.allocateDirect(redColor.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_red_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_red = byteBuffer_red_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_red.put(redColor);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_red.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								redColor.length * 4,
								colorBuffer_red,
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

		//BLUE LINES
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_blue_line_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_blue_line_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_blue_vertex = ByteBuffer.allocateDirect(blueLines.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_blue_vertex.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_blue = byteBuffer_blue_vertex.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_blue.put(blueLines);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_blue.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								blueLines.length * 4,
								positionBuffer_blue,
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
		GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 1.0f);

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
		float[] translationMatrix = new float[16];

		//init above metrices to identity
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);

		//do necessary matrix multiplication
		Matrix.translateM(	translationMatrix, 0, 
							0.0f,
							0.0f,
							-1.2f);
		
		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							translationMatrix, 0);

		Matrix.multiplyMM(	modelViewProjectionMatrix, 0,
							perspectiveProjectionMatrix, 0,
							modelViewMatrix, 0);

		//send necessary matrics to shaders in respective uniforms
		GLES32.glUniformMatrix4fv(	mvpUniform,
									1,
									false,
									modelViewProjectionMatrix, 0);

		//bind with vao
		GLES32.glBindVertexArray(vao_red[0]);

		//draw scene
		GLES32.glDrawArrays(	GLES32.GL_LINES, 
								0, 
								2);
		
		//unbind vao
		GLES32.glBindVertexArray(0);

		//bind with vao
		GLES32.glBindVertexArray(vao_green[0]);

		//draw scene
		GLES32.glDrawArrays(	GLES32.GL_LINES, 
								0, 
								2);
		
		//unbind vao
		GLES32.glBindVertexArray(0);

		//bind with vao
		GLES32.glBindVertexArray(vao_blue[0]);

		//draw scene
		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 2, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 4, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 6, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 8, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 10, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 12, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 14, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 16, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 18, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 20, 2);//gap

		GLES32.glDrawArrays(GLES32.GL_LINES, 22, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 24, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 26, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 28, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 30, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 32, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 34, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 36, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 38, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 40, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 42, 2);//gap

		GLES32.glDrawArrays(GLES32.GL_LINES, 44, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 46, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 48, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 50, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 52, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 54, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 56, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 58, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 60, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 62, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 64, 2);//gap

		GLES32.glDrawArrays(GLES32.GL_LINES, 66, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 68, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 70, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 72, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 74, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 76, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 78, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 80, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 82, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 84, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 86, 2);//gap

		GLES32.glDrawArrays(GLES32.GL_LINES, 88, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 90, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 92, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 94, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 96, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 98, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 100, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 102, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 104, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 106, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 108, 2);//gap

		GLES32.glDrawArrays(GLES32.GL_LINES, 110, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 112, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 114, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 116, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 118, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 120, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 122, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 124, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 126, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 128, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 130, 2);//gap

		GLES32.glDrawArrays(GLES32.GL_LINES, 132, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 134, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 136, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 138, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 140, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 142, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 144, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 146, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 148, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 150, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 152, 2);//gap

		GLES32.glDrawArrays(GLES32.GL_LINES, 154, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 156, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 158, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 160, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 162, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 164, 2);
		
		//unbind vao
		GLES32.glBindVertexArray(0);
		
		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private void oglUninitialise()
	{
		//code
		if (vbo_red_line_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_red_line_position, 0);
			vbo_red_line_position[0] = 0;
		}
		if (vbo_red_line_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_red_line_color, 0);
			vbo_red_line_color[0] = 0;
		}

		if (vbo_green_line_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_green_line_position, 0);
			vbo_green_line_position[0] = 0;
		}
		if (vbo_green_line_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_green_line_color, 0);
			vbo_green_line_color[0] = 0;
		}

		if (vbo_blue_line_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_blue_line_position, 0);
			vbo_blue_line_position[0] = 0;
		}
		if (vbo_blue_line_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_blue_line_color, 0);
			vbo_blue_line_color[0] = 0;
		}
		
		if (vao_red[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_red, 0);
			vao_red[0] = 0;
		}
		if (vao_green[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_green, 0);
			vao_green[0] = 0;
		}
		if (vao_blue[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, vao_blue, 0);
			vao_blue[0] = 0;
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
