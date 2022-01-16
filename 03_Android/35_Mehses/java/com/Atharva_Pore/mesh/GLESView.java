package com.Atharva_Pore.mesh;

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
	
	public class buffers
	{
		private int[] vao = new int[1];
		private int[] vbo_position = new int[1];
		private int[] vbo_color = new int[1];
	}

	private int mvpUniform;
	private float[] perspectiveProjectionMatrix = new float[16];		//16 because it is 4 x 4 matrix.

	buffers one = new buffers();
	buffers two = new buffers();
	buffers three = new buffers();
	buffers four = new buffers();
	buffers five = new buffers();
	buffers six = new buffers();

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
				"gl_PointSize = 2.0;" +
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
				"out vec4 FragColor;" +
				"in vec4 out_color;" +
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

		//vertices declaration
		final float[] firstDesign_vertices = new float[]
		{
			//First Row
			-1.7f, 0.9f, 0.0f, 
			-1.5f, 0.9f, 0.0f, 
			-1.3f, 0.9f, 0.0f, 
			-1.1f, 0.9f, 0.0f, 

			//Second Row
			-1.7f, 0.7f, 0.0f, 
			-1.5f, 0.7f, 0.0f, 
			-1.3f, 0.7f, 0.0f, 
			-1.1f, 0.7f, 0.0f, 

			//Third Row
			-1.7f, 0.5f, 0.0f, 
			-1.5f, 0.5f, 0.0f, 
			-1.3f, 0.5f, 0.0f, 
			-1.1f, 0.5f, 0.0f, 

			//Fourth Row
			-1.7f, 0.3f, 0.0f, 
			-1.5f, 0.3f, 0.0f, 
			-1.3f, 0.3f, 0.0f, 
			-1.1f, 0.3f, 0.0f
		};

		final float[] secondDesign_vertice = new float[]
		{
			//1st Vertical Line
			-0.6f, 0.9f, 0.0f, 
			-0.6f, 0.3f, 0.0f, 
			//2nd Vertical Line
			-0.4f, 0.9f, 0.0f, 
			-0.4f, 0.3f, 0.0f, 
			//3rd Vertical Line
			-0.2f, 0.9f, 0.0f, 
			-0.2f, 0.3f, 0.0f, 
			
			//1st Horizontal Line
			-0.6f, 0.9f, 0.0f, 
			-0.0f, 0.9f, 0.0f, 

			//2nd Horizontal Line
			-0.6f, 0.7f, 0.0f, 
			-0.0f, 0.7f, 0.0f, 
			
			//3rd Horizontal Line
			-0.6f, 0.5f, 0.0f, 
			-0.0f, 0.5f, 0.0f, 

			//1st Olique Line
			-0.6f, 0.7f, 0.0f, 
			-0.4f, 0.9f, 0.0f, 

			//2nd Olique Line
			-0.6f, 0.5f, 0.0f, 
			-0.2f, 0.9f, 0.0f, 

			//3rd Olique Line
			-0.6f, 0.3f, 0.0f, 
			-0.0f, 0.9f, 0.0f, 

			//4th Olique Line
			-0.4f, 0.3f, 0.0f, 
			-0.0f, 0.7f, 0.0f, 

			-0.2f, 0.3f, 0.0f, 
			-0.0f, 0.5f, 0.0f
		};

		final float[] thirdDesign_vertices = new float[]
		{
			//1st Vertical Line
			0.3f, 0.9f, 0.0f,
			0.3f, 0.3f, 0.0f,
			//2nd Vertical Line
			0.5f, 0.9f, 0.0f,
			0.5f, 0.3f, 0.0f,
			//3rd Vertical Line
			0.7f, 0.9f, 0.0f,
			0.7f, 0.3f, 0.0f,
			//4th Vertical Line
			0.9f, 0.9f, 0.0f,
			0.9f, 0.3f, 0.0f,
			//1st Horizontal Line
			0.3f, 0.9f, 0.0f,
			0.9f, 0.9f, 0.0f,
			//2nd Horizontal Line
			0.3f, 0.7f, 0.0f,
			0.9f, 0.7f, 0.0f,
			//3rd Horizontal Line
			0.3f, 0.5f, 0.0f,
			0.9f, 0.5f, 0.0f,
			//4th Horizontal Line
			0.3f, 0.3f, 0.0f,
			0.9f, 0.3f, 0.0f
		};

		final float[] fourthDesign_vertices = new float[]
		{
			//4th Row
			-1.7f, -0.9f, 0.0f, 
			-1.1f, -0.9f, 0.0f, 
			//3rd Row
			-1.7f, -0.7f, 0.0f, 
			-1.1f, -0.7f, 0.0f, 
			//2nd Row
			-1.7f, -0.5f, 0.0f, 
			-1.1f, -0.5f, 0.0f, 
			//1st Row
			-1.7f, -0.3f, 0.0f, 
			-1.1f, -0.3f, 0.0f, 

			//4th column
			-1.7f, -0.9f, 0.0f, 
			-1.7f, -0.3f, 0.0f, 
			//3rd Column
			-1.5f, -0.9f, 0.0f, 
			-1.5f, -0.3f, 0.0f, 
			//2nd Column
			-1.3f, -0.9f, 0.0f, 
			-1.3f, -0.3f, 0.0f, 
			//1st Column
			-1.1f, -0.9f, 0.0f, 
			-1.1f, -0.3f, 0.0f, 

			//1st Olique Line
			-1.7f, -0.5f, 0.0f, 
			-1.5f, -0.3f, 0.0f, 
			//2nd Olique Line
			-1.7f, -0.7f, 0.0f, 
			-1.3f, -0.3f, 0.0f, 
			//3rd Olique Line
			-1.7f, -0.9f, 0.0f, 
			-1.1f, -0.3f, 0.0f, 
			//4th Olique Line
			-1.5f, -0.9f, 0.0f, 
			-1.1f, -0.5f, 0.0f, 
			//5th Olique Line
			-1.3f, -0.9f, 0.0f, 
			-1.1f, -0.7f, 0.0f
		};

		final float[] fifthDesign_vertices = new float[]
		{
			//4th Row
			-0.6f, -0.9f, 0.0f,
			-0.0f, -0.9f, 0.0f,
			//1st Row
			-0.6f, -0.3f, 0.0f,
			-0.0f, -0.3f, 0.0f,

			//4th column
			-0.6f, -0.9f, 0.0f,
			-0.6f, -0.3f, 0.0f,
			//1st Column
			0.0f, -0.9f, 0.0f,
			0.0f, -0.3f, 0.0f,

			//Ray
			-0.6f, -0.3f, 0.0f,
			0.0f, -0.5f, 0.0f,

			-0.6f, -0.3f, 0.0f,
			0.0f, -0.7f, 0.0f,

			-0.6f, -0.3f, 0.0f,
			0.0f, -0.9f, 0.0f,

			-0.6f, -0.3f, 0.0f,
			-0.4f, -0.9f, 0.0f,

			-0.6f, -0.3f, 0.0f,
			-0.2f, -0.9f, 0.0f
		};

		final float[] sixthDesign_vertices = new float[]
		{
			//first quad
			0.5f, -0.3f, 0.0f, 
			0.3f, -0.3f, 0.0f, 
			0.3f, -0.9f, 0.0f, 
			0.5f, -0.9f, 0.0f, 

			//second quad
			0.7f, -0.3f, 0.0f, 
			0.5f, -0.3f, 0.0f, 
			0.5f, -0.9f, 0.0f, 
			0.7f, -0.9f, 0.0f, 

			//third quad
			0.9f, -0.3f, 0.0f,
			0.7f, -0.3f, 0.0f,
			0.7f, -0.9f, 0.0f,
			0.9f, -0.9f, 0.0f,

			//vertical line 1
			0.5f, -0.3f, 0.0f,
			0.5f, -0.9f, 0.0f,

			//vertical line 2
			0.7f, -0.3f, 0.0f,
			0.7f, -0.9f, 0.0f,

			//Horizontal Line 1
			0.3f, -0.5f, 0.0f,
			0.9f, -0.5f, 0.0f,

			//Horizontal Line 1
			0.3f, -0.7f, 0.0f,
			0.9f, -0.7f, 0.0f
		};

		final float[] sixthDesign_color = new float[]
		{
			//first quad
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,

			//second quad
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,

			//third quad
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,

			//vertical line 1
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,

			//vertical line 2
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,

			//Horizontal Line 1
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,

			//Horizontal Line 1
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f
		};


		/* First Design */
		//create and bind vao
		GLES32.glGenVertexArrays(1, one.vao, 0);
		GLES32.glBindVertexArray(one.vao[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, one.vbo_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, one.vbo_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_one = ByteBuffer.allocateDirect(firstDesign_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_one.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_one = byteBuffer_one.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_one.put(firstDesign_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_one.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								firstDesign_vertices.length * 4,
								positionBuffer_one,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

		GLES32.glBindVertexArray(0);

		/* Second Design */
		//create and bind vao
		GLES32.glGenVertexArrays(1, two.vao, 0);
		GLES32.glBindVertexArray(two.vao[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, two.vbo_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, two.vbo_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_two = ByteBuffer.allocateDirect(secondDesign_vertice.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_two.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_two = byteBuffer_two.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_two.put(secondDesign_vertice);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_two.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								secondDesign_vertice.length * 4,
								positionBuffer_two,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

		GLES32.glBindVertexArray(0);

		/* Third Design */
		//create and bind vao
		GLES32.glGenVertexArrays(1, three.vao, 0);
		GLES32.glBindVertexArray(three.vao[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, three.vbo_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, three.vbo_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_three = ByteBuffer.allocateDirect(thirdDesign_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_three.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_three = byteBuffer_three.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_three.put(thirdDesign_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_three.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								thirdDesign_vertices.length * 4,
								positionBuffer_three,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

		GLES32.glBindVertexArray(0);

		/* Fourth Design */
		//create and bind vao
		GLES32.glGenVertexArrays(1, four.vao, 0);
		GLES32.glBindVertexArray(four.vao[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, four.vbo_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, four.vbo_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_four = ByteBuffer.allocateDirect(fourthDesign_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_four.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_four = byteBuffer_four.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_four.put(fourthDesign_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_four.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								fourthDesign_vertices.length * 4,
								positionBuffer_four,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

		GLES32.glBindVertexArray(0);

		/* Fifth Design */
		//create and bind vao
		GLES32.glGenVertexArrays(1, five.vao, 0);
		GLES32.glBindVertexArray(five.vao[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, five.vbo_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, five.vbo_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_five = ByteBuffer.allocateDirect(fifthDesign_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_five.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_five = byteBuffer_two.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_five.put(fifthDesign_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_five.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								fifthDesign_vertices.length * 4,
								positionBuffer_five,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

		GLES32.glBindVertexArray(0);

		/* Sixth Design */
		//create and bind vao
		GLES32.glGenVertexArrays(1, six.vao, 0);
		GLES32.glBindVertexArray(six.vao[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, six.vbo_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, six.vbo_position[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_six = ByteBuffer.allocateDirect(sixthDesign_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_six.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_six = byteBuffer_six.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_six.put(sixthDesign_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_six.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								sixthDesign_vertices.length * 4,
								positionBuffer_six,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		//create and bind vbo
		GLES32.glGenBuffers(1, six.vbo_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, six.vbo_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_six_color = ByteBuffer.allocateDirect(sixthDesign_color.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_six_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_six_color = byteBuffer_six_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_six_color.put(sixthDesign_color);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_six_color.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								sixthDesign_color.length * 4,
								positionBuffer_six_color,
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
							-3.0f);
		
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
		DottedSquare();
		SquareAndObliqueLine();

		//init above metrices to identity
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);

		//do necessary matrix multiplication
		Matrix.translateM(	translationMatrix, 0, 
							0.2f,
							0.0f,
							-3.0f);
		
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

		

		Design_two();
		SquareAndRay();

		//init above metrices to identity
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);

		//do necessary matrix multiplication
		Matrix.translateM(	translationMatrix, 0, 
							0.6f,
							0.0f,
							-3.0f);
		
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

		Square();
		RGB_Quads();

		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private void oglUninitialise()
	{
		//code
		if (one.vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, one.vbo_position, 0);
			one.vbo_position[0] = 0;
		}
		if (one.vao[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, one.vao, 0);
			one.vao[0] = 0;
		}

		if (two.vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, two.vbo_position, 0);
			two.vbo_position[0] = 0;
		}
		if (two.vao[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, two.vao, 0);
			two.vao[0] = 0;
		}

		if (three.vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, three.vbo_position, 0);
			three.vbo_position[0] = 0;
		}
		if (three.vao[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, three.vao, 0);
			three.vao[0] = 0;
		}

		if (four.vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, four.vbo_position, 0);
			four.vbo_position[0] = 0;
		}
		if (four.vao[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, four.vao, 0);
			four.vao[0] = 0;
		}

		if (five.vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, five.vbo_position, 0);
			five.vbo_position[0] = 0;
		}
		if (five.vao[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, five.vao, 0);
			five.vao[0] = 0;
		}

		if (six.vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, six.vbo_position, 0);
			six.vbo_position[0] = 0;
		}
		if (six.vbo_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, six.vbo_color, 0);
			six.vbo_color[0] = 0;
		}
		if (six.vao[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1, six.vao, 0);
			six.vao[0] = 0;
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

	private void DottedSquare()
	{
		//glPointSize(2.0f);
		GLES32.glBindVertexArray(one.vao[0]);

		//First Row
		GLES32.glDrawArrays(GLES32.GL_POINTS, 0, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 1, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 2, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 3, 1);

		//Second Row
		GLES32.glDrawArrays(GLES32.GL_POINTS, 4, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 5, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 6, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 7, 1);

		//Third Row
		GLES32.glDrawArrays(GLES32.GL_POINTS, 8, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 9, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 10, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 11, 1);

		//Fourth Row
		GLES32.glDrawArrays(GLES32.GL_POINTS, 12, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 13, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 14, 1);
		GLES32.glDrawArrays(GLES32.GL_POINTS, 15, 1);

		GLES32.glBindVertexArray(0);
	}

	private void Design_two()
	{

		GLES32.glBindVertexArray(two.vao[0]);
		
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

		GLES32.glDrawArrays(GLES32.GL_LINES, 20, 2);
		
		GLES32.glDrawArrays(GLES32.GL_LINES, 22, 2);
		
		

		GLES32.glBindVertexArray(0);

	}

	private void Square()
	{


		GLES32.glBindVertexArray(three.vao[0]);

		//1st Vertical Line
		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 2);
		//2nd Vertical Line
		GLES32.glDrawArrays(GLES32.GL_LINES, 2, 2);
		//3rd Vertical Line
		GLES32.glDrawArrays(GLES32.GL_LINES, 4, 2);
		//4th Vertical Line
		GLES32.glDrawArrays(GLES32.GL_LINES, 6, 2);

		//1st Horizontal Line
		GLES32.glDrawArrays(GLES32.GL_LINES, 8, 2);
		//2nd Horizontal Line
		GLES32.glDrawArrays(GLES32.GL_LINES, 10, 2);
		//3rd Horizontal Line
		GLES32.glDrawArrays(GLES32.GL_LINES, 12, 2);
		//4th Horizontal Line
		GLES32.glDrawArrays(GLES32.GL_LINES, 14, 2);

		GLES32.glBindVertexArray(0);
	}

	private void SquareAndObliqueLine()
	{

		GLES32.glBindVertexArray(four.vao[0]);

		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 2);//4th Row
		GLES32.glDrawArrays(GLES32.GL_LINES, 2, 2);//3rd Row
		GLES32.glDrawArrays(GLES32.GL_LINES, 4, 2);//2nd Row
		GLES32.glDrawArrays(GLES32.GL_LINES, 6, 2);//1st Row
		GLES32.glDrawArrays(GLES32.GL_LINES, 8, 2);//4th column
		GLES32.glDrawArrays(GLES32.GL_LINES, 10, 2);//3rd column
		GLES32.glDrawArrays(GLES32.GL_LINES, 12, 2);//2nd column
		GLES32.glDrawArrays(GLES32.GL_LINES, 14, 2);//1st column
		GLES32.glDrawArrays(GLES32.GL_LINES, 16, 2);//1st OliqueLine
		GLES32.glDrawArrays(GLES32.GL_LINES, 18, 2);//2nd OliqueLine
		GLES32.glDrawArrays(GLES32.GL_LINES, 20, 2);//3rd OliqueLine
		GLES32.glDrawArrays(GLES32.GL_LINES, 22, 2);//4th OliqueLine
		GLES32.glDrawArrays(GLES32.GL_LINES, 24, 2);//5th OliqueLine

		GLES32.glBindVertexArray(0);
	}

	private void SquareAndRay()
	{
		GLES32.glBindVertexArray(five.vao[0]);

		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 2);//4th Row
		GLES32.glDrawArrays(GLES32.GL_LINES, 2, 2);//1st Row
		GLES32.glDrawArrays(GLES32.GL_LINES, 4, 2);//4th column
		GLES32.glDrawArrays(GLES32.GL_LINES, 6, 2);//1st Column
		
		//ray
		GLES32.glDrawArrays(GLES32.GL_LINES, 8, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 10, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 12, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 14, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 16, 2);

		GLES32.glBindVertexArray(0);
	}

	private void RGB_Quads()
	{
		GLES32.glLineWidth(3.0f);
		GLES32.glBindVertexArray(six.vao[0]);

		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 8, 4);

		GLES32.glDrawArrays(GLES32.GL_LINES, 12, 2);//vertical line 1
		GLES32.glDrawArrays(GLES32.GL_LINES, 14, 2);//vertical line 2
		GLES32.glDrawArrays(GLES32.GL_LINES, 16, 2);//Horizontal Line 1
		GLES32.glDrawArrays(GLES32.GL_LINES, 18, 2);//Horizontal Line 2

		GLES32.glBindVertexArray(0);
	}
}
