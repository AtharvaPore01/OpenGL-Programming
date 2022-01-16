package com.Atharva_Pore.two_3d_textured_shapes_animated;

//programmable related (OpenGL Related) packages
import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import javax.microedition.khronos.opengles.GL10;			//for basic features of openGL-ES
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;									//nio = Non Blocking I/O OR Native I/O,	For Opengl Buffers.
import java.nio.ByteOrder;									//for arranginf byte order of the buffer in native byte order(Little Indian / Big Indian)
import java.nio.FloatBuffer;								//to create float type buffer.

import android.opengl.Matrix;								//for matrix mathematics.

//texture related packages
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.opengl.GLUtils;								//for texImage2D()

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
	private int[] vbo_pyramid_texture = new int[1];

	private int[] vao_cube = new int[1];
	private int[] vbo_cube_position = new int[1];
	private int[] vbo_cube_texture = new int[1];

	private int mvpUniform;
	private int samplerUniform;

	private float[] perspectiveProjectionMatrix = new float[16];		//16 because it is 4 x 4 matrix.

	//texture related variables
	private int[] texture_kudali = new int[1];
	private int[] texture_stone = new int[1];

	//Rotation variables
	float rotation_angle_pyramid = 0.0f;
	float rotation_angle_cube = 0.0f;

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
		oglupdate();
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
				"in vec2 vTexcoord;" +
				"out vec2 out_texcoord;" +
				"uniform mat4 u_mvp_matirx;" +
				"void main(void)" +
				"{" +
				"gl_Position = u_mvp_matirx * vPosition;" +
				"out_texcoord = vTexcoord;" +
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
				"in vec2 out_texcoord;" +
				"out vec4 FragColor;" +
				"uniform highp sampler2D u_sampler;" +
				"void main(void)" +
				"{" +
				"FragColor = texture(u_sampler, out_texcoord);" +
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
										GLESMacros.AMC_ATTRIBUTE_TEXCOORD_0,
										"vTexcoord");

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
		samplerUniform = GLES32.glGetUniformLocation(	shaderProgramObject,
														"u_sampler");

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

		final float[] pyramidTexcoord = new float[]
		{
			0.5f, 1.0f,
			0.0f, 0.0f, 
			1.0f, 0.0f,
	
			0.5f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,
	
			0.5f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,
	
			0.5f, 1.0f,
			0.0f, 0.0f,
			1.0f, 0.0f
		};

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
			-1.0f, -1.0f, -1.0f,
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

		final float[] cubeTexcoord = new float[]
		{
			
			0.0f, 1.0f,
			0.0f, 0.0f, 
			1.0f, 0.0f,
			1.0f, 1.0f,
	
			1.0f, 1.0f,
			0.0f, 1.0f,
			0.0f, 0.0f,
			1.0f, 0.0f,
	
			0.0f, 0.0f,
			1.0f, 0.0f,
			1.0f, 1.0f, 
			0.0f, 1.0f,
	
			1.0f, 0.0f,
			1.0f, 1.0f,
			0.0f, 1.0f,
			0.0f, 0.0f,
	
			1.0f, 0.0f,
			1.0f, 1.0f,
			0.0f, 1.0f, 
			0.0f, 0.0f,
	
			0.0f, 0.0f,
			1.0f, 0.0f,
			1.0f, 1.0f,
			0.0f, 1.0f
		
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

		/* texture */
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_pyramid_texture, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_pyramid_texture[0]);

		//steps to make triangleColours array as buffer compatible for glBufferData().

		//step 1:	allocate buffer from native memory directly
		ByteBuffer byteBuffer_pyramid_texture = ByteBuffer.allocateDirect(pyramidTexcoord.length * 4);

		//step 2:	set the byte order as native byte order
		byteBuffer_pyramid_texture.order(ByteOrder.nativeOrder());

		//step 3:	create a float type buffer of our byte type buffer
		FloatBuffer textureBuffer_pyramid = byteBuffer_pyramid_texture.asFloatBuffer();

		//step 4:	now put the array in COOKED buffer
		textureBuffer_pyramid.put(pyramidTexcoord);

		//step 5:	set the starting position as 0th position
		textureBuffer_pyramid.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								pyramidTexcoord.length * 4,
								textureBuffer_pyramid,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_TEXCOORD_0,
										2,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_TEXCOORD_0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		/* Rectangle */

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

		/* texture */
		//create and bind buffer
		GLES32.glGenBuffers(1, vbo_cube_texture, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_cube_texture[0]);

		//convert the array in glBufferData() compatibile buffer.

		//step 1: allocate the buffer directly from native memory
		ByteBuffer byteBuffer_cube_texture = ByteBuffer.allocateDirect(cubeTexcoord.length * 4);

		//step 2: change the byte order to the native byte order
		byteBuffer_cube_texture.order(ByteOrder.nativeOrder());

		//step 3: make the byte array as float array
		FloatBuffer textureBuffer_cube = byteBuffer_cube_texture.asFloatBuffer();

		//step 4: put our array in that COOKED buffer.
		textureBuffer_cube.put(cubeTexcoord);

		//step 5: set the array position as 0th position
		textureBuffer_cube.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								cubeTexcoord.length * 4,
								textureBuffer_cube,
								GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_TEXCOORD_0,
										2,
										GLES32.GL_FLOAT,
										false,
										0,
										0);

		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_TEXCOORD_0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		//clear
		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		//depth
		//GLES32.glClearDepth(1.0f);
		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		//texture
		GLES32.glEnable(GLES32.GL_ARRAY_BUFFER);
		texture_stone[0] = oglLoadTexture(R.raw.stone);		//R 		=> 	res directory is considered as R.
															//raw 		=>	folder in res directory =, it is traditional android folder to save textures and shaders,
															//				in 'ant' it comes by default but in gradle we have to create this folder.
															//stone 	=>	this is texture file name WITHOUT EXTENSION.

		texture_kudali[0] = oglLoadTexture(R.raw.kundali);	//R 		=> 	res directory is considered as R.
															//raw 		=>	folder in res directory =, it is traditional android folder to save textures and shaders,
															//				in 'ant' it comes by default but in gradle we have to create this folder.
															//kundali 	=>	this is texture file name WITHOUT EXTENSION.


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
		float[] modelViewProjectionMatrix = new float[16];
		float[] translationMatrix = new float[16];
		float[] rotationMatrix = new float[16];

		float[] rotationMatrix_x = new float[16];
		float[] rotationMatrix_y = new float[16];
		float[] rotationMatrix_z = new float[16];

		/* Triangle */

		//init above metrices to identity
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.setIdentityM(rotationMatrix, 0);

		//do necessary matrix multiplication
		Matrix.translateM(	translationMatrix, 0, 
							-2.0f,
							0.0f,
							-8.0f);
		//static void rotateM(float[] m, int mOffset, float a, float x, float y, float z)
		Matrix.setRotateM(	rotationMatrix, 0,
							rotation_angle_pyramid,
							0.0f,
							1.0f,
							0.0f);
		
		Matrix.multiplyMM(	modelViewMatrix, 0,
							modelViewMatrix, 0,
							translationMatrix, 0);

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

		//active texture
		GLES32.glActiveTexture(GLES32.GL_TEXTURE0);

		//bind texture
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_stone[0]);

		//push the data in fragment shader
		GLES32.glUniform1i(samplerUniform, 0);

		//bind with vao
		GLES32.glBindVertexArray(vao_pyramid[0]);

		//draw scene
		GLES32.glDrawArrays(	GLES32.GL_TRIANGLES,
								0,
								12);
		
		//unbind vao
		GLES32.glBindVertexArray(0);

		//unbind texture
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, 0);

		/* Rectangle */
		
		//make all matrices indentity.
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.setIdentityM(rotationMatrix_x, 0);
		Matrix.setIdentityM(rotationMatrix_y, 0);
		Matrix.setIdentityM(rotationMatrix_z, 0);

		//do matrix multiplication
		Matrix.translateM(	translationMatrix, 0,
							2.0f, 
							0.0f, 
							-8.0f);

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

		Matrix.multiplyMM(	modelViewProjectionMatrix, 0,
							perspectiveProjectionMatrix, 0,
							modelViewMatrix, 0);

		//send this data to shader
		GLES32.glUniformMatrix4fv(	mvpUniform,
									1,
									false,
									modelViewProjectionMatrix, 0);
		//active texture
		GLES32.glActiveTexture(GLES32.GL_TEXTURE0);

		//bind texture
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_kudali[0]);

		//push the data in fragment shader
		GLES32.glUniform1i(samplerUniform, 0);

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

		//unbind texture
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, 0);
		
		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private int oglLoadTexture(int imageResourceID)
	{
		//variable declaration
		int[] texture = new int[1];

		//code
		BitmapFactory.Options options = new BitmapFactory.Options();
		options.inScaled = false; //android asks whether do i scale your incoming image, we say false(no).
		
		//static Bitmap	decodeResource(Resources res, int id, BitmapFactory.Options opts)
		/*	He Active Class Give Me Your All Rsources And Find The Resource which is matched 
		*	to given imageResourceID, and attach all the options to that resource.
		*/
		Bitmap bitmap = BitmapFactory.decodeResource(	context.getResources(),
														imageResourceID,
														options);

		GLES32.glPixelStorei(GLES32.GL_UNPACK_ALIGNMENT, 4);

		//generate texture
		GLES32.glGenTextures(1, texture, 0);

		//bind texture
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture[0]);

		//set parameters
		GLES32.glTexParameteri(	GLES32.GL_TEXTURE_2D,
								GLES32.GL_TEXTURE_MAG_FILTER,
								GLES32.GL_LINEAR);
		GLES32.glTexParameteri(	GLES32.GL_TEXTURE_2D,
								GLES32.GL_TEXTURE_MIN_FILTER,
								GLES32.GL_LINEAR_MIPMAP_LINEAR);
		
		//static void texImage2D(int target, int level, Bitmap bitmap, int border)
		GLUtils.texImage2D(	GLES32.GL_TEXTURE_2D,
							0,
							bitmap, //contains glTexImage2D(internalFormat(3rd), width(4th), height(5th), externalFormat(7th), GL_UNSIGNED_BYTE(8th), bm.bmBits(9th))
							0);

		GLES32.glGenerateMipmap(GLES32.GL_TEXTURE_2D);
		
		//unbind texture.
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, 0);

		return(texture[0]);
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

		if(vbo_pyramid_texture[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_pyramid_texture, 0);
			vbo_pyramid_texture[0] = 0;
		}

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

		if(vbo_cube_texture[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_cube_texture, 0);
			vbo_cube_texture[0] = 0;
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
