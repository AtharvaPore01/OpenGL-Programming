package com.Atharva_Pore.dynamic_india;

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
	private int[] vao_middleStrips = new int[1];
	private int[] vao_plane = new int[1];

	private int[] vbo_I_position				=	new int[1];	
	private int[] vbo_I_color					=	new int[1];	
	private int[] vbo_N_position				=	new int[1];	
	private int[] vbo_N_color					=	new int[1];	
	private int[] vbo_D_position				=	new int[1];	
	private int[] vbo_D_color					=	new int[1];	
	private int[] vbo_i_position				=	new int[1];	
	private int[] vbo_i_color					=	new int[1];	
	private int[] vbo_A_position				=	new int[1];	
	private int[] vbo_A_color					=	new int[1];
	private int[] vbo_middleStrips_position 	= 	new int[1];
	private int[] vbo_middleStrips_color 		= 	new int[1];
	private int[] vbo_plane_position 			= 	new int[1];
	private int[] vbo_plane_color 				= 	new int[1];	

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

	//flags
	boolean b_I_Done = false;
	boolean b_N_Done = false;
	boolean b_D_Done = false;
	boolean b_i_Done = false;
	boolean b_A_Done = false;
	boolean b_clip_top_plane = false;
	boolean b_clip_bottom_plane = false;
	boolean b_unclip_top_plane = false;
	boolean b_unclip_bottom_plane = false;

	boolean b_appear_middle_strip = true;

	boolean b_top_plane_smoke_done = false;
	boolean b_bottom_plane_smoke_done = false;
	boolean b_middle_plane_smoke_done = false;

	boolean b_start_decrementing = false;
	boolean b_start_incrementing = false;
	boolean b_PlaneTrue = false;

	//variables for translation of I,N,D,i,A
	float f_Translate_I = -3.0f;		//translate along x
	float f_Translate_N = 3.0f;		//translate along y
	float f_Translate_D = 0.0f;
	float f_Translate_i = -3.0f;		//translate along y
	float f_Translate_A = 3.0f;		//translate along x

	//color values for D
	float f_DRedColor = 0.0f;
	float f_DGreenColor = 0.0f;
	float f_DBlueColor = 0.0f;

	//A middle strips colors
	float f_ARedColor = 0.0f;
	float f_AGreenColor = 0.0f;
	float f_ABlueColor = 0.0f;
	float f_AWhiteColor = 0.0f;

	//smoke colors
	float f_red = 1.0f;
	float f_green = 0.5f;
	float f_blue = 1.0f;
	float f_white = 1.0f;

	/* angles to draw a smoke */
	float top_angle_1 = 3.14159f;
	float top_angle_2 = 3.14659f;

	float top_angle_3 = 4.71238f;
	float top_angle_4 = 4.71738f;

	float bottom_angle_1 = 3.13659f;
	float bottom_angle_2 = 3.14159f;

	float bottom_angle_3 = 1.57079f;
	float bottom_angle_4 = 1.56579f;

	//plane initial position variable
	float x_plane_pos = 0.0f;
	float y_plane_pos = 0.0f;

	float middle_plane_smoke = 0.0f;

	private class Plane
	{
		float _x;
		float _y;
		float radius = 10.0f;
		float angle = (float)M_PI;
		float rotation_angle = 0.0f;
	}

	Plane top = new Plane();
	Plane bottom = new Plane();

	double M_PI = 3.14159265358979323846;
	double M_PI_2 = 1.57079632679489661923;

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
			1.05f, -0.7f, 0.0f
		};

		final float[] planeVertices = new float[]
		{
			/* Vertices */
			//body
			2.0f, 0.35f, 0.0f,									//0
			-1.0f, 0.3f, 0.0f,									//1
			-1.0f, -0.3f, 0.0f,									//2										
			2.0f, -0.35f, 0.0f,									//3											

			//exahaust
			-0.3f, 0.0f, 0.0f,									//4
			-1.2f, 0.4f, 0.0f,									//5
			-1.2f, -0.4f, 0.0f,									//6

			//orange 
			-1.2f, 0.3f, 0.0f,
			-2.5f, 0.3f, 0.0f,
			-2.5f, 0.1f, 0.0f,
			-1.2f, 0.1f, 0.0f,

			//white
			-1.2f, 0.1f, 0.0f,
			-2.5f, 0.1f, 0.0f,
			-2.5f, -0.1f, 0.0f,
			-1.2f, -0.1f, 0.0f,

			//green
			-1.2f, -0.1f, 0.0f,
			-2.5f, -0.1f, 0.0f,
			-2.5f, -0.3f, 0.0f,
			-1.2f, -0.3f, 0.0f,

			//separator line between exhaust and body
			-1.0f, 0.15f, 0.0f,									//7
			-1.0f, -0.15f, 0.0f,								//8

			//front tip
			2.8f, 0.0f, 0.0f,									//9
			2.0f, 0.35f, 0.0f,									//10
			2.0f, -0.35f, 0.0f,									//11

			//sperator line between front tip and body
			2.0f, 0.35f, 0.0f,									//12
			2.0f, -0.35f, 0.0f,									//13

			//upper wing
			1.5f, 0.32f, 0.0f,									//14
			-0.6f, 1.5f, 0.0f,									//15
			-0.6f, 0.22f, 0.0f,									//16

			//lower wing
			1.5f, -0.32f, 0.0f,									//17
			-0.6f, -1.5f, 0.0f,									//18
			-0.6f, -0.22f, 0.0f,								//19

			//IAF Letters
			/* Vertices */
			//1. I
			-0.0f, 0.15f, 0.0f,									//20
			-0.0f, -0.15f, 0.0f,								//21

			//2. A
			0.2f, 0.15f, 0.0f,									//22
			0.1f, -0.15f, 0.0f,									//23

			0.2f, 0.15f, 0.0f,									//24
			0.3f, -0.15f, 0.0f,									//25

			0.15f, 0.0f, 0.0f,									//26
			0.25f, 0.0f, 0.0f,									//27

			//3. F
			0.4f, 0.15f, 0.0f,									//28
			0.4f, -0.15f, 0.0f,									//29

			0.4f, 0.15f, 0.0f,									//30
			0.55f, 0.15f, 0.0f,									//31

			0.4f, 0.0f, 0.0f,									//32
			0.5f, 0.0f, 0.0f									//33
		};

		final float[] middleStrips_vertices = new float[]
		{
			//orange 
			-1.2f, 0.3f, 0.0f,
			-2.5f, 0.3f, 0.0f,
			-2.5f, 0.1f, 0.0f,
			-1.2f, 0.1f, 0.0f,

			//white
			-1.2f, 0.1f, 0.0f,
			-2.5f, 0.1f, 0.0f,
			-2.5f, -0.1f, 0.0f,
			-1.2f, -0.1f, 0.0f,

			//green
			-1.2f, -0.1f, 0.0f,
			-2.5f, -0.1f, 0.0f,
			-2.5f, -0.3f, 0.0f,
			-1.2f, -0.3f, 0.0f
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
		};

		final float[] planeColor = new float[]
		{
			/* color */
			//body
			0.7294117f, 0.8862745f, 0.9333333f,									//0
			0.7294117f, 0.8862745f, 0.9333333f,									//1
			0.7294117f, 0.8862745f, 0.9333333f,									//2										
			0.7294117f, 0.8862745f, 0.9333333f,									//3											

			//exahaust
			0.7294117f, 0.8862745f, 0.9333333f,									//4
			0.7294117f, 0.8862745f, 0.9333333f,									//5
			0.7294117f, 0.8862745f, 0.9333333f,									//6

			//orange
			1.0f, 0.5f, 0.0f,	//right top										//7
			0.0f, 0.0f, 0.0f,	//right bottom									//8
			0.0f, 0.0f, 0.0f,	//left top										//9
			1.0f, 0.5f, 0.0f,	//left top										//10

			//white
			1.0f, 1.0f, 1.0f,	//right top							//11	
			0.0f, 0.0f, 0.0f,	//right bottom		//12
			0.0f, 0.0f, 0.0f,	//left top			13
			1.0f, 1.0f, 1.0f,	//left top			14

			//green
			0.0f, 0.5f, 0.0f,	//right top			15
			0.0f, 0.0f, 0.0f,	//right bottom		16
			0.0f, 0.0f, 0.0f,	//left top			17
			0.0f, 0.5f, 0.0f,	//left top			18


			//separator line between exhaust and body
			0.0f, 0.0f, 0.0f,													//19
			0.0f, 0.0f, 0.0f,													//20

			//front tip
			0.7294117f, 0.8862745f, 0.9333333f,									//21
			0.7294117f, 0.8862745f, 0.9333333f,									//22
			0.7294117f, 0.8862745f, 0.9333333f,									//23

			//sperator line between front tip and body
			0.0f, 0.0f, 0.0f,													//24
			0.0f, 0.0f, 0.0f,													//25

			//upper wing
			0.7294117f, 0.8862745f, 0.9333333f,									//26
			0.7294117f, 0.8862745f, 0.9333333f,									//27
			0.7294117f, 0.8862745f, 0.9333333f,									//28

			//lower wing
			0.7294117f, 0.8862745f, 0.9333333f,									//29
			0.7294117f, 0.8862745f, 0.9333333f,									//30
			0.7294117f, 0.8862745f, 0.9333333f,									//31

			//IAF Letters
			/* color */
			//1. I
			0.0f, 0.0f, 0.0f,													//32
			0.0f, 0.0f, 0.0f,													//33

			//2. A
			0.0f, 0.0f, 0.0f,													//34
			0.0f, 0.0f, 0.0f,													//35

			0.0f, 0.0f, 0.0f,													//36
			0.0f, 0.0f, 0.0f,													//37

			0.0f, 0.0f, 0.0f,													//38
			0.0f, 0.0f, 0.0f,													//39

			//3. F
			0.0f, 0.0f, 0.0f,													//40
			0.0f, 0.0f, 0.0f,													//41

			0.0f, 0.0f, 0.0f,													//42
			0.0f, 0.0f, 0.0f,													//43

			0.0f, 0.0f, 0.0f,													//44
			0.0f, 0.0f, 0.0f,													//45
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
/*
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
*/
		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								16 * 3 * 4,
								null,
								GLES32.GL_DYNAMIC_DRAW);

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

		//middlestrips

		//position
	
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_middleStrips, 0);
		GLES32.glBindVertexArray(vao_middleStrips[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_middleStrips_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_middleStrips_position[0]);

		//now from here below we are doing 5 steps to convert the I_vertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_middleStrips_position = ByteBuffer.allocateDirect(middleStrips_vertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_middleStrips_position.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_middleStrips = byteBuffer_middleStrips_position.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_middleStrips.put(middleStrips_vertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_middleStrips.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								middleStrips_vertices.length * 4,
								positionBuffer_middleStrips,
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
		GLES32.glGenBuffers(1, vbo_middleStrips_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_middleStrips_color[0]);
/*
		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_middleSyrips_color = ByteBuffer.allocateDirect(.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_middleSyrips_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_middleStrips = byteBuffer_middleSyrips_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_middleStrips.put(A_color);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_middleStrips.position(0);
*/
		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								12 * 3 * 4,
								null,
								GLES32.GL_DYNAMIC_DRAW);

		GLES32.glVertexAttribPointer(	GLESMacros.AMC_ATTRIBUTE_COLOR,
										3,
										GLES32.GL_FLOAT,
										false,
										0,
										0);
		
		GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		//plane

		//position
	
		//create and bind vao
		GLES32.glGenVertexArrays(1, vao_plane, 0);
		GLES32.glBindVertexArray(vao_plane[0]);
		//create and bind vbo
		GLES32.glGenBuffers(1, vbo_plane_position, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_plane_position[0]);

		//now from here below we are doing 5 steps to convert the I_vertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_plane_position = ByteBuffer.allocateDirect(planeVertices.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_plane_position.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer positionBuffer_plane = byteBuffer_plane_position.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		positionBuffer_plane.put(planeVertices);

		//step 5: set the array at 0th position of the buffer.
		positionBuffer_plane.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								planeVertices.length * 4,
								positionBuffer_plane,
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
		GLES32.glGenBuffers(1, vbo_plane_color, 0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_plane_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_plane_color = ByteBuffer.allocateDirect(planeColor.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_plane_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_plane = byteBuffer_plane_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_plane.put(planeColor);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_plane.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								planeColor.length * 4,
								colorBuffer_plane,
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

		//init dynamic india data
		oglInitData();

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
		float[] rotationMatrix = new float[16];

		//init above metrices to identity
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		Matrix.setIdentityM(translationMatrix, 0);
		
		//do necessary matrix multiplication
		Matrix.translateM(	translationMatrix, 0, 
							f_Translate_I,
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

		if(b_I_Done)
		{
			//init above metrices to identity
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			
			//do necessary matrix multiplication
			Matrix.translateM(	translationMatrix, 0, 
								f_Translate_A,
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
			oglDraw_A();

			//init above metrices to identity
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			
			//do necessary matrix multiplication
			Matrix.translateM(	translationMatrix, 0, 
								7.59f,
								0.0f,
								-23.0f);
			
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
			oglDraw_middleStrips();
		}

		if(b_A_Done)
		{
			//init above metrices to identity
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			
			//do necessary matrix multiplication
			Matrix.translateM(	translationMatrix, 0, 
								0.0f,
								f_Translate_N,
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
			oglDraw_N();
		}

		if(b_N_Done)
		{
			//init above metrices to identity
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			
			//do necessary matrix multiplication
			Matrix.translateM(	translationMatrix, 0, 
								0.0f,
								f_Translate_i,
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
			oglDraw_i();
		}

		if(b_i_Done)
		{
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
			oglDraw_D();
		}

		if(b_D_Done)
		{
			if(b_clip_top_plane == false)
			{
				//init above metrices to identity
				Matrix.setIdentityM(modelViewMatrix, 0);
				Matrix.setIdentityM(modelViewProjectionMatrix, 0);
				Matrix.setIdentityM(translationMatrix, 0);
				Matrix.setIdentityM(rotationMatrix, 0);
				
				//do necessary matrix multiplication
				Matrix.translateM(	translationMatrix, 0, 
									0.0f,
									0.0f,
									-3.0f);
				Matrix.translateM(	translationMatrix, 0, 
									top._x,
									top._y,
									-20.0f);
				Matrix.setRotateM(	rotationMatrix, 0,
									top.rotation_angle,
									0.0f,
									0.0f,
									1.0f);
				
				Matrix.multiplyMM(	modelViewMatrix, 0,
									modelViewMatrix, 0,
									translationMatrix, 0);

				Matrix.multiplyMM(	modelViewMatrix, 0,
									modelViewMatrix, 0,
									rotationMatrix, 0);

				Matrix.multiplyMM(	modelViewProjectionMatrix, 0,
									perspectiveProjectionMatrix, 0,
									modelViewMatrix, 0);
				//send necessary data to shader
				GLES32.glUniformMatrix4fv(	mvpUniform,
											1,
											false,
											modelViewProjectionMatrix, 0);
				oglDraw_plane();
			}

			//init above metrices to identity
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			
			//do necessary matrix multiplication
			Matrix.translateM(	translationMatrix, 0, 
								0.0f,
								0.0f,
								-20.0f);
			
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

			if(b_clip_bottom_plane == false)
			{
				//init above metrices to identity
				Matrix.setIdentityM(modelViewMatrix, 0);
				Matrix.setIdentityM(modelViewProjectionMatrix, 0);
				Matrix.setIdentityM(translationMatrix, 0);
				Matrix.setIdentityM(rotationMatrix, 0);
				
				//do necessary matrix multiplication
				Matrix.translateM(	translationMatrix, 0, 
									bottom._x,
									bottom._y,
									-20.0f);
				Matrix.setRotateM(	rotationMatrix, 0,
									bottom.rotation_angle,
									0.0f,
									0.0f,
									1.0f);
				
				Matrix.multiplyMM(	modelViewMatrix, 0,
									modelViewMatrix, 0,
									translationMatrix, 0);

				Matrix.multiplyMM(	modelViewMatrix, 0,
									modelViewMatrix, 0,
									rotationMatrix, 0);

				Matrix.multiplyMM(	modelViewProjectionMatrix, 0,
									perspectiveProjectionMatrix, 0,
									modelViewMatrix, 0);
				//send necessary data to shader
				GLES32.glUniformMatrix4fv(	mvpUniform,
											1,
											false,
											modelViewProjectionMatrix, 0);
				oglDraw_plane();
			}

			//init above metrices to identity
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);
			Matrix.setIdentityM(translationMatrix, 0);
			
			//do necessary matrix multiplication
			Matrix.translateM(	translationMatrix, 0, 
								x_plane_pos,
								0.0f,
								-20.0f);
			
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
			oglDraw_plane();
		}

		//unuse program
		GLES32.glUseProgram(0);
		requestRender();
	}

	private void oglUpdate()
	{
		//init translation Of I
		if (f_Translate_I <= 0.0f)
		{
			f_Translate_I = f_Translate_I + 0.0045f;
			if (f_Translate_I > 0.0f)
			{
				b_I_Done = true;
			}
		}

		//init translation A
		if (b_I_Done)
		{
			if (f_Translate_A > 0.0f)
			{
				f_Translate_A = f_Translate_A - 0.0041f;
				if (f_Translate_A < 0.0f)
				{
					b_A_Done = true;
				}
			}
		}

		//init translation N
		if (b_A_Done == true)
		{
			if (f_Translate_N > 0.0f)
			{
				f_Translate_N = f_Translate_N - 0.006f;
				if (f_Translate_N < 0.0f)
				{
					b_N_Done = true;
				}
			}
		}

		//init translation i
		if (b_N_Done == true)
		{
			if (f_Translate_i < 0.0f)
			{
				f_Translate_i = f_Translate_i + 0.00455f;
				if (f_Translate_i > 0.0f)
				{
					b_i_Done = true;
				}
			}
		}

		//init color transition of D
		if (b_i_Done == true)
		{
			if ((f_DRedColor <= 1.0f) && (f_DGreenColor <= 0.5f))
			{
				f_DRedColor += 0.00032f;
				f_DGreenColor += 0.00016f;

				if ((f_DRedColor > 1.0f) && (f_DGreenColor > 0.5f))
				{
					b_D_Done = true;
				}
			}
		}

		//plane transition
		if (b_D_Done == true)
		{
			if (x_plane_pos <= 22.0f)
			{
				x_plane_pos = x_plane_pos + 0.0445f;
				/* top plane */
				if (top.angle <= (float)(M_PI + M_PI_2))
				{
					top._x = top.radius * (float)Math.cos(top.angle) - 9.1f;
					top._y = top.radius * (float)Math.sin(top.angle) + 10.0f;
					top.angle += 0.005f;
					if (top.angle > (float)(M_PI + M_PI_2))
					{
						b_clip_top_plane = true;
					}
					if (top_angle_2 <= 4.71238f)
					{
						//top_angle_2 = top_angle_2 + 0.00475f;
						top_angle_2 = top_angle_2 + 0.000475f;
					}
				}
				//rotation angle calculation
				if (top.rotation_angle <= 0.0f)
				{
					top.rotation_angle += 0.21f;
				}

				/* bottom plane */
				if (bottom.angle >= M_PI_2)
				{
					bottom._x = bottom.radius * (float)Math.cos(bottom.angle) - 8.0f;
					bottom._y = bottom.radius * (float)Math.sin(bottom.angle) - 10.0f;
					bottom.angle -= 0.005f;
					if (bottom.angle <= M_PI_2)
					{
						b_clip_bottom_plane = true;
					}
					if (bottom_angle_2 <= M_PI_2)
					{
						bottom_angle_2 = bottom_angle_2 + 0.000475f;
					}
				}
				//rotation angle calculation
				if (bottom.rotation_angle >= 0.0f)
				{
					bottom.rotation_angle -= 0.21f;
				}

				if (x_plane_pos > 8.0f)
				{
					b_clip_top_plane = false;
					b_clip_bottom_plane = false;
					
					b_start_incrementing = true;
					b_start_decrementing = true;

					if (top.angle <= 2 * M_PI)
					{
						top._x = top.radius * (float)Math.cos(top.angle) + 9.1f;
						top._y = top.radius * (float)Math.sin(top.angle) + 10.0f;
						top.angle += 0.005f;

						if (top_angle_4 <= 6.28318f)
						{
							top_angle_4 += 0.00038f;
						}

						if (top.angle > 2 * M_PI)
						{
							b_clip_top_plane = true;
						}
						//angle related calculation
						top.rotation_angle += 0.21f;
					}
					
					
					if (bottom.angle >= 0.0f)
					{
						bottom._x = bottom.radius * (float)Math.cos(bottom.angle) + 8.0f;
						bottom._y = bottom.radius * (float)Math.sin(bottom.angle) - 10.0f;
						bottom.angle -= 0.005f;
						if (bottom.angle <= 0.0f)
						{
							b_clip_bottom_plane = true;
						}
						if (bottom_angle_4 >= 0.0f)
						{
							bottom_angle_4 -= 0.00038f;
						}
						//angle related calculation
						bottom.rotation_angle -= 0.21f;

					}
					if (x_plane_pos > 22.0f)
					{
						b_PlaneTrue = true;
						b_clip_top_plane = true;
						b_clip_bottom_plane = true;
					}
				}
			}
		}

		if (b_PlaneTrue)
		{
			if ((f_ARedColor <= 1.0f) && (f_AGreenColor <= 0.5f) && (f_ABlueColor <= 1.0f) && (f_AWhiteColor <= 1.0f))
			{
				f_ARedColor = f_ARedColor + 0.002f;
				f_AGreenColor = f_AGreenColor + 0.001f;
				f_ABlueColor = f_ABlueColor + 0.002f;
				f_AWhiteColor = f_AWhiteColor + 0.002f;
			}
		}
	}

	private void oglInitData()
	{
		//code
		top.rotation_angle = -60.0f;
		bottom.rotation_angle = 60.0f;
		x_plane_pos = -22.0f;		//plane starting position
		
		top.radius = 10.0f;
		top.angle = (float)M_PI;

		bottom.radius = 10.0f;
		bottom.angle = (float)M_PI;
		
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
		//variable declaration
		float[] D_color = new float[]
		{
			f_DRedColor, f_DGreenColor, 0.0f,
			f_DRedColor, f_DGreenColor, 0.0f,
			f_DRedColor, f_DGreenColor, 0.0f,
			f_DRedColor, f_DGreenColor, 0.0f,

			0.0f, f_DGreenColor, 0.0f,
			0.0f, f_DGreenColor, 0.0f,
			0.0f, f_DGreenColor, 0.0f,
			0.0f, f_DGreenColor, 0.0f,

			f_DRedColor, f_DGreenColor, 0.0f,
			f_DRedColor, f_DGreenColor, 0.0f,
			0.0f, f_DGreenColor, 0.0f,
			0.0f, f_DGreenColor, 0.0f,

			f_DRedColor, f_DGreenColor, 0.0f,
			f_DRedColor, f_DGreenColor, 0.0f,
			0.0f, f_DGreenColor, 0.0f,
			0.0f, f_DGreenColor, 0.0f
		};

		//code
		GLES32.glBindVertexArray(vao_D[0]);

		//color
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
								GLES32.GL_DYNAMIC_DRAW);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


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

	private void oglDraw_middleStrips()
	{
		//variable declaration
		float[] middleStrips_color = new float[]
		{
			f_ARedColor, f_AGreenColor, 0.0f,
			f_ARedColor, f_AGreenColor, 0.0f,
			f_ARedColor, f_AGreenColor, 0.0f,
			f_ARedColor, f_AGreenColor, 0.0f,

			f_ARedColor, f_AWhiteColor, f_ABlueColor,
			f_ARedColor, f_AWhiteColor, f_ABlueColor,
			f_ARedColor, f_AWhiteColor, f_ABlueColor,
			f_ARedColor, f_AWhiteColor, f_ABlueColor,

			0.0f, f_AGreenColor, 0.0f,
			0.0f, f_AGreenColor, 0.0f,
			0.0f, f_AGreenColor, 0.0f,
			0.0f, f_AGreenColor, 0.0f
		};
		//code
		GLES32.glBindVertexArray(vao_middleStrips[0]);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_middleStrips_color[0]);

		//now from here below we are doing 5 steps to convert the triangleVertices array in buffer(compatible to send to glBufferData() function)

		//step 1: allocate the buffer directly from native memory(unmanaged memory)
		ByteBuffer byteBuffer_middleStrips_color = ByteBuffer.allocateDirect(middleStrips_color.length * 4);	//java doesn't have sizeof() so, size of an array = array length * size of type 

		//step 2: arrange the byte order of buffer in native byte order
		byteBuffer_middleStrips_color.order(ByteOrder.nativeOrder());

		//step 3: create the float type buffer and convert our byte type buffer in float
		FloatBuffer colorBuffer_middleStrips = byteBuffer_middleStrips_color.asFloatBuffer();

		//step 4: now put your array in this COOKED buffer.
		colorBuffer_middleStrips.put(middleStrips_color);

		//step 5: set the array at 0th position of the buffer.
		colorBuffer_middleStrips.position(0);

		GLES32.glBufferData(	GLES32.GL_ARRAY_BUFFER,
								middleStrips_color.length * 4,
								colorBuffer_middleStrips,
								GLES32.GL_DYNAMIC_DRAW);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


		//draw scene
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 8, 4);
		

		//unbind vao
		GLES32.glBindVertexArray(0);
	}

	private void oglDraw_plane()
	{
		//code

		GLES32.glBindVertexArray(vao_plane[0]);

		//draw scene
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);	//body
		GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 4, 3);		//exahaust
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 7, 4);	//orange flag
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 11, 4);	//white	flag
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 15, 4);	//green	flag
		GLES32.glDrawArrays(GLES32.GL_LINES, 19, 2);			//separator line between exhaust and body
		GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 21, 3);		//front tip
		GLES32.glDrawArrays(GLES32.GL_LINES, 24, 2);			//sperator line between front tip and body
		GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 26, 3);		//upper wing
		GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 29, 3);		//lower wing

		//IAF
		GLES32.glDrawArrays(GLES32.GL_LINES, 32, 2);			//I

		GLES32.glDrawArrays(GLES32.GL_LINES, 34, 2);			//A
		GLES32.glDrawArrays(GLES32.GL_LINES, 36, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 38, 2);

		GLES32.glDrawArrays(GLES32.GL_LINES, 40, 2);			//F
		GLES32.glDrawArrays(GLES32.GL_LINES, 42, 2);
		GLES32.glDrawArrays(GLES32.GL_LINES, 44, 2);

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
		if (vbo_A_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_A_color, 0);
			vbo_A_color[0] = 0;
		}

		if (vbo_plane_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_plane_position, 0);
			vbo_plane_position[0] = 0;
		}
		if (vbo_plane_color[0] != 0)
		{
			GLES32.glDeleteBuffers(1, vbo_plane_color, 0);
			vbo_plane_color[0] = 0;
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
