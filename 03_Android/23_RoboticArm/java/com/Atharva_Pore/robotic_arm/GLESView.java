package com.Atharva_Pore.robotic_arm;

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
	
	private int Shoulder = 0;
	private int Elbow = 0;
	private int Palm = 0;

	private int iCount = 1; 

	private float[] perspectiveProjectionMatrix = new float[16];		//16 because it is 4 x 4 matrix.

	//sphere related variables
	private int numVertices;
	private int numElements;

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
		iCount += 1;
		if(iCount > 2)
		{
			iCount = 1;
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
		
				"uniform mat4 u_model_matrix;" +
				"uniform mat4 u_view_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				
				"void main(void)" +
				"{" +
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
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = vec4(0.5, 0.35, 0.05, 1.0);" +
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
		
		model_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_model_matrix");
		view_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_view_matrix");
		projection_uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");

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
		//code
		if(iCount == 1)
		{
			Elbow = (Elbow + 1) % 360;
			Shoulder = (Shoulder + 1) % 360;
		}
		if(iCount == 2)
		{
			Elbow = (Elbow - 1) % 360;
			Shoulder = (Shoulder - 1) % 360;
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
		float[] scaleMatrix = new float[16];
		
		//SHOULDER

		//make all matrices indentity.
		Matrix.setIdentityM(modelMatrix, 0);
		//Matrix.setIdentityM(projectionMatrix, 0);
		
		//do matrix multiplication
		Matrix.setIdentityM(viewMatrix, 0);
		Matrix.setLookAtM(	viewMatrix, 0,
							0.0f, 0.0f, 9.0f, 
							0.0f, 0.0f, 0.0f,
							0.0f, 1.0f, 0.0f);
		
		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.translateM(	translationMatrix, 0,
							0.0f, 
							0.0f, 
							0.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							translationMatrix, 0);

		Matrix.setIdentityM(rotationMatrix, 0);
		Matrix.rotateM(	rotationMatrix, 0,
						(float)Shoulder,
						0.0f, 
						0.0f, 
						1.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							rotationMatrix, 0);

		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.translateM(	translationMatrix, 0,
							1.0f, 
							0.0f, 
							0.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							translationMatrix, 0);

		Matrix.setIdentityM(scaleMatrix, 0);
		Matrix.scaleM(	scaleMatrix, 0,
						2.0f, 
						0.5f, 
						1.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							scaleMatrix, 0);

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
									perspectiveProjectionMatrix, 0);

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
		
		//ELBOW
        
		//make matrices indentity.
		Matrix.setIdentityM(modelMatrix, 0);
		
		//do matrix multiplication
		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.translateM(	translationMatrix, 0,
							0.0f, 
							0.0f, 
							0.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							translationMatrix, 0);

		Matrix.setIdentityM(rotationMatrix, 0);
		Matrix.rotateM(	rotationMatrix, 0,
						(float)Shoulder,
						0.0f, 
						0.0f, 
						1.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							rotationMatrix, 0);

		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.translateM(	translationMatrix, 0,
							2.0f, 
							0.0f, 
							0.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							translationMatrix, 0);

		Matrix.setIdentityM(rotationMatrix, 0);
		Matrix.rotateM(	rotationMatrix, 0,
						(float)Elbow,
						0.0f, 
						0.0f, 
						1.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							rotationMatrix, 0);

		Matrix.setIdentityM(translationMatrix, 0);
		Matrix.translateM(	translationMatrix, 0,
							1.0f, 
							0.0f, 
							0.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							translationMatrix, 0);

		Matrix.setIdentityM(scaleMatrix, 0);
		Matrix.scaleM(	scaleMatrix, 0,
						2.0f, 
						0.5f, 
						1.0f);
		Matrix.multiplyMM(	modelMatrix, 0,
							modelMatrix, 0,
							scaleMatrix, 0);

		//send this data to shader
		GLES32.glUniformMatrix4fv(	model_uniform,
									1,
									false,
									modelMatrix, 0);
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
}
