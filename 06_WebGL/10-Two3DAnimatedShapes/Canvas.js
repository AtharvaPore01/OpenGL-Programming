//global variables
var canvas 	= null;
var gl = null;	//webgl context
var bFullscreen = false;
var canvas_original_width;
var canvas_original_height;

// in webGL this called as key-value coding. 
const WebGLMacros = //when whole WebGLMacros Are const then whole inside it are automatically const
{
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,	
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3
}

//shader and program objectes
var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

//vao and vbo declaration
var vao_pyramid;
var vao_cube;
var vbo_pyramid_position;
var vbo_pyramid_color;
var vbo_cube_position;
var vbo_cube_color;
var mvpUniform;

//rotation related variables
var anglePyramid = 0.0;
var angleCube = 0.0;

//declaration of perspective matrix.
var perspectiveMatrixProjection;

//To start animation : To Have requestAnimationFrame() to be called "cross-browser" compatible
var requestAnimationFrame = 
window.requestAnimationFrame || 
window.webkitRequestAnimationFrame ||
window.mozRequestAnimationFrame ||
window.oRequestAnimationFrame ||
window.msRequestAnimationFrame ||
null;

//To Stop animation : To Have cancelAnimationFrame() to be called "cross=browser" compatible
var cancelAnimationFrame = 
window.cancelAnimationFrame || window.cancelRequestAnimationFrame ||
window.webkitCancelAnimatinFrame || window.webkitCancelRequestAnimationFrame ||
window.mozCancelAnimationFrame || window.mozCancelRequestAnimationFrame ||
window.oCancelAnimationFrame || window.oCancelRequestAnimationFrame || 
window.msCancelAnimationFrame || window.msCancelRequestAnimationFrame ||
null;

//onload function
function main()
{
	//get <canvas> element
	canvas = document.getElementById("AAP");
	if(!canvas)
		console.log("Obtaining Canvas Failed\n");
	else
		console.log("Obtaining Canvas Succeeded\n");

	//assigning width and height to global variables
	canvas_original_width = canvas.width;
	canvas_original_height = canvas.height;

	//register keyboard's keydown handle
	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
	window.addEventListener("resize", wglResize, false);

	//initialise webGL
	wglInit();
	
	//start drawing here
	wglResize();
	wglDraw();
}

//event handlers
function keyDown(event)
{
	//code
	switch(event.keyCode)
	{
		case 70: 	
			toggleFullScreen();
			break;
		case 27:
			wglUnintialise();
			window.close();
			break;
	}
}

function mouseDown()
{
	//code
}

function wglInit()
{
	//code
	gl = canvas.getContext("webgl2");
	if(gl == null)
	{
		console.log("Failed to get the rendering context for WebGL");
		return;
	}

	gl.viewportWidth = canvas.width;
	gl.viewportHeight = canvas.height;

	//vertex shader
	vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);

	var vertexShaderSourceCode = 
	"#version 300 es" +
	"\n" +
	"in vec4 vPosition;" +
	"in vec4 vColor;" +
	"uniform mat4 u_mvp_matrix;" +
	"out vec4 out_color;" +
	"void main(void)" +
	"{" +
		"gl_Position = u_mvp_matrix * vPosition;" +
		"out_color = vColor;" +
	"}"; 

	gl.shaderSource(vertexShaderObject, vertexShaderSourceCode);
	gl.compileShader(vertexShaderObject);
	if(gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS) == false)
	{
		var error = gl.getShaderInfoLog(vertexShaderObject);
		if(error.length > 0)
		{
			console.log("vertex shader error.\n");
			alert(error);
			wglUnintialise();
		}
	}

	//fragment shader
	fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);

	var fragmentShaderSourceCode = 
	"#version 300 es" +
	"\n" +
	"precision highp float;" +
	"in vec4 out_color;" +
	"out vec4 FragColor;" +
	"void main(void)" +
	"{" +
		"FragColor = out_color;" +
	"}";

	gl.shaderSource(fragmentShaderObject, fragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject);
	if(gl.getShaderParameter(fragmentShaderObject, gl.COMPILE_STATUS) ==  false)
	{
		var error = gl.getShaderInfoLog(fragmentShaderObject);
		if(error.length > 0)
		{
			console.log("fragment shader error.\n");
			alert(error);
			wglUnintialise();
		}
	}


	//shader program
	shaderProgramObject = gl.createProgram();

	gl.attachShader(shaderProgramObject, vertexShaderObject);
	gl.attachShader(shaderProgramObject, fragmentShaderObject);

	//pre link binding
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vColor");

	//linking
	gl.linkProgram(shaderProgramObject);
	if(!gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS))
	{
		var error = gl.getProgramInfoLog(shaderProgramObject);
		if(error.length > 0)
		{
			alert(error);
			wglUnintialise();
		}
	}

	//get MVP uniform location
	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");

	//pyramid vertices
	var pyramidVertice = new Float32Array([
											0.0, 1.0, 0.0,
											-1.0, -1.0, 1.0,
											1.0, -1.0, 1.0,
											
											0.0, 1.0, 0.0,
											1.0, -1.0, 1.0,
											1.0, -1.0, -1.0,

											0.0, 1.0, 0.0,
											1.0, -1.0, -1.0,
											-1.0, -1.0, -1.0,

											0.0, 1.0, 0.0, 
											-1.0, -1.0, -1.0, 
											-1.0, -1.0, 1.0
										]);
	var pyramidColor = new Float32Array ([
											1.0, 0.0, 0.0,		//red
											0.0, 1.0, 0.0, 		//green
											0.0, 0.0, 1.0,		//blue

											1.0, 0.0, 0.0,
											0.0, 0.0, 1.0,
											0.0, 1.0, 0.0,
									
											1.0, 0.0, 0.0,
											0.0, 1.0, 0.0,
											0.0, 0.0, 1.0,
									
											1.0, 0.0, 0.0,
											0.0, 0.0, 1.0,
											0.0, 1.0, 0.0	
										]);

	var cubeVertices = new Float32Array	([
											1.0, 1.0, 0.0,		//right top
											-1.0, 1.0, 0.0,		//left top
											-1.0, -1.0, 0.0,	//left bottom
											1.0, -1.0, 0.0,		//right bottom

											1.0, -1.0, -1.0,
											-1.0, -1.0, -1.0,
											-1.0, -1.0, 1.0,
											1.0, -1.0, 1.0,
									
											1.0, 1.0, 1.0,
											-1.0, 1.0, 1.0, 
											-1.0, -1.0, 1.0, 
											1.0, -1.0, 1.0,
									
											1.0, 1.0, -1.0, 
											-1.0, 1.0, -1.0,
											- 1.0, -1.0, -1.0,
											1.0, -1.0, -1.0,
									
											1.0, 1.0, -1.0,
											1.0, 1.0, 1.0,
											1.0, -1.0, 1.0,
											1.0, -1.0, -1.0,
									
											-1.0, 1.0, -1.0, 
											-1.0, 1.0, 1.0, 
											-1.0, -1.0, 1.0,
											-1.0, -1.0, -1.0
										]);
	var cubeColor = new Float32Array ([
											0.0, 0.0, 1.0,		//blue
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,

											0.0, 0.0, 1.0,		//blue
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,

											0.0, 0.0, 1.0,		//blue
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,

											0.0, 0.0, 1.0,		//blue
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,

											0.0, 0.0, 1.0,		//blue
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,

											0.0, 0.0, 1.0,		//blue
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
										]);											
	//pyramid
	vao_pyramid = gl.createVertexArray();
	gl.bindVertexArray(vao_pyramid);
	
	//position
	vbo_pyramid_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_pyramid_position);
	gl.bufferData(gl.ARRAY_BUFFER, pyramidVertice, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	//color
	vbo_pyramid_color = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_pyramid_color);
	gl.bufferData(gl.ARRAY_BUFFER, pyramidColor, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	//cube

	for (var i = 0; i < 72; i++)
	{
		if (cubeVertices[i] == -1.0)
		{
			cubeVertices[i] = cubeVertices[i] + 0.25;
		}
		else if (cubeVertices[i] == 1.0)
		{
			cubeVertices[i] = cubeVertices[i] - 0.25;
		}
	}

	vao_cube = gl.createVertexArray();
	gl.bindVertexArray(vao_cube);

	//position
	vbo_cube_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_cube_position);
	gl.bufferData(gl.ARRAY_BUFFER, cubeVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	//color
	vbo_cube_color = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_cube_color);
	gl.bufferData(gl.ARRAY_BUFFER, cubeColor, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	//set clear color
	gl.clearColor(0.0, 0.0, 0.0, 1.0);

	//depth test
	gl.enable(gl.DEPTH_TEST);

	//enable cull face
	//gl.enable(gl.CULL_FACE);

	//initialise projection matrix
	perspectiveMatrixProjection = mat4.create();
}

function wglResize()
{
	//code
	if(bFullscreen == true)
	{
		canvas.width = window.innerWidth;
		canvas.height = window.innerHeight;
	}
	else
	{
		canvas.width = canvas_original_width;
		canvas.height = canvas_original_height;
	}

	//set the viewport to match
	gl.viewport(0, 0, canvas.width, canvas.height);

	//perspective Projection Matrix
	mat4.perspective(	perspectiveMatrixProjection,
						45.0,
						parseFloat(canvas.width) / parseFloat(canvas.height),
						0.1,
						100.0);
	
}

function wglUpdate()
{
	//code
	anglePyramid = anglePyramid + 1.0;
	if(anglePyramid >= 360.0)
	{
		anglePyramid = 0.0
	}
	angleCube = angleCube + 1.0;
	if(angleCube >= 360.0)
	{
		angleCube = 0.0
	}
}

function wglDraw()
{
	//code
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	gl.useProgram(shaderProgramObject);

	/*pyramid*/

	//declare and initialise the matrices.
	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();

	//transformation
	mat4.translate(modelViewMatrix, modelViewMatrix, [-1.5, 0.0, -5.0]);
	mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(anglePyramid));

	//do necessary matrix multiplication
	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	//send necessary matrics to shaders in respective uniforms
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	//bind with vao
	gl.bindVertexArray(vao_pyramid);
	
	//draw scene
	gl.drawArrays(gl.TRIANGLES, 0, 12);

	//unbind with vao
	gl.bindVertexArray(null);

	/* cube */

	//make identity
	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	//transformation
	mat4.translate(modelViewMatrix, modelViewMatrix, [1.5, 0.0, -5.0]);
	mat4.rotateX(modelViewMatrix, modelViewMatrix, degToRad(angleCube));
	mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(angleCube));
	mat4.rotateZ(modelViewMatrix, modelViewMatrix, degToRad(angleCube));

	//do necessary matrix multiplication
	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	//send necessary matrics to shaders in respective uniforms
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	//bind with vao
	gl.bindVertexArray(vao_cube);
	
	//draw scene
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 8, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 12, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 16, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 20, 4);

	//unbind with vao
	gl.bindVertexArray(null);

	//unuse program
	gl.useProgram(null);

	//call update
	wglUpdate();

	//animation loop
	requestAnimationFrame(wglDraw, canvas);
}

function toggleFullScreen()
{
	//code
	var fullscreen_element = 
	document.fullscreenElement ||
	document.webkitFullscreenElement ||
	document.mozFullScreenElement ||
	document.msFullscreenElement ||
	null;

	//if not fullscreen
	if(fullscreen_element == null)
	{
		if(canvas.requestFullscreen)
			canvas.requestFullscreen();
		else if(canvas.mozRequestFullScreen)
			canvas.mozRequestFullScreen();
		else if(canvas.webkitRequestFullscreenElement)
			canvas.webkitRequestFullscreenElement();
		else if(canvas.msRequestFullscreenElement)
			canvas.msRequestFullscreenElement();

		bFullScreen = true;
	}
	else
	{
		if(document.exitFullscreen)
			document.exitFullscreen();
		else if(document.mozCancelFullScreen)
			document.mozCancelFullScreen();
		else if(document.webkitExitFullscreen)
			document.webkitExitFullscreen();
		else if(document.msExitFullscreen)
			document.msExitFullscreen();

		bFullScreen = false;
	}
}

function degToRad(degrees)
{
	var radians = 0.0;
	//code
	radians = degrees * Math.PI / 180;
	return(radians)
}

function wglUnintialise()
{
	//code
	if(vao_pyramid)
	{
		gl.deleteVertexArray(vao_pyramid);
		vao_pyramid = null;
	}

	if(vbo_pyramid_position)
	{
		gl.deleteBuffer(vbo_pyramid_position);
		vbo_pyramid_position = null;
	}

	if(vao_cube)
	{
		gl.deleteVertexArray(vao_cube);
		vao_cube = null;
	}

	if(vbo_cube_position)
	{
		gl.deleteBuffer(vbo_cube_position);
		vbo_cube_position = null;
	}

	if(vbo_cube_color)
	{
		gl.deleteBuffer(vbo_cube_color);
		vbo_cube_color = null;
	}

	if(shaderProgramObject)
	{
		if(fragmentShaderObject)
		{
			gl.detachShader(shaderProgramObject, fragmentShaderObject);
			gl.deleteShader(fragmentShaderObject);
			fragmentShaderObject = null;
		}
		if(vertexShaderObject)
		{
			gl.detachShader(shaderProgramObject, vertexShaderObject);
			gl.deleteShader(vertexShaderObject);
			vertexShaderObject = null;
		}

		gl.deleteProgram(shaderProgramObject);
		shaderProgramObject = null;
	}
}
