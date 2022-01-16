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
var vao_I;
var vao_N;
var vao_D;
var vao_i;
var vao_A;

var vbo_I_position;
var vbo_I_color;
var vbo_N_position;
var vbo_N_color;
var vbo_D_position;
var vbo_D_color;
var vbo_i_position;
var vbo_i_color;
var vbo_A_position;
var vbo_A_color;

var mvpUniform;

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
	"out vec4 out_color;" +
	"uniform mat4 u_mvp_matrix;" +
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

	//triangle vertices
	var triangleVertice = new Float32Array	([
												0.0, 1.0, 0.0,			//appex
												-1.0, -1.0, 0.0,		//left-bottom
												1.0, -1.0, 0.0		//right-bottom
											]);

	//vertices
	var I_vertices = new Float32Array	([
											-1.15, 0.7, 0.0,
											-1.25, 0.7, 0.0,
											-1.25, -0.7, 0.0,
											-1.15, -0.7, 0.0
										]);

	var N_vertices = new Float32Array	([
											-0.95, 0.7, 0.0,
											-1.05, 0.7, 0.0,
											-1.05, -0.7, 0.0,
											-0.95, -0.7, 0.0,

											-0.55, 0.7, 0.0,
											-0.65, 0.7, 0.0,
											-0.65, -0.7, 0.0,
											-0.55, -0.7, 0.0,

											-0.95, 0.7, 0.0,
											-0.95, 0.5, 0.0,
											-0.65, -0.7, 0.0,
											-0.65, -0.5, 0.0
										]);

	var D_vertices = new Float32Array	([
											0.15, 0.7, 0.0,
											-0.45, 0.7, 0.0,
											-0.45, 0.6, 0.0,
											0.15, 0.6, 0.0,
											//bottom
											0.15, -0.7, 0.0,
											-0.45, -0.7, 0.0,
											-0.45, -0.6, 0.0,
											0.15, -0.6, 0.0,
											//left
											0.15, 0.7, 0.0,
											0.05, 0.7, 0.0,
											0.05, -0.7, 0.0,
											0.15, -0.7, 0.0,
											//right
											-0.25, 0.6, 0.0,
											-0.35, 0.6, 0.0,
											-0.35, -0.6, 0.0,
											-0.25, -0.6, 0.0
										]);

	var i_vertices = new Float32Array 	([
												0.35, 0.7, 0.0,
												0.25, 0.7, 0.0,
												0.25, -0.7, 0.0,
												0.35, -0.7, 0.0
										]);

	var A_vertices =  new Float32Array	([
											//left
											0.75, 0.7, 0.0,
											0.75, 0.5, 0.0,
											0.55, -0.7, 0.0,
											0.45, -0.7, 0.0,
											//right
											0.75, 0.7, 0.0,
											0.75, 0.5, 0.0,
											0.95, -0.7, 0.0,
											1.05, -0.7, 0.0,
											//middle strips
											0.66, -0.05, 0.0,
											0.84, -0.05, 0.0,

											0.65, -0.1, 0.0,
											0.85, -0.1, 0.0,

											0.64, -0.15, 0.0,
											0.86, -0.15, 0.0,
										]);

	//color declarations
	var I_color = new Float32Array ([
										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0
									]);

	var N_color = new Float32Array ([
										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,

										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,

										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
									]);

	var D_color = new Float32Array 	([
										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,

										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,

										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,

										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0
									]);

	var i_color = new Float32Array	([
										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0
									]);

	var A_color = new Float32Array 	([
										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,

										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,
										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0,

										1.0, 0.5, 0.0,
										1.0, 0.5, 0.0,

										1.0, 1.0, 1.0,
										1.0, 1.0, 1.0,

										0.0, 0.5, 0.0,
										0.0, 0.5, 0.0
									]);

	//I
	vao_I = gl.createVertexArray();
	gl.bindVertexArray(vao_I);

	vbo_I_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_I_position);
	gl.bufferData(gl.ARRAY_BUFFER, I_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	vbo_I_color = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_I_color);
	gl.bufferData(gl.ARRAY_BUFFER, I_color, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	//N
	vao_N = gl.createVertexArray();
	gl.bindVertexArray(vao_N);

	vbo_N_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_N_position);
	gl.bufferData(gl.ARRAY_BUFFER, N_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	vbo_N_color = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_N_color);
	gl.bufferData(gl.ARRAY_BUFFER, N_color, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	//D
	vao_D = gl.createVertexArray();
	gl.bindVertexArray(vao_D);

	vbo_D_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_D_position);
	gl.bufferData(gl.ARRAY_BUFFER, D_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	vbo_D_color = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_D_color);
	gl.bufferData(gl.ARRAY_BUFFER, D_color, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);
	//i
	vao_i = gl.createVertexArray();
	gl.bindVertexArray(vao_i);

	vbo_i_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_i_position);
	gl.bufferData(gl.ARRAY_BUFFER, i_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	vbo_i_color = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_i_color);
	gl.bufferData(gl.ARRAY_BUFFER, i_color, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);
	//A
	vao_A = gl.createVertexArray();
	gl.bindVertexArray(vao_A);

	vbo_A_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_A_position);
	gl.bufferData(gl.ARRAY_BUFFER, A_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	vbo_A_color = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_A_color);
	gl.bufferData(gl.ARRAY_BUFFER, A_color, gl.STATIC_DRAW);
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

function wglDraw()
{
	//code
	gl.clear(gl.COLOR_BUFFER_BIT);

	gl.useProgram(shaderProgramObject);

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();

	mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);

	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	oglDraw_I();
	oglDraw_N();
	oglDraw_D();
	oglDraw_i();
	oglDraw_A();

	gl.useProgram(null);

	//animation loop
	requestAnimationFrame(wglDraw, canvas);
}

function oglDraw_I()
{
	//code
	gl.bindVertexArray(vao_I);

	//draw scene
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

	//unbind vao
	gl.bindVertexArray(null);
}

function oglDraw_N()
{
	//code
	gl.bindVertexArray(vao_N);
	
	//draw scene
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);

	gl.lineWidth(20.0);
	gl.drawArrays(gl.TRIANGLE_FAN, 8, 4);

	//unbind vao
	gl.bindVertexArray(null);
}

function oglDraw_D()
{
	//code
	gl.bindVertexArray(vao_D);

	//draw scene
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 8, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 12, 4);

	//unbind vao
	gl.bindVertexArray(null);
}

function oglDraw_i()
{
	//code
	gl.bindVertexArray(vao_i);

	//draw scene
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

	//unbind vao
	gl.bindVertexArray(null);
}

function oglDraw_A()
{
	//code
	gl.bindVertexArray(vao_A);

	//draw scene
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);

	gl.lineWidth(3.0);
	gl.drawArrays(gl.LINES, 8, 2);
	gl.drawArrays(gl.LINES, 10, 2);
	gl.drawArrays(gl.LINES, 12, 2);

	//unbind vao
	gl.bindVertexArray(null);
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

function wglUnintialise()
{
	//code
	//A
	if(vao_A)
	{
		gl.deleteVertexArray(vao_A);
		vao_A = null;
	}

	if(vbo_A_position)
	{
		gl.deleteBuffer(vbo_A_position);
		vbo_A_position = null;
	}

	if(vbo_A_color)
	{
		gl.deleteBuffer(vbo_A_color);
		vbo_A_color = null;
	}

	//i
	if(vao_i)
	{
		gl.deleteVertexArray(vao_i);
		vao_i = null;
	}

	if(vbo_i_position)
	{
		gl.deleteBuffer(vbo_i_position);
		vbo_i_position = null;
	}

	if(vbo_i_color)
	{
		gl.deleteBuffer(vbo_i_color);
		vbo_i_color = null;
	}

	//D
	if(vao_D)
	{
		gl.deleteVertexArray(vao_D);
		vao_D = null;
	}

	if(vbo_D_position)
	{
		gl.deleteBuffer(vbo_D_position);
		vbo_D_position = null;
	}

	if(vbo_D_color)
	{
		gl.deleteBuffer(vbo_D_color);
		vbo_D_color = null;
	}
	//N
	if(vao_N)
	{
		gl.deleteVertexArray(vao_N);
		vao_N = null;
	}

	if(vbo_N_position)
	{
		gl.deleteBuffer(vbo_N_position);
		vbo_N_position = null;
	}

	if(vbo_N_color)
	{
		gl.deleteBuffer(vbo_N_color);
		vbo_N_color = null;
	}
	//I
	if(vao_I)
	{
		gl.deleteVertexArray(vao_I);
		vao_I = null;
	}

	if(vbo_I_position)
	{
		gl.deleteBuffer(vbo_I_position);
		vbo_I_position = null;
	}

	if(vbo_I_color)
	{
		gl.deleteBuffer(vbo_I_color);
		vbo_I_color = null;
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
