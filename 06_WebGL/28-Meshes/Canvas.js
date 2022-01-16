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
var vao_one;
var vao_two;
var vao_three;
var vao_four;
var vao_five;
var vao_six;

var vbo_position_one;
var vbo_position_two;
var vbo_position_three;
var vbo_position_four;
var vbo_position_five;
var vbo_position_six;
var vbo_color_six

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
	"uniform mat4 u_mvp_matrix;" +
	"out vec4 out_color;" +
	"void main(void)" +
	"{" +
		"gl_Position = u_mvp_matrix * vPosition;" +
		"gl_PointSize = 2.0;" +
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
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vCOlor");
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

	var firstDesign_vertices = new Float32Array ([
													//First Row
													-1.7, 0.9, 0.0, 
													-1.5, 0.9, 0.0, 
													-1.3, 0.9, 0.0, 
													-1.1, 0.9, 0.0, 

													//Second Row
													-1.7, 0.7, 0.0, 
													-1.5, 0.7, 0.0, 
													-1.3, 0.7, 0.0, 
													-1.1, 0.7, 0.0, 

													//Third Row
													-1.7, 0.5, 0.0, 
													-1.5, 0.5, 0.0, 
													-1.3, 0.5, 0.0, 
													-1.1, 0.5, 0.0, 

													//Fourth Row
													-1.7, 0.3, 0.0, 
													-1.5, 0.3, 0.0, 
													-1.3, 0.3, 0.0, 
													-1.1, 0.3, 0.0
												]);

	var secondDesign_vertice = new Float32Array ([
													//1st Vertical Line
													-0.6, 0.9, 0.0, 
													-0.6, 0.3, 0.0, 
													//2nd Vertical Line
													-0.4, 0.9, 0.0, 
													-0.4, 0.3, 0.0, 
													//3rd Vertical Line
													-0.2, 0.9, 0.0, 
													-0.2, 0.3, 0.0, 
													
													//1st Horizontal Line
													-0.6, 0.9, 0.0, 
													-0.0, 0.9, 0.0, 

													//2nd Horizontal Line
													-0.6, 0.7, 0.0, 
													-0.0, 0.7, 0.0, 
													
													//3rd Horizontal Line
													-0.6, 0.5, 0.0, 
													-0.0, 0.5, 0.0, 

													//1st Olique Line
													-0.6, 0.7, 0.0, 
													-0.4, 0.9, 0.0, 

													//2nd Olique Line
													-0.6, 0.5, 0.0, 
													-0.2, 0.9, 0.0, 

													//3rd Olique Line
													-0.6, 0.3, 0.0, 
													-0.0, 0.9, 0.0, 

													//4th Olique Line
													-0.4, 0.3, 0.0, 
													-0.0, 0.7, 0.0, 

													-0.2, 0.3, 0.0, 
													-0.0, 0.5, 0.0

												]);

	var thirdDesign_vertices = new Float32Array ([

													//1st Vertical Line
													0.3, 0.9, 0.0,
													0.3, 0.3, 0.0,
													//2nd Vertical Line
													0.5, 0.9, 0.0,
													0.5, 0.3, 0.0,
													//3rd Vertical Line
													0.7, 0.9, 0.0,
													0.7, 0.3, 0.0,
													//4th Vertical Line
													0.9, 0.9, 0.0,
													0.9, 0.3, 0.0,
													//1st Horizontal Line
													0.3, 0.9, 0.0,
													0.9, 0.9, 0.0,
													//2nd Horizontal Line
													0.3, 0.7, 0.0,
													0.9, 0.7, 0.0,
													//3rd Horizontal Line
													0.3, 0.5, 0.0,
													0.9, 0.5, 0.0,
													//4th Horizontal Line
													0.3, 0.3, 0.0,
													0.9, 0.3, 0.0

												]);

	var fourthDesign_vertices = new Float32Array ([

													//4th Row
													-1.7, -0.9, 0.0, 
													-1.1, -0.9, 0.0, 
													//3rd Row
													-1.7, -0.7, 0.0, 
													-1.1, -0.7, 0.0, 
													//2nd Row
													-1.7, -0.5, 0.0, 
													-1.1, -0.5, 0.0, 
													//1st Row
													-1.7, -0.3, 0.0, 
													-1.1, -0.3, 0.0, 

													//4th column
													-1.7, -0.9, 0.0, 
													-1.7, -0.3, 0.0, 
													//3rd Column
													-1.5, -0.9, 0.0, 
													-1.5, -0.3, 0.0, 
													//2nd Column
													-1.3, -0.9, 0.0, 
													-1.3, -0.3, 0.0, 
													//1st Column
													-1.1, -0.9, 0.0, 
													-1.1, -0.3, 0.0, 

													//1st Olique Line
													-1.7, -0.5, 0.0, 
													-1.5, -0.3, 0.0, 
													//2nd Olique Line
													-1.7, -0.7, 0.0, 
													-1.3, -0.3, 0.0, 
													//3rd Olique Line
													-1.7, -0.9, 0.0, 
													-1.1, -0.3, 0.0, 
													//4th Olique Line
													-1.5, -0.9, 0.0, 
													-1.1, -0.5, 0.0, 
													//5th Olique Line
													-1.3, -0.9, 0.0, 
													-1.1, -0.7, 0.0

												]);

	var fifthDesign_vertices = new Float32Array ([

													//4th Row
													-0.6, -0.9, 0.0,
													-0.0, -0.9, 0.0,
													//1st Row
													-0.6, -0.3, 0.0,
													-0.0, -0.3, 0.0,

													//4th column
													-0.6, -0.9, 0.0,
													-0.6, -0.3, 0.0,
													//1st Column
													0.0, -0.9, 0.0,
													0.0, -0.3, 0.0,

													//Ray
													-0.6, -0.3, 0.0,
													0.0, -0.5, 0.0,

													-0.6, -0.3, 0.0,
													0.0, -0.7, 0.0,

													-0.6, -0.3, 0.0,
													0.0, -0.9, 0.0,

													-0.6, -0.3, 0.0,
													-0.4, -0.9, 0.0,

													-0.6, -0.3, 0.0,
													-0.2, -0.9, 0.0
												]);

	var sixthDesign_vertices = new Float32Array ([

													//first quad
													0.5, -0.3, 0.0, 
													0.3, -0.3, 0.0, 
													0.3, -0.9, 0.0, 
													0.5, -0.9, 0.0, 

													//second quad
													0.7, -0.3, 0.0, 
													0.5, -0.3, 0.0, 
													0.5, -0.9, 0.0, 
													0.7, -0.9, 0.0, 

													//third quad
													0.9, -0.3, 0.0,
													0.7, -0.3, 0.0,
													0.7, -0.9, 0.0,
													0.9, -0.9, 0.0,

													//vertical line 1
													0.5, -0.3, 0.0,
													0.5, -0.9, 0.0,

													//vertical line 2
													0.7, -0.3, 0.0,
													0.7, -0.9, 0.0,

													//Horizontal Line 1
													0.3, -0.5, 0.0,
													0.9, -0.5, 0.0,

													//Horizontal Line 1
													0.3, -0.7, 0.0,
													0.9, -0.7, 0.0

												]);

	var sixthDesign_color = new Float32Array ([

													//first quad
													1.0, 0.0, 0.0,
													1.0, 0.0, 0.0,
													1.0, 0.0, 0.0,
													1.0, 0.0, 0.0,

													//second quad
													0.0, 1.0, 0.0,
													0.0, 1.0, 0.0,
													0.0, 1.0, 0.0,
													0.0, 1.0, 0.0,

													//third quad
													0.0, 0.0, 1.0,
													0.0, 0.0, 1.0,
													0.0, 0.0, 1.0,
													0.0, 0.0, 1.0,

													//vertical line 1
													1.0, 1.0, 1.0,
													1.0, 1.0, 1.0,

													//vertical line 2
													1.0, 1.0, 1.0,
													1.0, 1.0, 1.0,

													//Horizontal Line 1
													1.0, 1.0, 1.0,
													1.0, 1.0, 1.0,

													//Horizontal Line 1
													1.0, 1.0, 1.0,
													1.0, 1.0, 1.0

											]);

	vao_one = gl.createVertexArray();
	gl.bindVertexArray(vao_one);

	vbo_position_one = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position_one);
	gl.bufferData(gl.ARRAY_BUFFER, firstDesign_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 1.0, 1.0, 1.0)

	gl.bindVertexArray(null);

	vao_two = gl.createVertexArray();
	gl.bindVertexArray(vao_two);

	vbo_position_two = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position_two);
	gl.bufferData(gl.ARRAY_BUFFER, secondDesign_vertice, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 1.0, 1.0, 1.0)

	gl.bindVertexArray(null);

	vao_three = gl.createVertexArray();
	gl.bindVertexArray(vao_three);

	vbo_position_three = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position_three);
	gl.bufferData(gl.ARRAY_BUFFER, thirdDesign_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 1.0, 1.0, 1.0)

	gl.bindVertexArray(null);

	vao_four = gl.createVertexArray();
	gl.bindVertexArray(vao_four);

	vbo_position_four = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position_four);
	gl.bufferData(gl.ARRAY_BUFFER, fourthDesign_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 1.0, 1.0, 1.0)

	gl.bindVertexArray(null);

	vao_five = gl.createVertexArray();
	gl.bindVertexArray(vao_five);

	vbo_position_five = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position_five);
	gl.bufferData(gl.ARRAY_BUFFER, fifthDesign_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 1.0, 1.0, 1.0)

	gl.bindVertexArray(null);

	vao_six = gl.createVertexArray();
	gl.bindVertexArray(vao_six);

	vbo_position_six = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position_six);
	gl.bufferData(gl.ARRAY_BUFFER, sixthDesign_vertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	vbo_color_six = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_color_six);
	gl.bufferData(gl.ARRAY_BUFFER, sixthDesign_color, gl.STATIC_DRAW);
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

	console.log("initialise complete");

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

	//mat4.identity(modelViewMatrix);
	//mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);

	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	DottedSquare();
	SquareAndObliqueLine();

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [0.2, 0.0, -3.0]);

	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	Design_two();
	SquareAndRay();

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [0.6, 0.0, -3.0]);

	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	Square();
	RGB_Quads();

	gl.useProgram(null);

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

function wglUnintialise()
{
	//code
	if(vao_one)
	{
		gl.deleteVertexArray(vao_one);
		vao_one = null;
	}

	if(vao_two)
	{
		gl.deleteVertexArray(vao_two);
		vao_two = null;
	}
	if(vao_three)
	{
		gl.deleteVertexArray(vao_three);
		vao_three = null;
	}
	if(vao_four)
	{
		gl.deleteVertexArray(vao_four);
		vao_four = null;
	}
	if(vao_five)
	{
		gl.deleteVertexArray(vao_five);
		vao_five = null;
	}
	if(vao_six)
	{
		gl.deleteVertexArray(vao_six);
		vao_six = null;
	}

	if(vbo_position_one)
	{
		gl.deleteBuffer(vbo_position_one);
		vbo_position_one = null;
	}

	if(vbo_position_two)
	{
		gl.deleteBuffer(vbo_position_two);
		vbo_position_two = null;
	}

	if(vbo_position_three)
	{
		gl.deleteBuffer(vbo_position_three);
		vbo_position_three = null;
	}

	if(vbo_position_four)
	{
		gl.deleteBuffer(vbo_position_four);
		vbo_position_four = null;
	}

	if(vbo_position_five)
	{
		gl.deleteBuffer(vbo_position_five);
		vbo_position_five = null;
	}

	if(vbo_position_six)
	{
		gl.deleteBuffer(vbo_position_six);
		vbo_position_six = null;
	}

	if(vbo_color_six)
	{
		gl.deleteBuffer(vbo_color_six);
		vbo_color_six = null;
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

function DottedSquare()
{
	
	gl.bindVertexArray(vao_one);

	//First Row
	gl.drawArrays(gl.POINTS, 0, 1);
	gl.drawArrays(gl.POINTS, 1, 1);
	gl.drawArrays(gl.POINTS, 2, 1);
	gl.drawArrays(gl.POINTS, 3, 1);

	//Second Row
	gl.drawArrays(gl.POINTS, 4, 1);
	gl.drawArrays(gl.POINTS, 5, 1);
	gl.drawArrays(gl.POINTS, 6, 1);
	gl.drawArrays(gl.POINTS, 7, 1);

	//Third Row
	gl.drawArrays(gl.POINTS, 8, 1);
	gl.drawArrays(gl.POINTS, 9, 1);
	gl.drawArrays(gl.POINTS, 10, 1);
	gl.drawArrays(gl.POINTS, 11, 1);

	//Fourth Row
	gl.drawArrays(gl.POINTS, 12, 1);
	gl.drawArrays(gl.POINTS, 13, 1);
	gl.drawArrays(gl.POINTS, 14, 1);
	gl.drawArrays(gl.POINTS, 15, 1);

	gl.bindVertexArray(null);
}

function Design_two()
{

	gl.bindVertexArray(vao_two);
	
	gl.drawArrays(gl.LINES, 0, 2);

	gl.drawArrays(gl.LINES, 2, 2);
	
	gl.drawArrays(gl.LINES, 4, 2);
	
	gl.drawArrays(gl.LINES, 6, 2);
	
	
	gl.drawArrays(gl.LINES, 8, 2);
	
	gl.drawArrays(gl.LINES, 10, 2);
	
	gl.drawArrays(gl.LINES, 12, 2);
	
	gl.drawArrays(gl.LINES, 14, 2);
	
	gl.drawArrays(gl.LINES, 16, 2);
	
	gl.drawArrays(gl.LINES, 18, 2);

	gl.drawArrays(gl.LINES, 20, 2);
	
	//gl.drawArrays(gl.LINES, 22, 2);
	
	gl.bindVertexArray(null);

}

function Square()
{


	gl.bindVertexArray(vao_three);

	//1st Vertical Line
	gl.drawArrays(gl.LINES, 0, 2);
	//2nd Vertical Line
	gl.drawArrays(gl.LINES, 2, 2);
	//3rd Vertical Line
	gl.drawArrays(gl.LINES, 4, 2);
	//4th Vertical Line
	gl.drawArrays(gl.LINES, 6, 2);

	//1st Horizontal Line
	gl.drawArrays(gl.LINES, 8, 2);
	//2nd Horizontal Line
	gl.drawArrays(gl.LINES, 10, 2);
	//3rd Horizontal Line
	gl.drawArrays(gl.LINES, 12, 2);
	//4th Horizontal Line
	gl.drawArrays(gl.LINES, 14, 2);

	gl.bindVertexArray(null);
}

function SquareAndObliqueLine()
{

	gl.bindVertexArray(vao_four);

	gl.drawArrays(gl.LINES, 0, 2);//4th Row
	gl.drawArrays(gl.LINES, 2, 2);//3rd Row
	gl.drawArrays(gl.LINES, 4, 2);//2nd Row
	gl.drawArrays(gl.LINES, 6, 2);//1st Row
	
	gl.drawArrays(gl.LINES, 8, 2);//4th column
	gl.drawArrays(gl.LINES, 10, 2);//3rd column
	gl.drawArrays(gl.LINES, 12, 2);//2nd column
	gl.drawArrays(gl.LINES, 14, 2);//1st column

	gl.drawArrays(gl.LINES, 16, 2);//1st OliqueLine
	gl.drawArrays(gl.LINES, 18, 2);//2nd OliqueLine
	gl.drawArrays(gl.LINES, 20, 2);//3rd OliqueLine
	gl.drawArrays(gl.LINES, 22, 2);//4th OliqueLine
	gl.drawArrays(gl.LINES, 24, 2);//5th OliqueLine

	gl.bindVertexArray(null);
}

function SquareAndRay()
{
	gl.bindVertexArray(vao_five);

	gl.drawArrays(gl.LINES, 0, 2);//4th Row
	gl.drawArrays(gl.LINES, 2, 2);//1st Row
	gl.drawArrays(gl.LINES, 4, 2);//4th column
	gl.drawArrays(gl.LINES, 6, 2);//1st Column
	
	//ray
	gl.drawArrays(gl.LINES, 8, 2);
	gl.drawArrays(gl.LINES, 10, 2);
	gl.drawArrays(gl.LINES, 12, 2);
	gl.drawArrays(gl.LINES, 14, 2);
	gl.drawArrays(gl.LINES, 16, 2);

	gl.bindVertexArray(null);
}

function RGB_Quads()
{
	gl.lineWidth(3.0);
	gl.bindVertexArray(vao_six);

	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);
	gl.drawArrays(gl.TRIANGLE_FAN, 8, 4);

	gl.drawArrays(gl.LINES, 12, 2);//vertical line 1
	gl.drawArrays(gl.LINES, 14, 2);//vertical line 2
	gl.drawArrays(gl.LINES, 16, 2);//Horizontal Line 1
	gl.drawArrays(gl.LINES, 18, 2);//Horizontal Line 2

	gl.bindVertexArray(null);
}
