//global variables
var canvas 	= null;
var gl = null;	
var bFullscreen = false;
var canvas_original_width;
var canvas_original_height;

const WebGLMacros = 
{
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3,
};

var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

var vao;
var vbo;
var mvpUniform;

var orthographicsProjectionMatrix;

//To Start Animation : To Have requestAnimationFrame() to be called "cross-browser" compatible
var requestAnimationFrame = 
window.requestAnimationFrame ||
window.webkitRequestAnimationFrame ||
window.mozRequestAnimationFrame ||
window.oRequestAnimationFrame ||
window.msRequestAnimationFrame ||
null;

//To Stop Animation : To Have cancelAnimationFrame() to be called "cross-browser" compatible
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

	canvas_original_width = canvas.width;
	canvas_original_height = canvas.height;

	//register keyboard's keydown handle
	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
	window.addEventListener("resize", wglResize, false);

	//initialize WebGL
	wglInit();

	//start drawing here as warming-up
	wglResize();
	wglDraw();
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

		bFullscreen = true;
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

		bFullscreen = false;
	}
}

function wglInit()
{
	//code
	//get WebGL 2.0 context
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
	"uniform mat4 u_mvp_matrix;" +
	"void main(void)" +
	"{" +
		"gl_Position = u_mvp_matrix * vPosition;" +
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
	"out vec4 FragColor;" +
	"void main(void)" +
	"{" +
		"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" +
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
												0.0, 50.0, 0.0,			//appex
												-50.0, -50.0, 0.0,		//left-bottom
												50.0, -50.0, 0.0		//right-bottom
											]);

	vao = gl.createVertexArray();
	gl.bindVertexArray(vao);

	vbo = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bufferData(gl.ARRAY_BUFFER, triangleVertice, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	gl.bindVertexArray(null);

	//set clear color
	gl.clearColor(0.0, 0.0, 1.0, 1.0);

	//initialise projection matrix
	orthographicsProjectionMatrix = mat4.create();
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

	//orthographics Projection Matrix
	if(canvas.width <= canvas.height)
	{
		mat4.ortho(	orthographicsProjectionMatrix, 
					-100.0, 
					100.0, 
					(-100.0 * (canvas.height / canvas.width)),
					(100.0 * (canvas.height / canvas.width)),
					-100.0,
					100.0);
	}
	else
	{
		mat4.ortho(	orthographicsProjectionMatrix, 
					(-100.0 * (canvas.width / canvas.height)), 
					(100.0 * (canvas.width / canvas.height)), 
					-100.0,
					100.0,
					-100.0,
					100.0);
	}
}

function wglDraw()
{
	//code
	gl.clear(gl.COLOR_BUFFER_BIT);

	gl.useProgram(shaderProgramObject);

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();

	mat4.multiply(modelViewProjectionMatrix, orthographicsProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	gl.bindVertexArray(vao);
	gl.drawArrays(gl.TRIANGLES, 0, 3);

	gl.bindVertexArray(null);

	gl.useProgram(null);

	//animation loop
	requestAnimationFrame(wglDraw, canvas);
}

function wglUnintialise()
{
	//code
	if(vao)
	{
		gl.deleteVertexArray(vao);
		vao = null;
	}

	if(vbo)
	{
		gl.deleteBuffer(vbo);
		vbo = null;
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
