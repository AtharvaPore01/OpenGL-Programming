//global variables
var canvas 	= null;
var gl = null;	//webgl context
var bFullscreen = false;
var canvas_original_width;
var canvas_original_height;

// in webGL this called as key-value coding. 
const WebGLMacros = //when whole WebGLMacros Are const then whole inside it are automatically const
{
	VDG_ATTRIBUTE_VERTEX:0,
	VDG_ATTRIBUTE_COLOR:1,	
	VDG_ATTRIBUTE_NORMAL:2,
	VDG_ATTRIBUTE_TEXTURE0:3
}

//shader and program objectes
var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

var sphere_shoulder=null;
var sphere_elbow=null;
var sphere_wrist=null;

//variables
var Shoulder = 0;
var Elbow = 0;
var Palm = 0;

var uniform_texture_0_sampler;
var modelMatrixUniform, viewMatrixUniform, projectionMatrixUniform;

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
		case 69: //E
			Elbow = (Elbow + 3) % 360;
			break;
		case 101: //e
			Elbow = (Elbow - 3) % 360;
			break;

		case 115: //s 
			Shoulder = (Shoulder - 3) % 360;
			break;
		case 83: 	//S
			Shoulder = (Shoulder + 3) % 360;
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
	"uniform mat4 u_model_matrix;" +
	"uniform mat4 u_view_matrix;" +
	"uniform mat4 u_projection_matrix;" +
	"out vec4 out_color;" +
	"void main(void)" +
	"{" +
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
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
		"FragColor = vec4(0.5, 0.35, 0.05, 1.0);" +
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
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.VDG_ATTRIBUTE_VERTEX, "vPosition");
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.VDG_ATTRIBUTE_COLOR, "vColor");

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
	//mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");
	modelMatrixUniform = gl.getUniformLocation(shaderProgramObject, "u_model_matrix");
	viewMatrixUniform = gl.getUniformLocation(shaderProgramObject, "u_view_matrix");
	projectionMatrixUniform = gl.getUniformLocation(shaderProgramObject, "u_projection_matrix");

	//sphere
	sphere_shoulder = new Mesh();
	makeSphere(sphere_shoulder, 0.5, 10, 10);

	sphere_elbow = new Mesh();
	makeSphere(sphere_elbow, 0.5, 10, 10);

	sphere_wrist = new Mesh();
	makeSphere(sphere_wrist, 0.5, 10, 10);
	
	//depth test
	gl.enable(gl.DEPTH_TEST);

	//depth test to do
	gl.depthFunc(gl.LEQUAL);

	//cull face
	gl.enable(gl.CULL_FACE);

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
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	gl.useProgram(shaderProgramObject);

	var modelMatrix = mat4.create();
	var viewMatrix = mat4.create();
	
	//Shoulder
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -12.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.rotateZ(modelMatrix, modelMatrix, degToRad(Shoulder));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.translate(modelMatrix, modelMatrix, [1.0, 0.0, 0.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.scale(modelMatrix, modelMatrix, [2.0, 0.5, 1.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveMatrixProjection);

	sphere_shoulder.draw();

	//Elbow
	mat4.identity(modelMatrix);

	//do necessary transformations here
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -12.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.rotateZ(modelMatrix, modelMatrix, degToRad(Shoulder));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.translate(modelMatrix, modelMatrix, [2.0, 0.0, 0.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.rotateZ(modelMatrix, modelMatrix, degToRad(Elbow));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.translate(modelMatrix, modelMatrix, [1.0, 0.0, 0.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.scale(modelMatrix, modelMatrix, [2.0, 0.5, 1.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveMatrixProjection);

	sphere_elbow.draw();

	gl.useProgram(null);

	//animation loop
	requestAnimationFrame(wglDraw, canvas);

}

function degToRad(degrees)
{
	var radians = 0.0;
	//code
	radians = degrees * Math.PI / 180;
	return(radians)
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
	if(sphere)
	{
		sphere.deallocate();
		sphere=null;
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
