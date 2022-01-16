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

//vao and vbo declaration
var vao_cube;
var vbo_cube_position;
var vbo_cube_normal;

var modelUniform;
var viewUniform;
var projectionUniform;
var ldUniform;
var kdUniform;
var LKeyPressedUniform;
var lightPositionUniform;

var bLKeyPressed = false;

//rotation related variables
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
		case 76:
			if(bLKeyPressed == false)
				bLKeyPressed = true;
			else
				bLKeyPressed = false;
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
	"in vec3 vNormal;" +
	"uniform mat4 u_model_matrix;" +
	"uniform mat4 u_view_matrix;" +
	"uniform mat4 u_projection_matrix;" +
	"uniform mediump int u_LKeyPressed;" +
	"uniform vec3 u_Ld;" +
	"uniform vec3 u_Kd;" +
	"uniform vec4 u_Light_Position;" +
	"out vec3 diffuse_light;" +
	"void main(void)" +
	"{" +
		"if(u_LKeyPressed == 1)" +
		"{" +
			"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;" +
			"vec3 t_norm = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);" +
			"vec3 s = normalize(vec3(u_Light_Position - eyeCoordinates));" +
			"diffuse_light = u_Ld * u_Kd * max(dot(s, t_norm), 0.0);" +
		"}" +
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
	"in vec3 diffuse_light;" +
	"out vec4 FragColor;" +
	"uniform int u_LKeyPressed;" +
	"void main(void)" +
	"{" +
		"vec4 color;" +
		"if(u_LKeyPressed == 1)" +
		"{" +
			"color = vec4 (diffuse_light, 1.0);"+
		"}" +
		"else" +
		"{"+
			"color = vec4(1.0, 1.0, 1.0, 1.0);" +
		"}"+
		"FragColor = color;" +
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
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.VDG_ATTRIBUTE_NORMAL, "vNormal");

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
	modelUniform = gl.getUniformLocation(shaderProgramObject, "u_model_matrix");
	viewUniform = gl.getUniformLocation(shaderProgramObject, "u_view_matrix");
	projectionUniform = gl.getUniformLocation(shaderProgramObject, "u_projection_matrix");

	ldUniform = gl.getUniformLocation(shaderProgramObject, "u_Ld");
	kdUniform = gl.getUniformLocation(shaderProgramObject, "u_Kd");
	lightPositionUniform = gl.getUniformLocation(shaderProgramObject, "u_Light_Position");
	LKeyPressedUniform = gl.getUniformLocation(shaderProgramObject, "u_LKeyPressed");

	var cubeVertices = new Float32Array	([
											//top surface
											1.0, 1.0, -1.0,		
											-1.0, 1.0, -1.0,		
											-1.0, 1.0, 1.0,	
											1.0, 1.0, 1.0,		
											
											//bottom surface
											1.0, -1.0, 1.0,
											-1.0, -1.0, 1.0,
											-1.0, -1.0, -1.0,
											1.0, -1.0, -1.0,
											
											//front surface
											1.0, 1.0, 1.0,
											-1.0, 1.0, 1.0, 
											-1.0, -1.0, 1.0, 
											1.0, -1.0, 1.0,

											//back surface
											1.0, -1.0, -1.0, 
											-1.0, -1.0, -1.0,
											-1.0, 1.0, -1.0,
											1.0, 1.0, -1.0,
										
											//left surface
											-1.0, 1.0, 1.0,
											-1.0, 1.0, -1.0,
											-1.0, -1.0, -1.0,
											-1.0, -1.0, 1.0,
											//right surface
											1.0, 1.0, -1.0, 
											1.0, 1.0, 1.0, 
											1.0, -1.0, 1.0,
											1.0, -1.0, -1.0
										]);
	var cubeNormal = new Float32Array ([
											//top
											0.0, 1.0, 0.0,
											0.0, 1.0, 0.0,
											0.0, 1.0, 0.0,
											0.0, 1.0, 0.0,

											//bottom
											0.0, -1.0, 0.0,
											0.0, -1.0, 0.0,
											0.0, -1.0, 0.0,
											0.0, -1.0, 0.0,

											//front
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,
											0.0, 0.0, 1.0,

											//back
											0.0, 0.0, -1.0,
											0.0, 0.0, -1.0,
											0.0, 0.0, -1.0,
											0.0, 0.0, -1.0,

											//left
											-1.0, 0.0, 0.0,
											-1.0, 0.0, 0.0,
											-1.0, 0.0, 0.0,
											-1.0, 0.0, 0.0,

											//right
											1.0, 0.0, 0.0,
											1.0, 0.0, 0.0,
											1.0, 0.0, 0.0,
											1.0, 0.0, 0.0
										]);											
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
	gl.vertexAttribPointer(	WebGLMacros.VDG_ATTRIBUTE_VERTEX,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.VDG_ATTRIBUTE_VERTEX);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	//color
	vbo_cube_normal = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_cube_normal);
	gl.bufferData(gl.ARRAY_BUFFER, cubeNormal, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.VDG_ATTRIBUTE_NORMAL,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.VDG_ATTRIBUTE_NORMAL);
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
	
	/* cube */

	if(bLKeyPressed == true)
	{
		gl.uniform1i(LKeyPressedUniform, 1);

		//setting light properties
		gl.uniform3f(ldUniform, 1.0, 1.0, 1.0);
		//setting material proprties
		gl.uniform3f(kdUniform, 0.5, 0.5, 0.5);
		var lightPosition = [0.0, 0.0, 2.0, 1.0];
		gl.uniform4fv(lightPositionUniform, lightPosition);
	}
	else
	{
		gl.uniform1i(LKeyPressedUniform, 0);
	}

	var modelMatrix = mat4.create();
	var viewMatrix = mat4.create();
	var projectionMatrix = mat4.create();

	//transformation
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -3.0]);
	mat4.rotateX(modelMatrix, modelMatrix, degToRad(angleCube));
	mat4.rotateY(modelMatrix, modelMatrix, degToRad(angleCube));
	mat4.rotateZ(modelMatrix, modelMatrix, degToRad(angleCube));

	//send necessary matrics to shaders in respective uniforms
	gl.uniformMatrix4fv(modelUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionUniform, false, perspectiveMatrixProjection);

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

	if(vbo_cube_normal)
	{
		gl.deleteBuffer(vbo_cube_normal);
		vbo_cube_normal = null;
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
