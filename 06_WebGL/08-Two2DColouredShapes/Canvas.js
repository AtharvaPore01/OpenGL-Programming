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
var vao_triangle;
var vao_rectangle;
var vbo_triangle_position;
var vbo_triangle_color;
var vbo_rectangle_position;
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
	var triangleColor = new Float32Array 	([
												1.0, 0.0, 0.0,		//red
												0.0, 1.0, 0.0, 		//green
												0.0, 0.0, 1.0		//blue
											])

	var rectangleVertices = new Float32Array	([
													1.0, 1.0, 0.0,		//right top
													-1.0, 1.0, 0.0,		//left top
													-1.0, -1.0, 0.0,	//left bottom
													1.0, -1.0, 0.0		//right bottom
												]);

	//triangle
	vao_triangle = gl.createVertexArray();
	gl.bindVertexArray(vao_triangle);
	
	//position
	vbo_triangle_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_triangle_position);
	gl.bufferData(gl.ARRAY_BUFFER, triangleVertice, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(vao_triangle);
	
	//color
	vbo_triangle_color = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_triangle_color);
	gl.bufferData(gl.ARRAY_BUFFER, triangleColor, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	//rectangle
	vao_rectangle = gl.createVertexArray();
	gl.bindVertexArray(vao_rectangle);

	//position
	vbo_rectangle_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_rectangle_position);
	gl.bufferData(gl.ARRAY_BUFFER, rectangleVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	//color
	gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 0.0, 0.0, 1.0);

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

	/*triangle*/

	//declare and initialise the matrices.
	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();

	//transformation
	mat4.translate(modelViewMatrix, modelViewMatrix, [-3.0, 0.0, -8.0]);

	//do necessary matrix multiplication
	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	//send necessary matrics to shaders in respective uniforms
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	//bind with vao
	gl.bindVertexArray(vao_triangle);
	
	//draw scene
	gl.drawArrays(gl.TRIANGLES, 0, 3);

	//unbind with vao
	gl.bindVertexArray(null);

	/* rectangle */

	//make identity
	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	//transformation
	mat4.translate(modelViewMatrix, modelViewMatrix, [3.0, 0.0, -8.0]);

	//do necessary matrix multiplication
	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	//send necessary matrics to shaders in respective uniforms
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	//bind with vao
	gl.bindVertexArray(vao_rectangle);
	
	//draw scene
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

	//unbind with vao
	gl.bindVertexArray(null);

	//unuse program
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
	if(vao_triangle)
	{
		gl.deleteVertexArray(vao_triangle);
		vao_triangle = null;
	}

	if(vbo_triangle_position)
	{
		gl.deleteBuffer(vbo_triangle_position);
		vbo_triangle_position = null;
	}

	if(vao_rectangle)
	{
		gl.deleteVertexArray(vao_rectangle);
		vao_rectangle = null;
	}

	if(vbo_rectangle_position)
	{
		gl.deleteBuffer(vbo_rectangle_position);
		vbo_rectangle_position = null;
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
