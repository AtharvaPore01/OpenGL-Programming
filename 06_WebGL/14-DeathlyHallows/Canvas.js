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
var vao_circle;
var vao_line;

var vbo_triangle;
var vbo_circle;
var vbo_line;

var mvpUniform;

//declaration of perspective matrix.
var perspectiveMatrixProjection;

//deathly hallows variables
var x_triangle = 5.0;
var y_triangle = -5.0;
var x_circle = -5.0;
var y_circle = -5.0;
var y_line = 5.0;
var rotationAngle = 0.0;

var a = 0.0, b = 0.0, c = 0.0;
var Perimeter = 0.0;
var x1 = 0.0;
var x2 = -1.0;
var x3 = 1.0;
var y1 = 1.0;
var y2 = -1.0;
var y3 = -1.0;
//for area of triangle
var AreaOfTriangle = 0.0;
//for circle
var x_center = 0.0;
var y_center = 0.0;
var radius = 0.0;

var bCircle = false;
var bLine = false;

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
												0.0, 1.0, 0.0,
												-1.0, -1.0, 0.0,
												-1.0, -1.0, 0.0,
												1.0, -1.0, 0.0,
												1.0, -1.0, 0.0,
												0.0, 1.0, 0.0
											]);
	var lineVertice = new Float32Array	([
											0.0, 1.0, 0.0,
											0.0, -1.0, 0.0
										]);

	//triangle
	vao_triangle = gl.createVertexArray();
	gl.bindVertexArray(vao_triangle);

	vbo_triangle = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_triangle);
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

	//circle
	vao_circle = gl.createVertexArray();
	gl.bindVertexArray(vao_circle);

	vbo_circle = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_circle);
	gl.bufferData(gl.ARRAY_BUFFER, 1 * 3 * 4, gl.DYNAMIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	gl.bindVertexArray(null);

	//line
	vao_line = gl.createVertexArray();
	gl.bindVertexArray(vao_line);

	vbo_line = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_line);
	gl.bufferData(gl.ARRAY_BUFFER, lineVertice, gl.STATIC_DRAW);
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

function wglUpdate()
{
	rotationAngle = rotationAngle + 1.0;
	if (rotationAngle >= 360.0)
	{
		rotationAngle = 0.0;
	}

	
	if(bCircle == true)
	{
		
	}

	if(bLine == true)
	{
		
	}
}

function wglDraw()
{
	//code
	gl.clear(gl.COLOR_BUFFER_BIT);

	gl.useProgram(shaderProgramObject);

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();

	//triangle
	mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -6.0]);
	mat4.translate(modelViewMatrix, modelViewMatrix, [x_triangle, y_triangle, 0.0]);
	mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(rotationAngle))

	mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	deathlyHalloesTriangle();

	if (x_triangle >= 0.0 && y_triangle <= 0.0)
	{
		y_triangle = y_triangle + 0.005;
		x_triangle = x_triangle - 0.005;
		if (y_triangle > 0.0)
		{
			bCircle = true;
		}
	}


	if(bCircle == true)
	{
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);

		//mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -6.0]);
		mat4.translate(modelViewMatrix, modelViewMatrix, [x_circle, y_circle, -6.0]);
		mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(rotationAngle))

		mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);
		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
		//circle
		deathlyHallowCircle();

		if ((x_circle <= 0.0 && y_circle <= 0.0))
		{
			y_circle = y_circle + 0.005;
			x_circle = x_circle + 0.005;
			if (x_circle > 0.0)
			{
				bLine = true;
			}
		}
	}

	if(bLine == true)
	{
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);

		//mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -6.0]);
		mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, y_line, -6.0]);
		mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(rotationAngle))

		mat4.multiply(modelViewProjectionMatrix, perspectiveMatrixProjection, modelViewMatrix);
		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
		//line
		deathlyHallowLine();

		if ((y_line >= 0.0))
		{
			y_line = y_line - 0.005;
		}
	}
	
	
	gl.useProgram(null);

	wglUpdate();

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

function deathlyHalloesTriangle()
{
	//code
	gl.bindVertexArray(vao_triangle);

	gl.drawArrays(gl.LINES, 0, 2);
	gl.drawArrays(gl.LINES, 2, 2);
	gl.drawArrays(gl.LINES, 4, 2);
	
	gl.bindVertexArray(null);
}

function deathlyHallowLine()
{
	//bind with vao
	gl.bindVertexArray(vao_line);

	gl.drawArrays(gl.LINES, 0, 2);

	gl.bindVertexArray(null);
}

function deathlyHallowCircle()
{
	calculateSemiPerimeter();
	calculateAreaOfTriangle();
	calculateRadius();
	calculateCenterOfTheCircle();

	var circleVertices = new Float32Array(3 * 1000);

	for(var i = 0; i < 1000; i++)
	{
		var angle = (2.0 * Math.PI * i) / 1000;
		circleVertices[3 * i + 0] = (Math.cos(angle) * radius) + x_center;
		circleVertices[3 * i + 1] = (Math.sin(angle) * radius) + y_center;
		circleVertices[3 * i + 2] = 0.0;
	}

	gl.bindVertexArray(vao_circle);

	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_circle);
	gl.bufferData(gl.ARRAY_BUFFER, circleVertices, gl.DYNAMIC_DRAW);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.drawArrays(gl.LINES, 0, 1000);

	gl.bindVertexArray(null);

}

function calculateSemiPerimeter()
{
	//code
	a = Math.sqrt((Math.pow((x2 - x1), 2) + Math.pow((y2 - y1), 2)));
	b = Math.sqrt((Math.pow((x3 - x2), 2) + Math.pow((y3 - y2), 2)));
	c = Math.sqrt((Math.pow((x1 - x3), 2) + Math.pow((y1 - y3), 2)));

	//console.log("a: "+a+" b: "+b+ " c: \n"+c);

	//Semi Perimeter
	Perimeter = (a + b + c) / 2;
	//console.log("Perimeter : \n"+Perimeter);
}

function calculateAreaOfTriangle()
{
	//code
	AreaOfTriangle = Math.sqrt(Perimeter * (Perimeter - a) * (Perimeter - b) * (Perimeter - c));
	//console.log("AreaOfTriangle : \n"+AreaOfTriangle);
}

function calculateRadius()
{
	//code
	radius = AreaOfTriangle / Perimeter;
	//console.log("radius : \n"+radius);
}

function calculateCenterOfTheCircle()
{
	//code
	x_center = ((a * x3) + (b * x1) + (c * x2)) / (a + b + c);
	y_center = ((a * (y3)) + (b * (y1)) + (c * (y2))) / (a + b + c);
	//console.log("x_center = \n"+x_center+"y_center = \n"+y_center);
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
