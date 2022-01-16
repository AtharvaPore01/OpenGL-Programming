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

var sphere=null;
var sphere_earth=null;
var sphere_moon=null;

//texture variables
var earth_texture = 0;
var sun_texture = 0;
var moon_texture = 0;

//variables
var Year = 0;
var Day = 0;
var MoonRotation = 0;

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
	"in vec2 vTexcoords;" +
	"uniform mat4 u_model_matrix;" +
	"uniform mat4 u_view_matrix;" +
	"uniform mat4 u_projection_matrix;" +
	"out vec2 out_tex_coord;" +
	"void main(void)" +
	"{" +
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
		"out_tex_coord = vTexcoords;" +
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
	"in vec2 out_tex_coord;" +
	"uniform highp sampler2D u_texture_sampler;" +
	"out vec4 FragColor;" +
	"void main(void)" +
	"{" +
		"FragColor = texture(u_texture_sampler, out_tex_coord);" +
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
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.VDG_ATTRIBUTE_TEXTURE0, "vTexcoords");

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
	uniform_texture_0_sampler = gl.getUniformLocation(shaderProgramObject, "u_texture_sampler");

	//load sun texture
	sun_texture = gl.createTexture();
	sun_texture.image = new Image();
	sun_texture.image.src = "sun.png";
	sun_texture.image.onload = function()
	{
		gl.bindTexture(gl.TEXTURE_2D, sun_texture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);	//flip the Y of my image.
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sun_texture.image);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	//load earth texture
	earth_texture = gl.createTexture();
	earth_texture.image = new Image();
	earth_texture.image.src = "earth.png";
	earth_texture.image.onload = function()
	{
		gl.bindTexture(gl.TEXTURE_2D, earth_texture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);	//flip the Y of my image.
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, earth_texture.image);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	//load moon texture
	moon_texture = gl.createTexture();
	moon_texture.image = new Image();
	moon_texture.image.src = "moon.png";
	moon_texture.image.onload = function()
	{
		gl.bindTexture(gl.TEXTURE_2D, moon_texture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);	//flip the Y of my image.
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, moon_texture.image);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	//sphere
	sphere = new Mesh();
	makeSphere(sphere, 0.75, 30, 30);

	sphere_earth = new Mesh();
	makeSphere(sphere_earth, 0.3, 20, 20);

	sphere_moon = new Mesh();
	makeSphere(sphere_moon, 0.15, 30, 30);
	
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

	mat4.lookAt(modelMatrix, [0.0, 0.0, 7.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	
	mat4.rotateY(modelMatrix, modelMatrix, degToRad(90.0));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	//mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -12.0]);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveMatrixProjection);

	//bind with texture
	gl.bindTexture(gl.TEXTURE_2D, sun_texture);
	gl.uniform1i(uniform_texture_0_sampler, 0);

	sphere.draw();

	//earth
	
	mat4.lookAt(modelMatrix, [0.0, 0.0, 7.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.rotateY(modelMatrix, modelMatrix, degToRad(Year));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.translate(modelMatrix, modelMatrix, [1.9, 0.0, 0.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.rotateY(modelMatrix, modelMatrix, degToRad(90.0));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.rotateY(modelMatrix, modelMatrix, degToRad(Day));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveMatrixProjection);

	//bind with texture
	gl.bindTexture(gl.TEXTURE_2D, earth_texture);
	gl.uniform1i(uniform_texture_0_sampler, 0);

	sphere_earth.draw();

	//moon
	mat4.rotateY(modelMatrix, modelMatrix, degToRad(Year));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.translate(modelMatrix, modelMatrix, [0.75, 0.0, 0.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	mat4.rotateY(modelMatrix, modelMatrix, degToRad(MoonRotation));
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveMatrixProjection);

	//bind with texture
	gl.bindTexture(gl.TEXTURE_2D, moon_texture);
	gl.uniform1i(uniform_texture_0_sampler, 0);

	sphere_moon.draw();

	mat4.lookAt(modelMatrix, [0.0, 0.0, 7.0], [0.0, 5.0, 0.0], [0.0, 1.0, 0.0]);
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);

	gl.useProgram(null);

	//update
	Day = (Day + 2) % 360;
	MoonRotation = (MoonRotation + 2) % 360;
	Year = (Year + 1) % 360;

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
