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

var light_ambient=[0.0, 0.0, 0.0];
var light_diffuse=[1.0, 1.0, 1.0];
var light_specular=[1.0, 1.0, 1.0];
var light_position=[100.0, 100.0, 100.0, 1.0];

var material_ambient=[0.0, 0.0, 0.0];
var material_diffuse=[1.0, 1.0, 1.0];
var material_specular=[1.0, 1.0, 1.0];
var material_shininess=128.0;

var modelUniform;
var viewUniform;
var projectionUniform;
var laUniform, ldUniform, lsUniform;
var kaUniform, kdUniform, ksUniform, materialShininessUniform;
var LKeyPressedUniform;
var lightPositionUniform;

var bLKeyPressed = false;

var sphere = null;

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
	"uniform vec4 u_Light_Position;" +
	"out vec3 transformed_normal;" +
	"out vec3 light_direction;" +
	"out vec3 viewer_vector;" +

	"void main(void)" +
	"{" +
		"if(u_LKeyPressed == 1)" +
		"{" +
			"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;" +
			"transformed_normal = mat3(u_view_matrix * u_model_matrix) * vNormal;" +
			"light_direction = vec3(u_Light_Position) - eyeCoordinates.xyz;" +
			"viewer_vector = -eyeCoordinates.xyz;" +
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
	"in vec3 transformed_normal;" +
	"in vec3 light_direction;" +
	"in vec3 viewer_vector;" +
	"out vec4 FragColor;" +
	"uniform vec3 u_La;" +
	"uniform vec3 u_Ld;" +
	"uniform vec3 u_Ls;" +
	"uniform vec3 u_Ka;" +
	"uniform vec3 u_Kd;" +
	"uniform vec3 u_Ks;" +
	"uniform float u_material_shininess;" +
	"uniform int u_LKeyPressed;" +
	"void main(void)" +
	"{" +
		"vec3 phong_ads_lighting;" +
		"if(u_LKeyPressed == 1)" +
		"{" +
			"vec3 normalized_t_norm = normalize(transformed_normal);" +
			"vec3 normalized_light_direcion = normalize(light_direction);" +
			"vec3 normalized_viewer_vector = normalize(viewer_vector);" +
			"float tn_dot_ld = max(dot(normalized_t_norm, normalized_light_direcion), 0.0);" +
			"vec3 ambient = u_La * u_Ka;" +
			"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" +
			"vec3 reflection_vector = reflect(-normalized_light_direcion, normalized_t_norm);" +
			"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0), u_material_shininess);" +
			"phong_ads_lighting = ambient + diffuse + specular;" +
		"}" +
		"else"+
		"{" +
			"phong_ads_lighting = vec3(1.0, 1.0, 1.0);" +
		"}" +
		"FragColor = vec4(phong_ads_lighting, 1.0);" +
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

	laUniform = gl.getUniformLocation(shaderProgramObject, "u_La");
	lsUniform = gl.getUniformLocation(shaderProgramObject, "u_Ls");
	ldUniform = gl.getUniformLocation(shaderProgramObject, "u_Ld");
	
	kaUniform = gl.getUniformLocation(shaderProgramObject, "u_Ka");
	ksUniform = gl.getUniformLocation(shaderProgramObject, "u_Ks");
	kdUniform = gl.getUniformLocation(shaderProgramObject, "u_Kd");
	materialShininessUniform = gl.getUniformLocation(shaderProgramObject, "u_material_shininess");

	lightPositionUniform = gl.getUniformLocation(shaderProgramObject, "u_Light_Position");
	LKeyPressedUniform = gl.getUniformLocation(shaderProgramObject, "u_LKeyPressed");

	//sphere
	sphere = new Mesh();
	makeSphere(sphere, 2.0, 30, 30);

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
		gl.uniform3fv(laUniform, light_ambient);
		gl.uniform3fv(lsUniform, light_specular);
		gl.uniform3fv(ldUniform, light_diffuse);
		//setting material proprties
		gl.uniform3fv(kaUniform, material_ambient);
		gl.uniform3fv(ksUniform, material_specular);
		gl.uniform3fv(kdUniform, material_diffuse);
		//var lightPosition = [0.0, 0.0, 2.0, 1.0];
		gl.uniform4fv(lightPositionUniform, light_position);
	
		gl.uniform1f(materialShininessUniform, material_shininess);
	}
	else
	{
		gl.uniform1i(LKeyPressedUniform, 0);
	}

	var modelMatrix = mat4.create();
	var viewMatrix = mat4.create();
	var projectionMatrix = mat4.create();

	//transformation
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -5.0]);

	//send necessary matrics to shaders in respective uniforms
	gl.uniformMatrix4fv(modelUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionUniform, false, perspectiveMatrixProjection);

	sphere.draw();

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
	if(sphere)
	{
		sphere.deallocate();
		sphere = null;
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
