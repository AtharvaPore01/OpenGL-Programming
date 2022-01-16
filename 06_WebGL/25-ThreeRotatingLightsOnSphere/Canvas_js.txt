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

//Red
var LightAmbient_red = [0.0, 0.0, 0.0];
var LightDiffuse_red = [1.0, 0.0, 0.0];
var LightSpecular_red = [1.0, 0.0, 0.0];
var LightPosition_red = [0.0, 0.0, 0.0, 1.0];
//float LightPosition_red[4];
//green
var LightAmbient_green = [0.0, 0.0, 0.0];
var LightDiffuse_green = [0.0, 1.0, 0.0];
var LightSpecular_green = [0.0, 1.0, 0.0];
var LightPosition_green = [0.0, 0.0, 0.0, 1.0];
//float LightPosition_green[4];
//blue
var LightAmbient_blue = [0.0, 0.0, 0.0];
var LightDiffuse_blue = [0.0, 0.0, 1.0];
var LightSpecular_blue = [0.0, 0.0, 1.0];
var LightPosition_blue = [0.0, 0.0, 0.0, 1.0];
//float LightPosition_blue[4];

var MaterialAmbient = [0.0, 0.0, 0.0];
var MaterialDiffuse = [1.0, 1.0, 1.0];
var MaterialSpecular = [1.0, 1.0, 1.0];
var MaterialShininess = 128.0;	

var modelUniform;
var viewUniform;
var projectionUniform;
var laUniform_red, ldUniform_red, lsUniform_red;
var laUniform_blue, ldUniform_blue, lsUniform_blue;
var laUniform_green, ldUniform_green, lsUniform_green;
var kaUniform, kdUniform, ksUniform, materialShininessUniform;
var LKeyPressedUniform;
var lightPositionUniform_red;
var lightPositionUniform_blue;
var lightPositionUniform_green;

var bLKeyPressed = false;

var sphere = null;

var LightAngle_red = 0.0;
var LightAngle_green = 0.0;
var LightAngle_blue = 0.0;

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

	"uniform vec4 u_light_position_red;" +
	"uniform vec4 u_light_position_green;" +
	"uniform vec4 u_light_position_blue;" +

	"out vec3 transformed_normal;" +

	"out vec3 light_direction_red;" +
	"out vec3 light_direction_green;" +
	"out vec3 light_direction_blue;" +

	"out vec3 viewer_vector;" +

	"void main(void)" +
	"{" +
		"if(u_LKeyPressed == 1)" +
		"{" +
			"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;" +
			"transformed_normal = mat3(u_view_matrix * u_model_matrix) * vNormal;" +
			"light_direction_red = vec3(u_light_position_red) - eyeCoordinates.xyz;" +
			"light_direction_green = vec3(u_light_position_green) - eyeCoordinates.xyz;" +
			"light_direction_blue = vec3(u_light_position_blue) - eyeCoordinates.xyz;" +
			
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
	
	"in vec3 light_direction_red;" +
	"in vec3 light_direction_green;" +
	"in vec3 light_direction_blue;" +

	"in vec3 viewer_vector;" +
	"out vec4 FragColor;" +

	"uniform vec3 u_La_red;" +
	"uniform vec3 u_La_green;" +
	"uniform vec3 u_La_blue;" +

	"uniform vec3 u_Ld_red;" +
	"uniform vec3 u_Ld_green;" +
	"uniform vec3 u_Ld_blue;" +
	
	"uniform vec3 u_Ls_red;" +
	"uniform vec3 u_Ls_green;" +
	"uniform vec3 u_Ls_blue;" +

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

			"vec3 normalized_light_direcion_red = normalize(light_direction_red);" +
			"vec3 normalized_light_direcion_green = normalize(light_direction_green);" +
			"vec3 normalized_light_direcion_blue = normalize(light_direction_blue);" +

			"vec3 normalized_viewer_vector = normalize(viewer_vector);" +
			
			"float tn_dot_ld_red = max(dot(normalized_t_norm, normalized_light_direcion_red), 0.0);" +
			"float tn_dot_ld_green = max(dot(normalized_t_norm, normalized_light_direcion_green), 0.0);" +
			"float tn_dot_ld_blue = max(dot(normalized_t_norm, normalized_light_direcion_blue), 0.0);" +

			"vec3 ambient = (u_La_red * u_Ka) + (u_La_green * u_Ka) + (u_La_blue * u_Ka);" +
			"vec3 diffuse = (u_Ld_red * u_Kd * tn_dot_ld_red) + (u_Ld_green * u_Kd * tn_dot_ld_green) + (u_Ld_blue * u_Kd * tn_dot_ld_blue);" +
			
			"vec3 reflection_vector_red = reflect(-normalized_light_direcion_red, normalized_t_norm);" +
			"vec3 reflection_vector_green = reflect(-normalized_light_direcion_green, normalized_t_norm);" +
			"vec3 reflection_vector_blue = reflect(-normalized_light_direcion_blue, normalized_t_norm);" +
			
			"vec3 specular = (u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red, normalized_viewer_vector), 0.0), u_material_shininess)) + (u_Ls_green * u_Ks * pow(max(dot(reflection_vector_green, normalized_viewer_vector), 0.0), u_material_shininess)) + (u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue, normalized_viewer_vector), 0.0), u_material_shininess));" +
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

	laUniform_red = gl.getUniformLocation(shaderProgramObject, "u_La_red");
	laUniform_green = gl.getUniformLocation(shaderProgramObject, "u_La_green");
	laUniform_blue = gl.getUniformLocation(shaderProgramObject, "u_La_blue");

	lsUniform_red = gl.getUniformLocation(shaderProgramObject, "u_Ls_red");
	lsUniform_green = gl.getUniformLocation(shaderProgramObject, "u_Ls_green");
	lsUniform_blue = gl.getUniformLocation(shaderProgramObject, "u_Ls_blue");

	ldUniform_red = gl.getUniformLocation(shaderProgramObject, "u_Ld_red");
	ldUniform_green = gl.getUniformLocation(shaderProgramObject, "u_Ld_green");
	ldUniform_blue = gl.getUniformLocation(shaderProgramObject, "u_Ld_blue");

	
	kaUniform = gl.getUniformLocation(shaderProgramObject, "u_Ka");
	ksUniform = gl.getUniformLocation(shaderProgramObject, "u_Ks");
	kdUniform = gl.getUniformLocation(shaderProgramObject, "u_Kd");
	materialShininessUniform = gl.getUniformLocation(shaderProgramObject, "u_material_shininess");

	lightPositionUniform_red = gl.getUniformLocation(shaderProgramObject, "u_light_position_red");
	lightPositionUniform_green = gl.getUniformLocation(shaderProgramObject, "u_light_position_green");
	lightPositionUniform_blue = gl.getUniformLocation(shaderProgramObject, "u_light_position_blue");

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
	LightAngle_red = LightAngle_red + 0.02;
	if (LightAngle_red >= 360)
	{
		LightAngle_red = 0.0;
	}

	LightAngle_green = LightAngle_green + 0.02;
	if (LightAngle_green >= 360)
	{
		LightAngle_green = 0.0;
	}

	LightAngle_blue = LightAngle_blue + 0.02;
	if (LightAngle_blue >= 360)
	{
		LightAngle_blue = 0.0;
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
		gl.uniform3fv(laUniform_red, LightAmbient_red);
		gl.uniform3fv(laUniform_green, LightAmbient_green);
		gl.uniform3fv(laUniform_blue, LightAmbient_blue);

		gl.uniform3fv(lsUniform_red, LightSpecular_red);
		gl.uniform3fv(lsUniform_green, LightSpecular_green);
		gl.uniform3fv(lsUniform_blue, LightSpecular_blue);

		gl.uniform3fv(ldUniform_red, LightDiffuse_red);
		gl.uniform3fv(ldUniform_green, LightDiffuse_green);
		gl.uniform3fv(ldUniform_blue, LightDiffuse_blue);

		//setting material proprties
		gl.uniform3fv(kaUniform, MaterialAmbient);
		gl.uniform3fv(ksUniform, MaterialSpecular);
		gl.uniform3fv(kdUniform, MaterialDiffuse);
		//var lightPosition = [0.0, 0.0, 2.0, 1.0];

		LightPosition_red = [0.0, 100.0 * Math.cos(LightAngle_red), 100.0 * Math.sin(LightAngle_red), 1.0];
		gl.uniform4fv(lightPositionUniform_red, LightPosition_red);

		LightPosition_green = [100.0 * Math.cos(LightAngle_green), 0.0, 100.0 * Math.sin(LightAngle_green), 1.0];
		gl.uniform4fv(lightPositionUniform_green, LightPosition_green);

		LightPosition_blue = [100.0 * Math.cos(LightAngle_blue), 100.0 * Math.sin(LightAngle_blue), 0.0, 1.0];
		gl.uniform4fv(lightPositionUniform_blue, LightPosition_blue);
	
		gl.uniform1f(materialShininessUniform, MaterialShininess);
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
