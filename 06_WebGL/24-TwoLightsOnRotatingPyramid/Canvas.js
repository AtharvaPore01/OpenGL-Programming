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

var vao_pyramid;
var vbo_pyramid_position;
var vbo_pyramid_normal;

var light_ambient_red=[0.0, 0.0, 0.0];
var light_diffuse_red=[1.0, 0.0, 0.0];
var light_specular_red=[1.0, 0.0, 0.0];
var light_position_red=[-2.0, 0.0, 0.0, 1.0];

var light_ambient_blue=[0.0, 0.0, 0.0];
var light_diffuse_blue=[0.0, 0.0, 1.0];
var light_specular_blue=[0.0, 0.0, 1.0];
var light_position_blue=[2.0, 0.0, 0.0, 1.0];

var material_ambient=[0.0, 0.0, 0.0];
var material_diffuse=[1.0, 1.0, 1.0];
var material_specular=[1.0, 1.0, 1.0];
var material_shininess=128.0;

var modelUniform;
var viewUniform;
var projectionUniform;

var laUniform_red, ldUniform_red, lsUniform_red, lightPositionUniform_red;
var laUniform_blue, ldUniform_blue, lsUniform_blue, lightPositionUniform_blue;

var kaUniform, kdUniform, ksUniform, materialShininessUniform;

var LKeyPressedUniform;

var bLKeyPressed = false;

var anglePyramid = 0.0;

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
	"uniform vec3 u_La_red;" +
	"uniform vec3 u_La_blue;" +
	"uniform vec3 u_Ld_red;" +
	"uniform vec3 u_Ld_blue;" +
	"uniform vec3 u_Ls_red;" +
	"uniform vec3 u_Ls_blue;" +
	"uniform vec4 u_light_position_red;" +
	"uniform vec4 u_light_position_blue;" +
	"uniform vec3 u_Ka;" +
	"uniform vec3 u_Kd;" +
	"uniform vec3 u_Ks;" +
	"uniform float u_material_shininess;" +
	"out vec3 phong_ads_lighting;" +
	"void main(void)" +
	"{" +
		"if(u_LKeyPressed == 1)" +
		"{" +
			"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;" +
			"vec3 t_norm = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);" +
			"vec3 s_red = normalize(vec3(u_light_position_red) - eyeCoordinates.xyz);" +
			"vec3 s_blue = normalize(vec3(u_light_position_blue) - eyeCoordinates.xyz);" +
			"float tn_dot_ld_red = max(dot(t_norm, s_red), 0.0);" +
			"float tn_dot_ld_blue = max(dot(t_norm, s_blue), 0.0);" +
			
			"vec3 ambient = (u_La_red * u_Ka) + (u_La_blue * u_Ka);" +
			"vec3 diffuse = (u_Ld_red * u_Kd * tn_dot_ld_red) + (u_Ld_blue * u_Kd * tn_dot_ld_blue);" +
			
			"vec3 reflection_vector_red = reflect(-s_red, t_norm);" +
			"vec3 reflection_vector_blue = reflect(-s_blue, t_norm);" +	
			"vec3 viewer_vector = normalize(-eyeCoordinates.xyz);" +
			
			"vec3 specular = u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red, viewer_vector), 0.0), u_material_shininess) + (u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue, viewer_vector), 0.0), u_material_shininess));" +
			"phong_ads_lighting = ambient + diffuse + specular;" +
		"}" +
		"else"+
		"{" +
			"phong_ads_lighting = vec3(1.0, 1.0, 1.0);" +
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
	"in vec3 phong_ads_lighting;" +
	"out vec4 FragColor;" +
	"void main(void)" +
	"{" +
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
	lsUniform_red = gl.getUniformLocation(shaderProgramObject, "u_Ls_red");
	ldUniform_red = gl.getUniformLocation(shaderProgramObject, "u_Ld_red");

	laUniform_blue = gl.getUniformLocation(shaderProgramObject, "u_La_blue");
	lsUniform_blue = gl.getUniformLocation(shaderProgramObject, "u_Ls_blue");
	ldUniform_blue = gl.getUniformLocation(shaderProgramObject, "u_Ld_blue");
	
	kaUniform = gl.getUniformLocation(shaderProgramObject, "u_Ka");
	ksUniform = gl.getUniformLocation(shaderProgramObject, "u_Ks");
	kdUniform = gl.getUniformLocation(shaderProgramObject, "u_Kd");
	materialShininessUniform = gl.getUniformLocation(shaderProgramObject, "u_material_shininess");

	lightPositionUniform_red = gl.getUniformLocation(shaderProgramObject, "u_Light_Position_red");
	lightPositionUniform_blue = gl.getUniformLocation(shaderProgramObject, "u_light_position_blue");
	LKeyPressedUniform = gl.getUniformLocation(shaderProgramObject, "u_LKeyPressed");

	//pyramid vertices
	var pyramidVertice = new Float32Array([
											0.0, 1.0, 0.0,
											-1.0, -1.0, 1.0,
											1.0, -1.0, 1.0,
											
											0.0, 1.0, 0.0,
											1.0, -1.0, 1.0,
											1.0, -1.0, -1.0,

											0.0, 1.0, 0.0,
											1.0, -1.0, -1.0,
											-1.0, -1.0, -1.0,

											0.0, 1.0, 0.0, 
											-1.0, -1.0, -1.0, 
											-1.0, -1.0, 1.0
										]);
	var pyramidNormal = new Float32Array ([
											0.0, 0.447214, 0.894427,
											0.0, 0.447214, 0.894427,
											0.0, 0.447214, 0.894427,

											0.89427, 0.447214, 0.0,
											0.89427, 0.447214, 0.0,
											0.89427, 0.447214, 0.0,

											0.0, 0.447214, -0.894427,
											0.0, 0.447214, -0.894427,
											0.0, 0.447214, -0.894427,

											-0.89427, 0.447214, 0.0,
											-0.89427, 0.447214, 0.0,
											-0.89427, 0.447214, 0.0	
										]);

	//pyramid
	vao_pyramid = gl.createVertexArray();
	gl.bindVertexArray(vao_pyramid);
	
	//position
	vbo_pyramid_position = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_pyramid_position);
	gl.bufferData(gl.ARRAY_BUFFER, pyramidVertice, gl.STATIC_DRAW);
	gl.vertexAttribPointer(	WebGLMacros.VDG_ATTRIBUTE_VERTEX,
							3,
							gl.FLOAT,
							false,
							0, 
							0);
	gl.enableVertexAttribArray(WebGLMacros.VDG_ATTRIBUTE_VERTEX);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	//color
	vbo_pyramid_normal = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_pyramid_normal);
	gl.bufferData(gl.ARRAY_BUFFER, pyramidNormal, gl.STATIC_DRAW);
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
	anglePyramid = anglePyramid + 1.0;
	if(anglePyramid >= 360.0)
	{
		anglePyramid = 0.0
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
		gl.uniform3fv(laUniform_red, light_ambient_red);
		gl.uniform3fv(lsUniform_red, light_specular_red);
		gl.uniform3fv(ldUniform_red, light_diffuse_red);

		gl.uniform3fv(laUniform_blue, light_ambient_blue);
		gl.uniform3fv(lsUniform_blue, light_specular_blue);
		gl.uniform3fv(ldUniform_blue, light_diffuse_blue);
		//setting material proprties
		gl.uniform3fv(kaUniform, material_ambient);
		gl.uniform3fv(ksUniform, material_specular);
		gl.uniform3fv(kdUniform, material_diffuse);
		//var lightPosition = [0.0, 0.0, 2.0, 1.0];
		gl.uniform4fv(lightPositionUniform_red, light_position_red);
		gl.uniform4fv(lightPositionUniform_blue, light_position_blue);
	
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
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -4.0]);
	mat4.rotateY(modelMatrix, modelMatrix, degToRad(anglePyramid));

	//send necessary matrics to shaders in respective uniforms
	gl.uniformMatrix4fv(modelUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionUniform, false, perspectiveMatrixProjection);

	//bind with vao
	gl.bindVertexArray(vao_pyramid);
	
	//draw scene
	gl.drawArrays(gl.TRIANGLES, 0, 12);

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

	if(vao_pyramid)
	{
		gl.deleteVertexArray(vao_pyramid);
		vao_pyramid = null;
	}

	if(vbo_pyramid_position)
	{
		gl.deleteBuffer(vbo_pyramid_position);
		vbo_pyramid_position = null;
	}

	if(vbo_pyramid_normal)
	{
		gl.deleteBuffer(vbo_pyramid_normal);
		vbo_pyramid_normal = null;
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