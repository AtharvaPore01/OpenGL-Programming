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
var gVertexShaderObject_perVertex;
var gVertexShaderObject_perFragment;
var gFragmentShaderObject_perVertex;
var gFragmentShaderObject_perFragment;
var gShaderProgramObject_perVertex;
var gShaderProgramObject_perFragment;

//the curly braces means the object and it has key value pairs
/*
	struct
	{
		GLuint model_uniform;
		GLuint view_uniform;
		GLuint projection_uniform;

		GLuint La_uniform;
		GLuint Ld_uniform;
		GLuint Ls_uniform;
		GLuint lightPosition_uniform;

		GLuint Ka_uniform;
		GLuint Kd_uniform;
		GLuint Ks_uniform;
		GLuint shininess_uniform;

		GLuint LKeyPressed_Uniform;
	}vertex, fragment;

	this is written in java script as we mentioned below
*/
var vertex = 
{
	model_uniform : 0,
	view_uniform : 0,
	projection_uniform : 0,
	La_uniform : 0,
	Ld_uniform : 0,
	Ls_uniform : 0,
	lightPosition_uniform : 0,
	Ka_uniform : 0,
	Kd_uniform : 0,
	Ks_uniform : 0,
	shininess_uniform : 0,
	LKeyPressed_Uniform : 0
}

var fragment = 
{
	model_uniform : 0,
	view_uniform : 0,
	projection_uniform : 0,
	La_uniform : 0,
	Ld_uniform : 0,
	Ls_uniform : 0,
	lightPosition_uniform : 0,
	Ka_uniform : 0,
	Kd_uniform : 0,
	Ks_uniform : 0,
	shininess_uniform : 0,
	LKeyPressed_Uniform : 0
}
	


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
var bPerVertex = true;
var bPerFragment = false;

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
		case 70: 							// F or f
			if (bPerFragment == false)
			{
			
				bPerFragment = true;
				if (bPerVertex == true)
				{
				
					bPerVertex = false;
				}
			}
			else
			{
			
				bPerFragment = false;
			}
			break;

		case 81: 							// Q or q
			wglUnintialise();
			window.close();
			break;

		case 86: 							// V or v
			if (bPerVertex == false)
			{
				bPerVertex = true;
				if (bPerFragment == true)
				{

					bPerFragment = false;
				}
			}
			else
			{
				bPerVertex = false;
			}
			break;

		case 76: 							// L or l
			if(bLKeyPressed == false)
				bLKeyPressed = true;
			else
				bLKeyPressed = false;
			break;

		case 27: 							//escape
			toggleFullScreen();
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

	/**** Per Vertex ****/

	//vertex shader
	gVertexShaderObject_perVertex = gl.createShader(gl.VERTEX_SHADER);

	var vertexShaderSourceCode_perVertex = 
	"#version 300 es" +
	"\n" +
	"in vec4 vPosition;" +
	"in vec3 vNormal;" +
	"uniform mat4 u_model_matrix;" +
	"uniform mat4 u_view_matrix;" +
	"uniform mat4 u_projection_matrix;" +
	"uniform mediump int u_LKeyPressed;" +
	"uniform vec3 u_La;" +
	"uniform vec3 u_Ld;" +
	"uniform vec3 u_Ls;" +
	"uniform vec4 u_Light_Position;" +
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
			"vec3 s = normalize(vec3(u_Light_Position) - eyeCoordinates.xyz);" +
			"float tn_dot_ld = max(dot(t_norm, s), 0.0);" +
			"vec3 ambient = u_La * u_Ka;" +
			"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" +
			"vec3 reflection_vector = reflect(-s, t_norm);" +
			"vec3 viewer_vector = normalize(-eyeCoordinates.xyz);" +
			"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, viewer_vector), 0.0), u_material_shininess);" +
			"phong_ads_lighting = ambient + diffuse + specular;" +
		"}" +
		"else"+
		"{" +
			"phong_ads_lighting = vec3(1.0, 1.0, 1.0);" +
		"}" +
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +

	"}"; 

	gl.shaderSource(gVertexShaderObject_perVertex, vertexShaderSourceCode_perVertex);
	gl.compileShader(gVertexShaderObject_perVertex);
	if(gl.getShaderParameter(gVertexShaderObject_perVertex, gl.COMPILE_STATUS) == false)
	{
		var error = gl.getShaderInfoLog(gVertexShaderObject_perVertex);
		if(error.length > 0)
		{
			console.log("vertex shader error.\n");
			alert(error);
			wglUnintialise();
		}
	}

	//fragment shader
	gFragmentShaderObject_perVertex = gl.createShader(gl.FRAGMENT_SHADER);

	var fragmentShaderSourceCode_perVertex = 
	"#version 300 es" +
	"\n" +
	"precision highp float;" +
	"in vec3 phong_ads_lighting;" +
	"out vec4 FragColor;" +
	"void main(void)" +
	"{" +
		"FragColor = vec4(phong_ads_lighting, 1.0);" +
	"}";

	gl.shaderSource(gFragmentShaderObject_perVertex, fragmentShaderSourceCode_perVertex);
	gl.compileShader(gFragmentShaderObject_perVertex);
	if(gl.getShaderParameter(gFragmentShaderObject_perVertex, gl.COMPILE_STATUS) ==  false)
	{
		var error = gl.getShaderInfoLog(gFragmentShaderObject_perVertex);
		if(error.length > 0)
		{
			console.log("fragment shader error.\n");
			alert(error);
			wglUnintialise();
		}
	}


	//shader program
	gShaderProgramObject_perVertex = gl.createProgram();

	gl.attachShader(gShaderProgramObject_perVertex, gVertexShaderObject_perVertex);
	gl.attachShader(gShaderProgramObject_perVertex, gFragmentShaderObject_perVertex);

	//pre link binding
	gl.bindAttribLocation(gShaderProgramObject_perVertex, WebGLMacros.VDG_ATTRIBUTE_VERTEX, "vPosition");
	gl.bindAttribLocation(gShaderProgramObject_perVertex, WebGLMacros.VDG_ATTRIBUTE_NORMAL, "vNormal");

	//linking
	gl.linkProgram(gShaderProgramObject_perVertex);
	if(!gl.getProgramParameter(gShaderProgramObject_perVertex, gl.LINK_STATUS))
	{
		var error = gl.getProgramInfoLog(gShaderProgramObject_perVertex);
		if(error.length > 0)
		{
			alert(error);
			wglUnintialise();
		}
	}

	//get MVP uniform location
	vertex.modelUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_model_matrix");
	vertex.viewUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_view_matrix");
	vertex.projectionUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_projection_matrix");

	vertex.laUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_La");
	vertex.lsUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_Ls");
	vertex.ldUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_Ld");
	
	vertex.kaUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_Ka");
	vertex.ksUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_Ks");
	vertex.kdUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_Kd");
	vertex.materialShininessUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_material_shininess");

	vertex.lightPositionUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_Light_Position");
	vertex.LKeyPressedUniform = gl.getUniformLocation(gShaderProgramObject_perVertex, "u_LKeyPressed");

	/**** Per Fragment ****/

	//vertex shader
	gVertexShaderObject_perFragment = gl.createShader(gl.VERTEX_SHADER);

	var vertexShaderSourceCode_perFragment = 
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

	gl.shaderSource(gVertexShaderObject_perFragment, vertexShaderSourceCode_perFragment);
	gl.compileShader(gVertexShaderObject_perFragment);
	if(gl.getShaderParameter(gVertexShaderObject_perFragment, gl.COMPILE_STATUS) == false)
	{
		var error = gl.getShaderInfoLog(gVertexShaderObject_perFragment);
		if(error.length > 0)
		{
			console.log("vertex shader (per fragment) error.\n");
			alert(error);
			wglUnintialise();
		}
	}

	//fragment shader
	gFragmentShaderObject_perFragment = gl.createShader(gl.FRAGMENT_SHADER);

	var fragmentShaderSourceCode_perFragment = 
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

	gl.shaderSource(gFragmentShaderObject_perFragment, fragmentShaderSourceCode_perFragment);
	gl.compileShader(gFragmentShaderObject_perFragment);
	if(gl.getShaderParameter(gFragmentShaderObject_perFragment, gl.COMPILE_STATUS) ==  false)
	{
		var error = gl.getShaderInfoLog(gFragmentShaderObject_perFragment);
		if(error.length > 0)
		{
			console.log("fragment shader (per fragment) error.\n");
			alert(error);
			wglUnintialise();
		}
	}


	//shader program
	gShaderProgramObject_perFragment = gl.createProgram();

	gl.attachShader(gShaderProgramObject_perFragment, gVertexShaderObject_perFragment);
	gl.attachShader(gShaderProgramObject_perFragment, gFragmentShaderObject_perFragment);

	//pre link binding
	gl.bindAttribLocation(gShaderProgramObject_perFragment, WebGLMacros.VDG_ATTRIBUTE_VERTEX, "vPosition");
	gl.bindAttribLocation(gShaderProgramObject_perFragment, WebGLMacros.VDG_ATTRIBUTE_NORMAL, "vNormal");

	//linking
	gl.linkProgram(gShaderProgramObject_perFragment);
	if(!gl.getProgramParameter(gShaderProgramObject_perFragment, gl.LINK_STATUS))
	{
		var error = gl.getProgramInfoLog(gShaderProgramObject_perFragment);
		if(error.length > 0)
		{
			alert(error);
			wglUnintialise();
		}
	}

	//get MVP uniform location
	fragment.modelUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_model_matrix");
	fragment.viewUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_view_matrix");
	fragment.projectionUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_projection_matrix");

	fragment.laUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_La");
	fragment.lsUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_Ls");
	fragment.ldUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_Ld");
	
	fragment.kaUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_Ka");
	fragment.ksUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_Ks");
	fragment.kdUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_Kd");
	fragment.materialShininessUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_material_shininess");

	fragment.lightPositionUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_Light_Position");
	fragment.LKeyPressedUniform = gl.getUniformLocation(gShaderProgramObject_perFragment, "u_LKeyPressed");

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

	var modelMatrix_perVertex = mat4.create();
	var modelMatrix_perFragment = mat4.create();
	var viewMatrix_perVertex = mat4.create();
	var viewMatrix_perFragment = mat4.create();
	var projectionMatrix_perVertex = mat4.create();
	var projectionMatrix_perFragment = mat4.create();
	var translationMatrix_perVertex = mat4.create();
	var translationMatrix_perFragment = mat4.create();

	//per vertex transformation
	mat4.translate(modelMatrix_perVertex, modelMatrix_perVertex, [0.0, 0.0, -5.0]);
	projectionMatrix_perVertex = perspectiveMatrixProjection

	//per fragment transformation
	mat4.translate(modelMatrix_perFragment, modelMatrix_perFragment, [0.0, 0.0, -2.5]);
	projectionMatrix_perFragment = perspectiveMatrixProjection



	if(bPerVertex)
	{
		gl.useProgram(gShaderProgramObject_perVertex);

		//send necessary matrics to shaders in respective uniforms
		gl.uniformMatrix4fv(vertex.modelUniform, false, modelMatrix_perVertex);
		gl.uniformMatrix4fv(vertex.viewUniform, false, viewMatrix_perVertex);
		gl.uniformMatrix4fv(vertex.projectionUniform, false, projectionMatrix_perVertex);

		if(bLKeyPressed == true)
		{
			gl.uniform1i(vertex.LKeyPressedUniform, 1);

			//setting light properties
			gl.uniform3fv(vertex.laUniform, light_ambient);
			gl.uniform3fv(vertex.lsUniform, light_specular);
			gl.uniform3fv(vertex.ldUniform, light_diffuse);
			//setting material proprties
			gl.uniform3fv(vertex.kaUniform, material_ambient);
			gl.uniform3fv(vertex.ksUniform, material_specular);
			gl.uniform3fv(vertex.kdUniform, material_diffuse);
			//var lightPosition = [0.0, 0.0, 2.0, 1.0];
			gl.uniform4fv(vertex.lightPositionUniform, light_position);
		
			gl.uniform1f(vertex.materialShininessUniform, material_shininess);
		}
		else
		{
			gl.uniform1i(vertex.LKeyPressedUniform, 0);
		}

	}

	if(bPerFragment)
	{
		gl.useProgram(gShaderProgramObject_perFragment);

		//send necessary matrics to shaders in respective uniforms
		gl.uniformMatrix4fv(fragment.modelUniform, false, modelMatrix_perFragment);
		gl.uniformMatrix4fv(fragment.viewUniform, false, modelMatrix_perFragment);
		gl.uniformMatrix4fv(fragment.projectionUniform, false, projectionMatrix_perFragment);

		if(bLKeyPressed == true)
		{
			gl.uniform1i(fragment.LKeyPressedUniform, 1);

			//setting light properties
			gl.uniform3fv(fragment.laUniform, light_ambient);
			gl.uniform3fv(fragment.lsUniform, light_specular);
			gl.uniform3fv(fragment.ldUniform, light_diffuse);
			//setting material proprties
			gl.uniform3fv(fragment.kaUniform, material_ambient);
			gl.uniform3fv(fragment.ksUniform, material_specular);
			gl.uniform3fv(fragment.kdUniform, material_diffuse);
			//var lightPosition = [0.0, 0.0, 2.0, 1.0];
			gl.uniform4fv(fragment.lightPositionUniform, light_position);
		
			gl.uniform1f(fragment.materialShininessUniform, material_shininess);
		}
		else
		{
			gl.uniform1i(fragment.LKeyPressedUniform, 0);
		}

	}

	if (bPerVertex == true || bPerFragment == true)
	{
		sphere.draw();
	}

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
