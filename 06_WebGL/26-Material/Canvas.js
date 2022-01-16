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

var MaterialAmbient=[0.0, 0.0, 0.0];
var MaterialDiffuse=[1.0, 1.0, 1.0];
var MaterialSpecular=[1.0, 1.0, 1.0];
var MaterialShininess=128.0;

/*
var modelUniform;
var viewUniform;
var projectionUniform;
var laUniform, ldUniform, lsUniform;
var kaUniform, kdUniform, ksUniform, materialShininessUniform;
var LKeyPressedUniform;
var lightPositionUniform;
*/

var model_uniform;
var view_uniform;
var projection_uniform;
var La_uniform;
var Ld_uniform;
var Ls_uniform;
var lightPosition_uniform;
var Ka_uniform;
var Kd_uniform;
var Ks_uniform;
var shininess_uniform;
var LKeyPressed_Uniform;

var mat_arr = 
[
	//emrald
	{
		MaterialAmbient : [0.0215, 0.1745, 0.07568],
		MaterialDiffuse : [0.07568, 0.61424, 0.07568],
		MaterialSpecular : [0.633, 0.727811, 0.633],
		MaterialShininess : 0.6 * 128.0
	},

	//jade
	{
		MaterialAmbient : [0.135, 0.2225, 0.1575],
		MaterialDiffuse : [0.54, 0.89, 0.63],
		MaterialSpecular : [0.316228, 0.316228, 0.316228],
		MaterialShininess : 0.1 * 128.0
	},

	//obsidian
	{
		MaterialAmbient : [0.05375, 0.05, 0.06625],
		MaterialDiffuse : [0.18275, 0.17, 0.22525],
		MaterialSpecular : [0.332741, 0.328634, 0.346435],
		MaterialShininess : 0.3 * 128.0
	},

	//pearl
	{
		MaterialAmbient : [0.25, 0.20725, 0.20725],
		MaterialDiffuse : [1.0, 0.829, 0.829],
		MaterialSpecular : [0.296648, 0.296648, 0.296648],
		MaterialShininess : 0.088 * 128.0
	},

	//ruby
	{
		MaterialAmbient : [0.1745, 0.01175, 0.01175],
		MaterialDiffuse : [0.61424, 0.04136, 0.04136],
		MaterialSpecular : [0.727811, 0.625969, 0.625969],
		MaterialShininess : 0.6 * 128.0
	},

	//Turquoise
	{
		MaterialAmbient : [0.1, 0.18725, 0.1745],
		MaterialDiffuse : [0.396, 0.74151, 0.69102],
		MaterialSpecular : [0.297254, 0.30829, 0.306678],
		MaterialShininess : 0.1 * 128.0
	},

	//brass
	{
		MaterialAmbient : [0.329412, 0.223529, 0.027451],
		MaterialDiffuse : [0.782392, 0.568627, 0.113725],
		MaterialSpecular : [0.992157, 0.941176, 0.807843],
		MaterialShininess : 0.21794872 * 128.0
	},

	//bronze
	{
		MaterialAmbient : [0.2125, 0.1275, 0.054],
		MaterialDiffuse : [0.714, 0.4284, 0.18144],
		MaterialSpecular : [0.393548, 0.271906, 0.166721],
		MaterialShininess : 0.2 * 128.0
	},

	//chrome
	{
		MaterialAmbient : [0.25, 0.25, 0.25],
		MaterialDiffuse : [0.4, 0.4, 0.4],
		MaterialSpecular : [0.774597, 0.774597, 0.774597],
		MaterialShininess : 0.6 * 128.0
	},

	//copper
	{
		MaterialAmbient : [0.19125, 0.0735, 0.0225],
		MaterialDiffuse : [0.7038, 0.27048, 0.0828],
		MaterialSpecular : [0.256777, 0.137622, 0.086014],
		MaterialShininess : 0.1 * 128.0
	},

	//gold
	{
		MaterialAmbient : [0.24725, 0.1995, 0.0745],
		MaterialDiffuse : [0.75164, 0.60648, 0.22648],
		MaterialSpecular : [0.628281, 0.555802, 0.366065],
		MaterialShininess : 0.4 * 128.0
	},

	//silver
	{
		MaterialAmbient : [0.19225, 0.19225, 0.19225],
		MaterialDiffuse : [0.50754, 0.50754, 0.50754],
		MaterialSpecular : [0.508273, 0.508273, 0.508273],
		MaterialShininess : 0.4 * 128.0
	},

	/* plastic */

	//black
	{
		MaterialAmbient : [0.0, 0.0, 0.0],
		MaterialDiffuse : [0.01, 0.01, 0.01],
		MaterialSpecular : [0.50, 0.50, 0.50],
		MaterialShininess : 0.25 * 128.0
	},

	//cyan
	{
		MaterialAmbient : [0.0, 0.1, 0.06],
		MaterialDiffuse : [0.01, 0.50980392, 0.50980392],
		MaterialSpecular : [0.50196078, 0.50196078, 0.50196078],
		MaterialShininess : 0.25 * 128.0
	},

	//green
	{
		MaterialAmbient : [0.0, 0.0, 0.0],
		MaterialDiffuse : [0.1, 0.35, 0.1],
		MaterialSpecular : [0.45, 0.55, 0.45],
		MaterialShininess : 0.25 * 128.0
	},

	//red
	{
		MaterialAmbient : [0.0, 0.0, 0.0],
		MaterialDiffuse : [0.5, 0.0, 0.0],
		MaterialSpecular : [0.7, 0.6, 0.6],
		MaterialShininess : 0.25 * 128.0
	},

	//white
	{
		MaterialAmbient : [0.0, 0.0, 0.0],
		MaterialDiffuse : [0.55, 0.55, 0.55],
		MaterialSpecular : [0.70, 0.70, 0.70],
		MaterialShininess : 0.25 * 128.0
	},

	//yellow
	{
		MaterialAmbient : [0.0, 0.0, 0.0],
		MaterialDiffuse : [0.5, 0.5, 0.0],
		MaterialSpecular : [0.60, 0.60, 0.50],
		MaterialShininess : 0.25 * 128.0
	},

	/* rubber */

	//black
	{
		MaterialAmbient : [0.02, 0.02, 0.02],
		MaterialDiffuse : [0.01, 0.01, 0.01],
		MaterialSpecular : [0.4, 0.4, 0.4],
		MaterialShininess : 0.078125 * 128.0
	},

	//cyan
	{
		MaterialAmbient : [0.0, 0.05, 0.05],
		MaterialDiffuse : [0.4, 0.5, 0.5],
		MaterialSpecular : [0.04, 0.7, 0.7],
		MaterialShininess : 0.078125 * 128.0
	},

	//green
	{
		MaterialAmbient : [0.0, 0.05, 0.0],
		MaterialDiffuse : [0.4, 0.5, 0.4],
		MaterialSpecular : [0.04, 0.7, 0.04],
		MaterialShininess : 0.078125 * 128.0
	},

	//red
	{
		MaterialAmbient : [0.05, 0.0, 0.0],
		MaterialDiffuse : [0.5, 0.4, 0.4],
		MaterialSpecular : [0.7, 0.04, 0.04],
		MaterialShininess : 0.078125 * 128.0
	},

	//white
	{
		MaterialAmbient : [0.05, 0.05, 0.05],
		MaterialDiffuse : [0.5, 0.5, 0.5],
		MaterialSpecular : [0.7, 0.7, 0.7],
		MaterialShininess : 0.078125 * 128.0
	},

	//yellow
	{
		MaterialAmbient : [0.05, 0.05, 0.0],
		MaterialDiffuse : [0.5, 0.5, 0.4],
		MaterialSpecular : [0.7, 0.7, 0.04],
		MaterialShininess : 0.078125 * 128.0
	},

];

var bLKeyPressed = false;

var iCount = 0;

var sphere = null;

//rotation related variables
var lightAngle = 0.0;

var giWindowWidth = 0
var giWindowHeight = 0

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
		// ASCII for x or X
		case 88:
			iCount = 1;
			break;
		// ASCII for y or Y
		case 89:
			iCount = 2;
			break;
		// ASCII for z or Z
		case 90:
			iCount = 3;
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
	model_uniform = gl.getUniformLocation(shaderProgramObject, "u_model_matrix");
	view_uniform = gl.getUniformLocation(shaderProgramObject, "u_view_matrix");
	projection_uniform = gl.getUniformLocation(shaderProgramObject, "u_projection_matrix");

	La_uniform = gl.getUniformLocation(shaderProgramObject, "u_La");
	Ls_uniform = gl.getUniformLocation(shaderProgramObject, "u_Ls");
	Ld_uniform = gl.getUniformLocation(shaderProgramObject, "u_Ld");
	
	Ka_uniform = gl.getUniformLocation(shaderProgramObject, "u_Ka");
	Ks_uniform = gl.getUniformLocation(shaderProgramObject, "u_Ks");
	Kd_uniform = gl.getUniformLocation(shaderProgramObject, "u_Kd");
	shininess_uniform = gl.getUniformLocation(shaderProgramObject, "u_material_shininess");

	lightPosition_uniform = gl.getUniformLocation(shaderProgramObject, "u_Light_Position");
	LKeyPressed_Uniform = gl.getUniformLocation(shaderProgramObject, "u_LKeyPressed");

	//sphere
	sphere = new Mesh();
	makeSphere(sphere, 2.0, 30, 30);

	//set clear color
	gl.clearColor(0.3, 0.3, 0.3, 1.0);

	//depth test
	gl.enable(gl.DEPTH_TEST);

	//toggleFullScreen();

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

	giWindowWidth = canvas.width;
	giWindowHeight = canvas.height;

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
	lightAngle = lightAngle + 0.005;
	if (lightAngle >= 360)
	{
		lightAngle = 0.0;
	}
}

function wglDraw24Spheres()
{
	//variable declaration
	var i = 0;
	var x  = 0;
	var y  = 0;
	var w  = 0;
	var h  = 0;

	//declaration of metrices
	var modelMatrix = mat4.create();
	var viewMatrix = mat4.create();
	var projectionMatrix = mat4.create();
	var translationMatrix = mat4.create();

	for (i = 0; i < 24; i++)
	{
		mat4.identity(modelMatrix);
		mat4.identity(viewMatrix);
		mat4.identity(projectionMatrix);
		mat4.identity(translationMatrix);


		x = parseInt((i % 6) * giWindowWidth / 6);
		y = parseInt(giWindowHeight - parseInt(i / 6 + 1) * giWindowHeight / 4);
		w = parseInt(giWindowWidth / 6);
		h = parseInt(giWindowHeight / 4);
		gl.viewport(x, y, w, h);

		//perspectiveMatrixProjection = vmath::perspective(45.0f, (GLfloat)(giWindowWidth / 6) / (GLfloat)(giWindowHeight / 4), 0.1f, 100.0f);
		
		//perspective Projection Matrix
		mat4.perspective(	perspectiveMatrixProjection,
							45.0,
							parseFloat(giWindowWidth / 6) / parseFloat(giWindowHeight / 4),
							0.1,
							100.0);

		mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -10.0]);

		//send necessary matrics to shaders in respective uniforms
		gl.uniformMatrix4fv(model_uniform, false, modelMatrix);
		gl.uniformMatrix4fv(view_uniform, false, viewMatrix);
		gl.uniformMatrix4fv(projection_uniform, false, perspectiveMatrixProjection);


		//if light is enabled
		if (bLKeyPressed)
		{
			gl.uniform1i(LKeyPressed_Uniform, 1);

			//setting light properties
			gl.uniform3fv(La_uniform, light_ambient);
			gl.uniform3fv(Ls_uniform, light_specular);
			gl.uniform3fv(Ld_uniform, light_diffuse);
			//setting material proprties
			gl.uniform3fv(Ka_uniform, mat_arr[i].MaterialAmbient);
			gl.uniform3fv(Ks_uniform, mat_arr[i].MaterialSpecular);
			gl.uniform3fv(Kd_uniform, mat_arr[i].MaterialDiffuse);
			gl.uniform1f(shininess_uniform, MaterialShininess);

			if(iCount == 1)
			{
				light_position = [0.0, 100.0 * Math.cos(lightAngle), 100.0 * Math.sin(lightAngle), 1.0];
				gl.uniform4fv(lightPosition_uniform, light_position);
			}

			if(iCount == 2)
			{
				light_position = [100.0 * Math.cos(lightAngle), 0.0, 100.0 * Math.sin(lightAngle), 1.0];	
				gl.uniform4fv(lightPosition_uniform, light_position);
			}

			if(iCount == 3)
			{
				light_position = [100.0 * Math.cos(lightAngle), 100.0 * Math.sin(lightAngle), 0.0, 1.0];	
				gl.uniform4fv(lightPosition_uniform, light_position);
			}			
		}
		else
		{
			//notify shader that we aren't pressed the "L" key
			gl.uniform1i(LKeyPressed_Uniform, 0);
		}

		sphere.draw();
	}
}

function wglDraw()
{
	//code
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	gl.useProgram(shaderProgramObject);
	
	
	wglDraw24Spheres()

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
