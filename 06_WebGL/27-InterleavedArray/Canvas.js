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
var vao;
var vbo;

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

var samplerUniform;
var marble_texture

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
	"precision mediump int;" +

	"in vec4 vPosition;" +
	"in vec4 vColor;" +
	"in vec3 vNormal;" +
	"in vec2 vTexCoord;" +

	"uniform mat4 u_model_matrix;" +
	"uniform mat4 u_view_matrix;" +
	"uniform mat4 u_projection_matrix;" +
	"uniform mat4 u_mvp_matrix;" +
	"uniform int u_LKeyPressed;" +
	"uniform vec4 u_light_position;" +

	"out vec3 t_norm;" +
	"out vec3 light_direction;" +
	"out vec3 viewer_vector;" +
	"out vec4 out_color;" +
	"out vec2 out_texcoord;" +
	"void main(void)" +
	"{" +
		"if (u_LKeyPressed == 1)" +
		"{" +
				"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;" +
				"mat3 normal_matrix = mat3(transpose(inverse(u_view_matrix * u_model_matrix)));" +
				"t_norm = normal_matrix * vNormal;" +
				"light_direction = vec3(u_light_position - eye_coordinates);" +
				"viewer_vector = vec3(-eye_coordinates);" +
		"}" +
	"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
	"out_color = vColor;" +
	"out_texcoord = vTexCoord;" +				
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
	"in vec3 t_norm;" +
	"in vec3 light_direction;" +
	"in vec3 viewer_vector;" +
	"in vec4 out_color;" +
	"in vec2 out_texcoord;" +

	"uniform int u_LKeyPressed;" +
	"uniform vec3 u_La;" +
	"uniform vec3 u_Ld;" +
	"uniform vec3 u_Ls;" +
	"uniform vec4 u_light_position;" +
	"uniform vec3 u_Ka;" +
	"uniform vec3 u_Kd;" +
	"uniform vec3 u_Ks;" +
	"uniform float shininess;" +
	"uniform sampler2D u_sampler;" +

	"vec3 phong_ads_light;" +
	"out vec4 FragColor;" +

	"void main(void)" +
	"{" +
		"if(u_LKeyPressed == 1)" +
			"{" +
			"vec3 normalised_transformed_normal = normalize(t_norm);" +
			"vec3 normalised_light_direction = normalize(light_direction);" +
			"vec3 normalised_viewer_vector = normalize(viewer_vector);" +
			"vec3 reflection_vector = reflect(-normalised_light_direction, normalised_transformed_normal);" +
			"float tn_dot_LightDirection = max(dot(normalised_light_direction, normalised_transformed_normal), 0.0);" +
			"vec3 ambient = u_La * u_Ka;" +
			"vec3 diffuse = u_Ld * u_Kd * tn_dot_LightDirection;" +
			"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalised_viewer_vector), 0.0), shininess);" +
			"phong_ads_light = (ambient + diffuse + specular) * vec3(out_color * texture(u_sampler, out_texcoord));" +
		"}" +
		"else" +
		"{" +
			"phong_ads_light = vec3(out_color * texture(u_sampler, out_texcoord));"  +
		"}" +
		"FragColor = vec4(phong_ads_light, 1.0);" +
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
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.VDG_ATTRIBUTE_COLOR, "vColor");
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.VDG_ATTRIBUTE_TEXTURE0, "vTexCoord");

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

	//load smiley texture
	marble_texture = gl.createTexture();
	marble_texture.image = new Image();
	marble_texture.image.src = "marble.png";
	marble_texture.image.onload = function()
	{
		gl.bindTexture(gl.TEXTURE_2D, marble_texture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, marble_texture.image);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.bindTexture(gl.TEXTURE_2D, null);
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
	materialShininessUniform = gl.getUniformLocation(shaderProgramObject, "shininess");

	lightPositionUniform = gl.getUniformLocation(shaderProgramObject, "u_light_position");
	LKeyPressedUniform = gl.getUniformLocation(shaderProgramObject, "u_LKeyPressed");

	samplerUniform = gl.getUniformLocation(shaderProgramObject, "u_sampler");

	var cubeVCNT = new Float32Array	([
											
										//vertices 			//color 			//texcoord 		//normal
										1.0, 1.0, -1.0,		1.0, 0.0, 0.0,		0.0, 1.0,		0.0, 1.0, 0.0,
										-1.0, 1.0, -1.0,	1.0, 0.0, 0.0,		0.0, 0.0,		0.0, 1.0, 0.0,
										-1.0, 1.0, 1.0,		1.0, 0.0, 0.0,		1.0, 0.0,		0.0, 1.0, 0.0,
										1.0, 1.0, 1.0,		1.0, 0.0, 0.0,		1.0, 1.0,		0.0, 1.0, 0.0,
								
										1.0, -1.0, -1.0,	0.0, 1.0, 0.0,		1.0, 1.0,		0.0, -1.0, 0.0,
										-1.0, -1.0, -1.0,	0.0, 1.0, 0.0,		0.0, 1.0,		0.0, -1.0, 0.0,
										-1.0, -1.0, 1.0,	0.0, 1.0, 0.0,		0.0, 0.0,		0.0, -1.0, 0.0,
										1.0, -1.0, 1.0,		0.0, 1.0, 0.0,		1.0, 0.0,		0.0, -1.0, 0.0,
								
										1.0, 1.0, 1.0,		0.0, 0.0, 1.0,		0.0, 0.0,		0.0, 0.0, 1.0,
										-1.0, 1.0, 1.0,		0.0, 0.0, 1.0,		1.0, 0.0,		0.0, 0.0, 1.0,
										-1.0, -1.0, 1.0,	0.0, 0.0, 1.0,		1.0, 1.0,		0.0, 0.0, 1.0,
										1.0, -1.0, 1.0,		0.0, 0.0, 1.0,		0.0, 1.0,		0.0, 0.0, 1.0,
								
										1.0, 1.0, -1.0,		0.0, 1.0, 1.0,		1.0, 0.0,		0.0, 0.0, -1.0,
										-1.0, 1.0, -1.0,	0.0, 1.0, 1.0,		1.0, 1.0,		0.0, 0.0, -1.0,
										-1.0, -1.0, -1.0,	0.0, 1.0, 1.0,		0.0, 1.0,		0.0, 0.0, -1.0,
										1.0, -1.0, -1.0,	0.0, 1.0, 1.0,		0.0, 0.0,		0.0, 0.0, -1.0,
								
										1.0, 1.0, -1.0,		1.0, 0.0, 1.0,		1.0, 0.0,		1.0, 0.0, 0.0,
										1.0, 1.0, 1.0,		1.0, 0.0, 1.0,		1.0, 1.0,		1.0, 0.0, 0.0,
										1.0, -1.0, 1.0,		1.0, 0.0, 1.0,		0.0, 1.0,		1.0, 0.0, 0.0,
										1.0, -1.0, -1.0,	1.0, 0.0, 1.0,		0.0, 0.0,		1.0, 0.0, 0.0,
								
										-1.0, 1.0, -1.0,	1.0, 1.0, 0.0,		0.0, 0.0,		-1.0, 0.0, 0.0,
										-1.0, 1.0, 1.0,		1.0, 1.0, 0.0,		1.0, 0.0,		-1.0, 0.0, 0.0,
										-1.0, -1.0, 1.0,	1.0, 1.0, 0.0,		1.0, 1.0,		-1.0, 0.0, 0.0,
										-1.0, -1.0, -1.0,	1.0, 1.0, 0.0,		0.0, 1.0,		-1.0, 0.0, 0.0

									]);
							
	//cube
	vao = gl.createVertexArray();
	gl.bindVertexArray(vao);

	//position
	vbo = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

	gl.bufferData(gl.ARRAY_BUFFER, cubeVCNT, gl.STATIC_DRAW);
	
	//position
	gl.vertexAttribPointer(	WebGLMacros.VDG_ATTRIBUTE_VERTEX,
							3,
							gl.FLOAT,
							false,
							11 * 4, 
							0 * 4);
	gl.enableVertexAttribArray(WebGLMacros.VDG_ATTRIBUTE_VERTEX);

	//color
	gl.vertexAttribPointer(	WebGLMacros.VDG_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							11 * 4,
							3 * 4);
	gl.enableVertexAttribArray(WebGLMacros.VDG_ATTRIBUTE_COLOR);

	//texcoord
	gl.vertexAttribPointer(	WebGLMacros.VDG_ATTRIBUTE_TEXTURE0,
							2,
							gl.FLOAT,
							false,
							11 * 4,
							6 * 4);
	gl.enableVertexAttribArray(WebGLMacros.VDG_ATTRIBUTE_TEXTURE0);

	//normal
	gl.vertexAttribPointer(	WebGLMacros.VDG_ATTRIBUTE_NORMAL,
							3,
							gl.FLOAT,
							false,
							11 * 4,
							8 * 4);
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
		gl.uniform3fv(laUniform, light_ambient);
		gl.uniform3fv(lsUniform, light_specular);
		gl.uniform3fv(ldUniform, light_diffuse);
		//setting material proprties
		gl.uniform3fv(kaUniform, material_ambient);
		gl.uniform3fv(ksUniform, material_specular);
		gl.uniform3fv(kdUniform, material_diffuse);
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
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -4.0]);
	mat4.rotateX(modelMatrix, modelMatrix, degToRad(angleCube));
	mat4.rotateY(modelMatrix, modelMatrix, degToRad(angleCube));
	mat4.rotateZ(modelMatrix, modelMatrix, degToRad(angleCube));

	//send necessary matrics to shaders in respective uniforms
	gl.uniformMatrix4fv(modelUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionUniform, false, perspectiveMatrixProjection);

	//bind texture
	gl.bindTexture(gl.TEXTURE_2D, marble_texture);
	gl.uniform1i(samplerUniform, 0);

	//bind with vao
	gl.bindVertexArray(vao);
	
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
