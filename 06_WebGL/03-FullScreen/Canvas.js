//global variables
var canvas 	= null;
var context = null;

//onload function
function main()
{
	//get <canvas> element
	canvas = document.getElementById("AAP");
	if(!canvas)
		console.log("Obtaining Canvas Failed\n");
	else
		console.log("Obtaining Canvas Succeeded\n");

	//print canvas width and height
	console.log("Canvas Width : "+canvas.width+"And Canvas Height : "+canvas.height);

	//get 2D context
	context = canvas.getContext("2d");
	if(!context)
		console.log("Obtaining 2d Context Failed\n");
	else
		console.log("Obtaining 2d Context Succeeded\n");

	//fill canvas with black color
	context.fillStyle="black";
	context.fillRect(0, 0, canvas.width, canvas.height);

	//calling draw text
	drawText("Hello World !!!");

	//register keyboard's keydown handle
	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
}

function drawText(text)
{
	//center the text
	context.textAlign="center";
	context.textBaseLine="middle";

	//text font
	context.font="48px sans-serif";

	//text color
	context.fillStyle="green";

	//display text in the center
	context.fillText(text, canvas.width/2, canvas.height/2)
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
	}
}

//event handlers
function keyDown(event)
{
	//code
	switch(event.keyCode)
	{
		case 70: 	//'F' or 'f' ascii value
			//full screen call
			toggleFullScreen();
			//repaint.
			drawText("Hello World !!!");
			break;
	}
}

function mouseDown()
{
	//code
}
