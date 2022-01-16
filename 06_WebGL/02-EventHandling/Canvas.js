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

	//center the text
	context.textAlign="center";
	context.textBaseLine="middle";

	//text
	var str = "Hello World !!!";

	//text font
	context.font="48px sans-serif";

	//text color
	context.fillStyle="green";

	//display text in the center
	context.fillText(str, canvas.width/2, canvas.height/2)

	//register keyboard's keydown handle
	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
}

//event handlers
function keyDown(event)
{
	//code
	alert("Key Is Pressed");
}

function mouseDown()
{
	//code
	alert("Mouse Is Clicked");
}
