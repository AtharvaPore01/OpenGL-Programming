//Steps To Create A Window
/*
	1.	Take A Canvas From HTML 5
	2.	Take A Ordinary Text And Windowing Context From Canvas
	3.	Make Background Color Black.
	4.	Define A String To Display
	5.	Set An Attribute To Put Defined String In The Middle Of The Canvas.
	6.	Set The Context Font Of The String.
	7.	Show The String
*/
function main()
{
	//get <canvas> element
	var canvas = document.getElementById("AAP");
	if(!canvas)
		console.log("Obtaining Canvas Failed\n");
	else
		console.log("Obtaining Canvas Succeeded\n");

	//print canvs width and height on console
	console.log("Canvas Width : "+canvas.width+"And Canvas Height : "+canvas.height);

	//get 2D context
	var context = canvas.getContext("2d");
	if(!context)
		console.log("Obtaining 2d Context Failed\n");
	else
		console.log("Obtaining 2d Context Succeeded\n");

	//fill the canvs with black color
	context.fillStyle="black";	//#000000
	context.fillRect(0, 0, canvas.width, canvas.height);

	//center the text
	context.textAlign="center";		//center horizontally
	context.textBaseline="middle";	//center vertically

	//text
	var str="Hello World !!!";

	//text font 
	context.font="48px sans-serif";

	//text color
	context.fillStyle="green";

	//display text in the center
	context.fillText(str, canvas.width/2, canvas.height/2);
}
