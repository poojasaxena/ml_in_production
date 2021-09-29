function()
{
    canvas.removeEventListener( "mousemove", onPaint, false );

}, false );

var onPaint = function()
{
    context.lineWidth = context.lineWidth;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.strokeStyle = context.color;

    context.beginPath();
    context.moveTo( lastMouse.x, lastMouse.y );
    context.lineTo( Mouse.x, Mouse.y );
    context.closePath();
    context.stroke();
};

function debug()
{
    /* CLEAR BUTTON */
    var clearButton = $( "#clearButton" );
    clearButton.on( "click", function()
		    {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);

		    });
    $( "#colors" ).change(function()
			  {
			      var color = $( "#colors" ).val();
			      context.color = color;
			  });
    $( "#lineWidth" ).change(function()
			     {
				 context.lineWidth = $( this ).val();
			     });
}
}());~
