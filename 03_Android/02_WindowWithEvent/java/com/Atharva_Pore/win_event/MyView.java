package com.Atharva_Pore.win_event;

//my packages
import androidx.appcompat.widget.AppCompatTextView; 		//for AppCompatTextView class
import android.content.Context; 							//for Context drawingContext class
import android.graphics.Color;								//for Color class
import android.view.Gravity;								//for Gravity class	
import android.view.MotionEvent;							//for MotionEvent			
import android.view.GestureDetector;						//for GestureDetector
import android.view.GestureDetector.OnGestureListener;		//for OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener;	//for OnDoubleTapListener

public class MyView extends AppCompatTextView implements OnGestureListener, OnDoubleTapListener
{
	private GestureDetector gestureDetector;

	public MyView(Context context)
	{
		super(context);
		setTextColor(Color.rgb(255, 128, 0));
		setTextSize(60);
		setGravity(Gravity.CENTER);
		setText("Hello World !!!");

		gestureDetector = new GestureDetector(context, this, null, false);
		/*	GestureDetector Asks 
		*	Parameter 1 :- Give Me The Global Environment.
		*	Parameter 2 :- Who Will Listen? We Tell Him Me So this is given.
		*	Parameter 3 :- Does There Anyone Who Will Handle This? We Say No So null.
		*	Parameter 4 :- unused(internally used by android) so STRICTLY false.
		*/ 
	}

	// Handling 'OnTouchEvent' Is The Most IMPORTANT
	// Because It Triggers All Gesture And Tap Event.

	@Override
	public boolean onTouchEvent(MotionEvent event)
	{
		//code
		int eventaction = event.getAction();	//this is not requiered in any OpenGL Code.
		if(!gestureDetector.onTouchEvent(event))
		{
			super.onTouchEvent(event);
		}
		return(true);
	} 

	// abstract method for onDoubleTapEvent so must be implementd
	@Override
	public boolean onDoubleTap(MotionEvent event)
	{
		setTextColor(Color.rgb(255, 0, 0));
		setGravity(Gravity.CENTER);
		setText("Double Tap");
		return(true);
	}

	// abstract method for onDoubleTapEvent so must be implementd
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//No Code Because We Already Written In 'onDoubleTap'
		return(true);
	}

	// abstract method for onDoubleTapEvent so must be implementd
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		setTextColor(Color.rgb(0, 255, 0));
		setGravity(Gravity.CENTER);
		setText("Single Tap");
		return(true);
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public boolean onDown(MotionEvent e)
	{
		//No Code Because We Already Written In onSingleTapConfirmed
		return(true);
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocity_x, float velocity_y)
	{
		return(true);
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public void onLongPress(MotionEvent e)
	{
		setTextColor(Color.rgb(0, 0, 255));
		setGravity(Gravity.CENTER);
		setText("Long Press");
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public boolean onScroll(MotionEvent e1, MotionEvent e2, float distance_x, float distance_y)
	{
		setTextColor(Color.rgb(255, 255, 0));
		setGravity(Gravity.CENTER);
		setText("Scroll");
		return(true);
	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public void onShowPress(MotionEvent e)
	{

	}

	// abstract method for onGestureListner so must be implementd
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return(true);
	}
}
