package com.Atharva_Pore.win_my_view_as_object;

//my packages
import androidx.appcompat.widget.AppCompatTextView; //for AppCompatTextView
import android.content.Context; 					//for Context drawingContext
import android.view.Gravity;						//for Gravity	
import android.graphics.Color;						//for Color				

public class MyView extends AppCompatTextView
{
	public MyView(Context drawingContext)
	{
		super(drawingContext);
		setTextColor(Color.rgb(0, 255, 0));
		setTextSize(60);
		setGravity(Gravity.CENTER);
		setText("Hello World !!!");
	}
}
