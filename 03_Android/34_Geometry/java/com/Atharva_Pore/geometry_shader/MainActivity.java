package com.Atharva_Pore.geometry_shader;

//defaul given packages
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;						//for Bundle savedInstanceState 

//my packages
import android.view.Window;						//for Window.FEATURE_NO_TITLE
import android.view.WindowManager;				//for WindowManager.layoutParams.FLAG_FULLSCREEN
import android.content.pm.ActivityInfo;			//for ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
import android.graphics.Color;
import android.view.View;

public class MainActivity extends AppCompatActivity 
{
	private GLESView glesView;

    @Override
    protected void onCreate(Bundle savedInstanceState) 
    {
        super.onCreate(savedInstanceState);
        //setContentView(R.layout.activity_main);
        //R->Resources
        //layout-> app\src\main\res\layout

        //Remove Title Bar
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);
        getSupportActionBar().hide();

        //Remove Navigation Bar
        this.getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_IMMERSIVE);

        //make fullscren
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        //Forced Landscape Rotation
        this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        //set Background Colour
        this.getWindow().getDecorView().setBackgroundColor(Color.BLACK);

        //define own view
        glesView = new GLESView(this);

        //set the view as our view
        setContentView(glesView);
    }

    @Override
    protected void onPause()
    {
    	super.onPause();
    }

    @Override
    protected void onResume()
    {
    	super.onResume();
    }
}
