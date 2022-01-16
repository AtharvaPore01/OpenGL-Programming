#pragma once
#ifndef _DYNAMIC_INDIA_H_
#define _DYNAMIC_INDIA_H_

#include <Windows.h>
#include <GL/glew.h>
#include <GL/GL.h>

#endif // !_DYNAMIC_INDIA_H_

//flags
bool b_I_Done = false;
bool b_N_Done = false;
bool b_D_Done = false;
bool b_i_Done = false;
bool b_A_Done = false;
bool b_clip_top_plane = false;
bool b_clip_bottom_plane = false;
bool b_unclip_top_plane = false;
bool b_unclip_bottom_plane = false;

bool b_appear_middle_strip = true;

bool b_top_plane_smoke_done = false;
bool b_bottom_plane_smoke_done = false;
bool b_middle_plane_smoke_done = false;

bool b_start_decrementing = false;
bool b_start_incrementing = false;
bool b_PlaneTrue = false;


//variables for translation of I,N,D,i,A
GLfloat f_Translate_I = -3.0f;		//translate along x
GLfloat f_Translate_N = 3.0f;		//translate along y
GLfloat f_Translate_D = 0.0f;
GLfloat f_Translate_i = -3.0f;		//translate along y
GLfloat f_Translate_A = 3.0f;		//translate along x

//color values for D
GLfloat f_DRedColor = 0.0f;
GLfloat f_DGreenColor = 0.0f;
GLfloat f_DBlueColor = 0.0f;

//A middle strips colors
GLfloat f_ARedColor = 0.0f;
GLfloat f_AGreenColor = 0.0f;
GLfloat f_ABlueColor = 0.0f;
GLfloat f_AWhiteColor = 0.0f;

//smoke colors
GLfloat f_red = 1.0f;
GLfloat f_green = 0.5f;
GLfloat f_blue = 1.0f;
GLfloat f_white = 1.0f;

/* angles to draw a smoke */
GLfloat top_angle_1 = 3.14159f;
GLfloat top_angle_2 = 3.14659f;

GLfloat top_angle_3 = 4.71238f;
GLfloat top_angle_4 = 4.71738f;

GLfloat bottom_angle_1 = 3.13659f;
GLfloat bottom_angle_2 = 3.14159f;

GLfloat bottom_angle_3 = 1.57079f;
GLfloat bottom_angle_4 = 1.56579f;

//plane initial position variable
GLfloat x_plane_pos = 0.0f;
GLfloat y_plane_pos = 0.0f;

GLfloat middle_plane_smoke = 0.0f;

//struct
struct plane
{
	GLfloat _x;
	GLfloat _y;
	GLfloat radius = 10.0f;
	GLfloat angle = (GLfloat)M_PI;
	GLfloat rotation_angle = 0.0f;
}top, bottom;
