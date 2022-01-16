#pragma once
#ifndef _VISUALIZATION_H

//header files
#include <stdio.h>
#include "Graph.h"
#include "ServerList.h"
#include "Stack.h"

//macros
#define A					0
#define B					1
#define C					2
#define D					3
#define E					4
#define F					5
#define G					6
#define H					7

#define LOWER_LIMIT			1
#define UPPER_LIMIT			5
#define COUNT				1

#define CIRCLE_POINTS		4000
#define RADIUS				1.0f
#define TRANSLATE_Z			-30.0f

#define LOWER_LIMIT_X_Y		1
#define UPPER_LIMIT_X_Y		16

#define NR_VERTICES			6

#define MYAUDIO				101
#define MYAUDIO_1				102

//typedef
typedef int lower_limit_t;
typedef int upper_limit_t;

typedef float f_lower_limit_t;
typedef float f_upper_limit_t;
typedef float fWeight_t;

typedef int count_t;

typedef GLfloat red_t;
typedef GLfloat green_t;
typedef GLfloat blue_t;
typedef GLfloat alpha_t;
typedef GLfloat translate_t;

//flags for visualization
bool bEdgeAdded = false;
bool bDrawEdge = false;
bool bVertexAdded = false;
bool bSPressed = false;
bool bVisual = false;
bool bNode = false;

bool b_B = false;
bool b_C = false;
bool b_E = false;
bool b_F = false;
bool b_G = false;
bool b_H = false;

bool AtoB = false;
bool AtoE = false;
bool AtoF = false;
bool AtoG = false;
bool AtoH = false;

bool BtoA = false;
bool BtoC = false;

bool CtoB = false;
bool CtoE = false;
bool CtoD = false;

bool EtoA = false;
bool EtoC = false;
bool EtoD = false;
bool EtoF = false;
bool EtoG = false;

bool FtoA = false;
bool FtoE = false;
bool FtoD = false;
bool FtoH = false;

bool GtoA = false;
bool GtoE = false;

bool HtoA = false;
bool HtoF = false;

bool b_edges = false;

bool b_p = false;
bool b_n = false;
bool b_s = false;


//variables for visualization
alpha_t alpha = 0.0f;
red_t red = 0.0f;
green_t green = 0.0f;
blue_t blue = 0.0f;
x_t x = 0.0f;
y_t y = 0.0f;
ap_node *list_run = NULL;
ap_node *start_list_run = NULL;

red_t Red = 0.0f;
green_t Green = 0.0f;
blue_t Blue = 0.0f;

float edge_draw_count = 0.0f;

static Ret_t ret_v;
static Ret_t ret_e;
static int count = 0;
static int count_e = 1;
float counter = 0.0f;
float path_counter = 0.0f;

float node_counter = 0.0f;
float string_counter = 0.0f;
float show_node_counter = 0.0f;
ap_node *nodeList_run = NULL;
int count_node_B = 0;
int count_node_C = 0;
int count_node_E = 0;
int count_node_F = 0;
int count_node_G = 0;
int count_node_H = 0;

char str_A[] = { "A" };
char str_B[] = { "B" };
char str_C[] = { "C" };
char str_D[] = { "D" };
char str_E[] = { "E" };
char str_F[] = { "F" };
char str_G[] = { "G" };
char str_H[] = { "H" };

char str_AtoB[] = { "A To B" };
char str_AtoE[] = { "A To E" };
char str_AtoF[] = { "A To F" };
char str_AtoG[] = { "A To G" };
char str_AtoH[] = { "A To H" };

char str_BtoA[] = { "B To A" };
char str_BtoC[] = { "B To C" };

char str_FtoA[] = { "F to A" };
char str_FtoE[] = { "F to E" };
char str_FtoD[] = { "F to D" };
char str_FtoH[] = { "F to H" };

char str_EtoA[] = { "E to A" };
char str_EtoC[] = { "E to C" };
char str_EtoF[] = { "E to F" };
char str_EtoD[] = { "E to D" };
char str_EtoG[] = { "E to G" };

char str_CtoB[] = { "C to B" };
char str_CtoD[] = { "C to D" };
char str_CtoE[] = { "C to E" };

char str_HtoA[] = { "H To A" };
char str_HtoF[] = { "H to F" };

char str_GtoA[] = { "G To A" };
char str_GtoE[] = { "G to E" };

char str_1[] = { "1" };
char str_2[] = { "2" };
char str_3[] = { "3" };
char str_5[] = { "5" };

char str_Algo_steps[] = { "Algorithm's Steps :- " };
char str_Algo_steps_1[] = { "1. Create A Graph" };
char str_Algo_steps_2[] = { "2. Create Stack To Get The Next Node To Search Its Adjecent Nodes." };
char str_Algo_steps_3[] = { "3. Now Push The StartNode In The Stack" };
char str_Algo_steps_3_1[] = { "push(stackName, vertex);" };
char str_Algo_steps_4[] = { "4. Now The Traverse The Adjecent Nodes Of The Start Node." };
char str_Algo_steps_5[] = { "5. During Adjecent Nodes Traversal Each Node's LOCAL Value Will Get Calculated." };
char str_Algo_steps_5_1[] = { "Local Value = Current's Local Value + Weight Between Current Node And Adjecent Node." };
char str_Algo_steps_5_2[] = { "if(Local Value < current adjecent node's local value)" };
char str_Algo_steps_5_3[] = { "{" };
char str_Algo_steps_5_4[] = { "Change The Adjecent Node's Current Value And Parent" };
char str_Algo_steps_5_5[] = { "}" };
char str_Algo_steps_5_6[] = { "Now Check The Global Value" };
char str_Algo_steps_5_7[] = { "Global Value = local value + weight(adjecent_node to end_node)" };
char str_Algo_steps_6[] = { "6. During Adjecent Nodes Traversal Each Node Will Get Pushed On To The Stack." };
char str_Algo_steps_7[] = { "7. Do Step-4 To Step-6 For Each Adjecent Node." };
char str_Algo_steps_8[] = { "8. if(All Adjecent Nodes Got Visited)" };
char str_Algo_steps_8_1[] = { "{" };
char str_Algo_steps_8_2[] = { "Pop That Node And SORT The Stack" };
char str_Algo_steps_8_3[] = { "And Goto That Node Which Has Lowest Global Value." };
char str_Algo_steps_8_4[] = { "}" };
char str_Algo_steps_8_5[] = { "Repeat From Step-4 To step-8" };
char str_Algo_steps_9[] = { "9. Now If All Steps Are Done For All Nodes Then Goto End Node And" };
char str_Algo_steps_9_1[] = { "While(node->parent == NULL)" };
char str_Algo_steps_9_2[] = { "{" };
char str_Algo_steps_9_3[] = { "Traverse That Respective Parent Node" };
char str_Algo_steps_9_4[] = { "}" };
char str_Algo_steps_10[] = { "10. Path Found...!!!" };

Edge_t edge[] = { {A, B, 2}, {A, E, 1}, {A, F, 3},{A, G, 1}, {A, H, 2}, {B, C, 1} , {C, D, 2} , {C, E, 1} , {E, D, 5} , {F, D, 1} , {E, F, 1}, {G, E, 2}, {H, F, 2} };
//Edge_t v_edge[] = { {A, D}, {A, B}, {A, E}, {A, F}, {F, A}, {F, D}, {F, E}, {B, A}, {B, C}, {C, B}, {C, E}, {C, D}, {E, A}, {E, D}, {E, C}, {E, F} };

//basic mandetory functions
Ret_t		add_edge_between_vertices(void);
Ret_t		add_vertex_in_graph(void);
Weight_t	get_weight(void);

//drawing functions
void draw_sphere(x_t, y_t);
void draw_cylinder(x_t, y_t, double, float);
void show_nodes(count_t);
void show_edge(void);
Ret_t set_flag(Vertex_t);

//font Redering
void RenderFont(GLfloat x_position, GLfloat y_position, GLfloat z_position, char *str);
void oglRenderFont(GLfloat, GLfloat, char *, GLfloat);
unsigned int Create_Font(char *fontName, int fontSize, float depth);

//Auxillary Routine
void font_init(void);
void topViewPort(void);
Weight_t get_integer_random_value(lower_limit_t, upper_limit_t, count_t);
fWeight_t get_float_random_value(f_lower_limit_t, f_upper_limit_t, count_t);
Ret_t sort_stack(stack_t *);
void Path_Display(ap_list *);
void PrintNode(ap_list *);
void grey_other_nodes(void);
void fonts(void);
Ret_t find_shortest_path(Vertex_t, Vertex_t);

//algorithm functions
Ret_t a_star_algorithm(vNode_t *, vNode_t *);
Ret_t heuristic_cost_estimate(Graph_t * ,Vertex_t, Vertex_t, Weight_t *);

#endif // !


