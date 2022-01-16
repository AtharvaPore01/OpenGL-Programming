#pragma once
//headers
#include<stdio.h>
#include<stdlib.h>

//Macros
#define TRUE				1
#define FALSE				0
#define SUCCESS				1
#define FAILURE				0
#define INVALID_VERTEX		2
#define INVALID_EDGE		3
#define VERTEX_MATCH		4

//new macros
#define MAZE_ROW			28
#define MAZE_COLUMN			31


//structures declaration
struct hNode;
struct vNode;
struct Edge;
struct Graph;
struct position;

//typedefs
typedef struct hNode hNode_t;
typedef struct vNode vNode_t;
typedef struct position pos_t;
typedef hNode_t hList_t;
typedef vNode_t vList_t;
typedef struct Edge Edge_t;
typedef struct Graph Graph_t;
typedef int Vertex_t;
typedef int Ret_t;
typedef int Weight_t;
typedef int Length_t;

//from visualization.h
typedef float x_t;
typedef float y_t;
typedef float z_t;


typedef enum
{
	WHITE = 0,
	GREY,
	BLACK
}Color_t;

/*
typedef enum
{
	W = 1,
	G,
	P,
	M,
	u,
	o,
	e,
	O,
	E,
	F,
	INKY,
	PINKY,
	CLYDE,
	BLINKY
}tile;
*/
//structure

struct hNode
{
	Vertex_t vertex;
	Weight_t W;
	struct hNode *prev;
	struct hNode *next;
	bool bVisited;
};

struct vNode
{
	Vertex_t vertex;
	Vertex_t local;
	Vertex_t global;
	bool bVisited;
	Vertex_t Pred_Vertex;
	hNode_t *ph_Head_Node;
	pos_t *pos;
	Color_t color;
	vNode_t *prev;
	vNode_t *next;
	vNode_t *parent;
};

struct Edge
{
	Vertex_t v_Start;
	Vertex_t v_End;
	Weight_t v_weight;
};

struct Graph
{
	vList_t *pV_Head_Node;
	unsigned int number_of_vertices;
	unsigned int number_of_edges;
};

struct position
{
	x_t _x;
	y_t _y;
};

//Graph Interface Routines
extern "C" Graph_t *CreateGraph(void);
extern "C" Ret_t AddVertex(Graph_t *, x_t, y_t);
extern "C" Ret_t AddEdge(Graph_t *, Vertex_t, Vertex_t, Weight_t);
extern "C" Ret_t RemoveVertex(Graph_t *, Vertex_t);
extern "C" Ret_t RemoveEdge(Graph_t *, Vertex_t, Vertex_t);
extern "C" Ret_t Degree(Graph_t *, Vertex_t, int *);
extern "C" void PrintGraph(Graph_t *);
extern "C" Ret_t DestroyGraph(Graph_t **);

//Graph Auxillary Routines

extern "C" int GetNextVertexNumber(void);

//Horizontal List
//Interface
extern "C" hList_t *hCreateList(void);
extern "C" Ret_t hInsertEnd(hList_t *, Vertex_t, Weight_t);
extern "C" Length_t hLengthOfTheList(hList_t *);
extern "C" Ret_t hDestroyList(hList_t *);

//Aurxillary
extern "C" hNode_t *hGetNode(Vertex_t, Weight_t);
extern "C" void hGenericInsert(hNode_t *, hNode_t *, hNode_t *);
extern "C" void hGenericDelete(hNode_t *);
extern "C" hNode_t *hSearchNode(hList_t *, Vertex_t);

//Vertical List
//Interface
extern "C" vList_t *vCreateList(void);
extern "C" Ret_t vInsertEnd(vList_t *, Vertex_t, x_t, y_t);
extern "C" Length_t vLengthOfTheList(vList_t *);
extern "C" Ret_t vDestroyList(vList_t *);

//Aurxillary
extern "C" vNode_t *vGetNode(Vertex_t, x_t, y_t);
extern "C" void vGenericInsert(vNode_t *, vNode_t *, vNode_t *);
extern "C" void vGenericDelete(vNode_t *);
extern "C" vNode_t *vSearchNode(vList_t *, Vertex_t);

//Auxillary Functions
extern "C" void *xcalloc(size_t, size_t);
