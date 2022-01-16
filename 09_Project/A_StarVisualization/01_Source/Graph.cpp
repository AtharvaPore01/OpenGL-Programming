//Headers
#include<Windows.h>
#include "Graph.h"

//Graph Interface Routines
extern "C" Graph_t *CreateGraph(void)
{
	//variable
	Graph_t *pGraph = NULL;

	//code
	pGraph = (Graph_t *)xcalloc(1, sizeof(Graph_t));

	pGraph->pV_Head_Node = vCreateList();
	pGraph->number_of_vertices = 0;
	pGraph->number_of_edges = 0;

	return(pGraph);
}

extern "C" Ret_t AddVertex(Graph_t *graph, x_t x, y_t y)
{
	//variable declaration and initialization
	Vertex_t Current_Vertex_Number = -1;

	Current_Vertex_Number = GetNextVertexNumber();
	vInsertEnd(graph->pV_Head_Node, Current_Vertex_Number, x, y);
	graph->number_of_vertices = graph->number_of_vertices + 1;
	return(SUCCESS);
}
extern "C" Ret_t AddEdge(Graph_t *graph, Vertex_t start, Vertex_t end, Weight_t weight)
{
	//variable
	vNode_t *vStartNode = NULL;
	vNode_t *vEndNode = NULL;
	Ret_t ret;

	vStartNode = vSearchNode(graph->pV_Head_Node, start);
	if (vStartNode == NULL)
	{
		return(INVALID_VERTEX);
	}

	vEndNode = vSearchNode(graph->pV_Head_Node, end);
	if (vEndNode == NULL)
	{
		return(INVALID_VERTEX);
	}

	ret = hInsertEnd(vStartNode->ph_Head_Node, end, weight);
	if (ret != SUCCESS)
	{
		fprintf_s(stderr, "Add Edge:Unexpected Error\n");
		exit(EXIT_FAILURE);
	}
	ret = hInsertEnd(vEndNode->ph_Head_Node, start, weight);
	if (ret != SUCCESS)
	{
		fprintf_s(stderr, "Add Edge:Unexpected Error\n");
		exit(EXIT_FAILURE);
	}

	return(SUCCESS);
}
extern "C" Ret_t RemoveVertex(Graph_t *graph, Vertex_t vertex)
{
	//variables
	vNode_t *vRemoveNode = NULL;
	vNode_t * vAdjecentNode = NULL;

	hNode_t *hRemoveAdjecentNode = NULL;
	hNode_t *hRun = NULL;
	hNode_t *hRunNext = NULL;

	//code
	vRemoveNode = vSearchNode(graph->pV_Head_Node, vertex);

	if (vRemoveNode == NULL)
	{
		return(INVALID_VERTEX);
	}
	hRun = vRemoveNode->ph_Head_Node->next;

	while (hRun != vRemoveNode->ph_Head_Node)
	{
		hRunNext = hRun->next;
		vAdjecentNode = vSearchNode(graph->pV_Head_Node, hRun->vertex);

		if (vAdjecentNode == NULL)
		{
			fprintf_s(stderr, "RemoveVertex: unexpected error\n");
			exit(EXIT_FAILURE);
		}

		hRemoveAdjecentNode = hSearchNode(vAdjecentNode->ph_Head_Node, vertex);
		if (hRemoveAdjecentNode == NULL)
		{
			fprintf_s(stderr, "RemoveVertex: unexpected error\n");
			exit(EXIT_FAILURE);
		}

		hGenericDelete(hRemoveAdjecentNode);
		hGenericDelete(hRun);
		hRun = hRunNext;
	}

	vGenericDelete(vRemoveNode);
	return(SUCCESS);
}
extern "C" Ret_t RemoveEdge(Graph_t *graph, Vertex_t start, Vertex_t end)
{
	//variable
	vNode_t *vStartNode = NULL;
	vNode_t *vEndNode = NULL;
	hNode_t *hStartInTheEndNode = NULL;
	hNode_t *hEndInTheStartNode = NULL;

	//code
	vStartNode = vSearchNode(graph->pV_Head_Node, start);
	if (vStartNode == NULL)
	{
		return(INVALID_VERTEX);
	}

	vEndNode = vSearchNode(graph->pV_Head_Node, end);
	if (vEndNode == NULL)
	{
		return(INVALID_VERTEX);
	}

	hStartInTheEndNode = hSearchNode(vEndNode->ph_Head_Node, start);
	if (hStartInTheEndNode == NULL)
	{
		return(INVALID_EDGE);
	}

	hEndInTheStartNode = hSearchNode(vStartNode->ph_Head_Node, end);
	if (hEndInTheStartNode == NULL)
	{
		return(INVALID_EDGE);
	}

	hGenericDelete(hStartInTheEndNode);
	hGenericDelete(hEndInTheStartNode);

	return(SUCCESS);
}

extern "C" Ret_t Degree(Graph_t *graph, Vertex_t vertex, int *degree)
{
	//variable
	vNode_t *pNode = NULL;
	hNode_t *pRun = NULL;
	Length_t hLength = -1;

	//code
	pNode = vSearchNode(graph->pV_Head_Node, vertex);
	if (pNode == NULL)
	{
		return(INVALID_VERTEX);
	}

	pRun = pNode->ph_Head_Node->next;

	hLength = 0;

	while (pRun != pNode->ph_Head_Node)
	{
		hLength = hLength + 1;
		pRun = pRun->next;
	}

	*degree = hLength;

	return(SUCCESS);
}

//extern "C" void PrintGraph(Graph_t *g)
//{
//	//variable
//	vNode_t *pv_run = NULL;
//	hNode_t *ph_run = NULL;
//	for (pv_run = g->pV_Head_Node->next; pv_run != g->pV_Head_Node; pv_run = pv_run->next)
//	{
//		fprintf_s(gpFile, "[%d]:\t\t", pv_run->vertex);
//		for (ph_run = pv_run->ph_Head_Node->next; ph_run != pv_run->ph_Head_Node; ph_run = ph_run->next)
//			fprintf_s(gpFile, "[%d]<->", ph_run->vertex);
//		fprintf_s(gpFile,"[end]");
//	}
//}

extern "C" Ret_t DestroyGraph(Graph_t **graph)
{
	//variable
	Graph_t *pGraph = NULL;
	vNode_t *vRun = NULL;
	hNode_t *hRun = NULL;
	vNode_t *vRunNext = NULL;

	//code
	pGraph = *graph;

	vRun = pGraph->pV_Head_Node->next;

	while (vRun != pGraph->pV_Head_Node)
	{
		if (hDestroyList(vRun->ph_Head_Node) != SUCCESS)
		{
			fprintf(stderr, "destroy_graph: unexpected error\n");
			exit(EXIT_FAILURE);
		}

		vRunNext = vRun->next;
		free(vRun);

		vRun = vRunNext;
	}

	free(pGraph->pV_Head_Node);
	free(pGraph);
	pGraph = NULL;
	*graph = NULL;

	return(SUCCESS);
}

//Graph Auxillary Routines
extern "C" int GetNextVertexNumber(void)
{
	//variable
	static int Next_Vertex_Number = -1;

	//code
	return(++Next_Vertex_Number);
}

//Horizontal List
//Interface
extern "C" hList_t *hCreateList(void)
{
	hNode_t *hHeadNode = NULL;

	hHeadNode = (hNode_t *)xcalloc(1, sizeof(hNode_t));
	hHeadNode->prev = hHeadNode;
	hHeadNode->next = hHeadNode;

	return(hHeadNode);
}

extern "C" Ret_t hInsertEnd(hList_t *hList, Vertex_t vertex, Weight_t weight)
{
	//variable
	hNode_t *hNewNode = NULL;

	//code
	hNewNode = hGetNode(vertex, weight);
	hGenericInsert(hList->prev, hNewNode, hList);

	return(SUCCESS);
}

extern "C" Length_t hLengthOfTheList(hList_t *list)
{
	//variable
	hNode_t *hRun = NULL;
	Length_t length = -1;

	//code
	hRun = list->next;
	while (hRun != list)
	{
		length = length + 1;
		hRun = hRun->next;
	}

	return(length);
}

extern "C" Ret_t hDestroyList(hList_t *pList)
{
	//variable
	hNode_t *hRun = NULL;
	hNode_t *hRunNext = NULL;

	//code
	hRun = pList->next;
	while (hRun != pList)
	{
		hRunNext = hRun->next;
		free(hRun);
		hRun = hRunNext;
	}

	free(pList);
	pList = NULL;

	return(SUCCESS);
}

//Aurxillary
extern "C" hNode_t *hGetNode(Vertex_t vertex, Weight_t weight)
{
	//variable
	hNode_t * pNode = NULL;

	//code
	pNode = (hNode_t *)xcalloc(1, sizeof(hNode_t));
	pNode->W = weight;
	pNode->vertex = vertex;

	return(pNode);
}

extern "C" void hGenericInsert(hNode_t *pBeg, hNode_t *pMid, hNode_t *pEnd)
{
	pMid->next = pEnd;
	pMid->prev = pBeg;
	pBeg->next = pMid;
	pEnd->prev = pMid;
}

extern "C" void hGenericDelete(hNode_t *pNode)
{
	//code
	pNode->next->prev = pNode->prev;
	pNode->prev->next = pNode->next;

	free(pNode);
	pNode = NULL;
}

extern "C" hNode_t *hSearchNode(hList_t *hList, Vertex_t vertex)
{
	//variable
	hNode_t *hRun = NULL;

	//code
	hRun = hList->next;
	while (hRun != hList)
	{
		if (hRun->vertex == vertex)
		{
			return(hRun);
		}
		hRun = hRun->next;
	}

	return((hNode *)NULL);
}

//Vertical List
//Interface
extern "C" vList_t *vCreateList(void)
{
	//variable
	vNode_t *vHeadNode = NULL;

	//code
	vHeadNode = vGetNode(-1, 0, 0);
	vHeadNode->prev = vHeadNode;
	vHeadNode->next = vHeadNode;

	return(vHeadNode);

}

extern "C" Ret_t vInsertEnd(vList_t *list, Vertex_t vertex, x_t x, y_t y)
{
	//variable
	vNode_t *pNewNode = NULL;

	//code
	pNewNode = vGetNode(vertex, x, y);
	vGenericInsert(list->prev, pNewNode, list);

	return(SUCCESS);
}

extern "C" Length_t vLengthOfTheList(vList_t *list)
{
	//variable
	vNode_t *vRun = NULL;
	Length_t length = -1;

	//code
	vRun = list->next;
	while (vRun != list)
	{
		length = length + 1;
		vRun = vRun->next;
	}
	return(length);
}

extern "C" Ret_t vDestroyList(vList_t *vList)
{
	//variable
	vNode_t *vRun = NULL;
	vNode_t *vRunNext = NULL;

	//code
	vRun = vList->next;

	while (vRun != vList)
	{
		vRunNext = vRun->next;

		free(vRun);
		vRun = vRunNext;
	}

	free(vList);
	vList = NULL;

	return(SUCCESS);
}

//Aurxillary
extern "C" vNode_t *vGetNode(Vertex_t vertex, x_t x, y_t y)
{
	//variable
	vNode_t *NewNode = NULL;

	//code
	NewNode = (vNode_t *)xcalloc(1, sizeof(vNode_t));
	NewNode->vertex = vertex;
	NewNode->Pred_Vertex = -1;
	NewNode->local = INT_MAX;
	NewNode->global = INT_MAX;
	//NewNode->pos->_x = 0;
	//NewNode->pos->_y = 0;
	NewNode->ph_Head_Node = hCreateList();
	NewNode->color = WHITE;
	NewNode->parent = NULL;
	NewNode->bVisited = false;

	return(NewNode);
}

extern "C" void vGenericInsert(vNode_t *pBeg, vNode_t *pMid, vNode_t *pEnd)
{
	//code
	pBeg->next = pMid;
	pEnd->prev = pMid;
	pMid->prev = pBeg;
	pMid->next = pEnd;

}

extern "C" void vGenericDelete(vNode_t *pNode)
{
	//code
	pNode->next->prev = pNode->prev;
	pNode->prev->next = pNode->next;

	free(pNode);
	pNode = NULL;
}

extern "C" vNode_t *vSearchNode(vList_t *pList, Vertex_t vertex)
{
	//variable
	vNode_t *vRun = NULL;

	//code
	vRun = pList->next;

	while (vRun != pList)
	{
		if (vRun->vertex == vertex)
		{
			return(vRun);
		}
		vRun = vRun->next;
	}

	return((vNode_t *)NULL);
}

//Auxillary Functions
extern "C" void *xcalloc(size_t n, size_t size)
{
	void *p = NULL;

	p = calloc(n, size);
	if (p == NULL)
	{
		fprintf_s(stderr, "calloc :fatal:Memory Not Allocated\n");
		exit(EXIT_FAILURE);
	}
	return(p);
}
