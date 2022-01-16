#include<stdio.h>
#include<stdlib.h>
#include"ServerList.h"

/* Insterface Functions */

ap_list *create_list(void)
{
	ap_list *head_node = get_node(0);
	head_node->next = head_node;
	head_node->prev = head_node;
	return(head_node);
}

ap_ret insert_beginning(ap_list *pList, ap_data new_data)
{
	ap_node *new_node = get_node(new_data);
	generalised_insert(pList, new_node, pList->next);
	return(SUCCESS);
}

ap_ret insert_end(ap_list *pList, ap_data new_data)
{
	ap_node *new_node = get_node(new_data);
	generalised_insert(pList->prev, new_node, pList);
	return(SUCCESS);
}

ap_ret insert_after_data(ap_list *pList, ap_data existing_data, ap_data new_data)
{

	ap_node *existing_node = NULL;
	existing_node = search_node(pList, existing_data);
	if(existing_node == NULL)
	{
		return(DATA_NOT_FOUND);
	}
	ap_node *new_node = get_node(new_data);
	generalised_insert(existing_node, new_node, existing_node->next);
	return(SUCCESS);
}
ap_ret insert_before_data(ap_list *pList, ap_data existing_data, ap_data new_data)
{
	ap_node *existing_node = NULL;
	existing_node = search_node(pList, existing_data);
	if(existing_node == NULL)
	{
		return(DATA_NOT_FOUND);
	}
	ap_node *new_node = get_node(new_data);
	generalised_insert(existing_node->prev, new_node, existing_node);
	return(SUCCESS);
}

ap_ret delete_beginning(ap_list *pList)
{
	if(is_empty(pList))
	{
		return(LIST_EMPTY);
	}
	generalised_delete(pList->next);
	return(SUCCESS);	
}

ap_ret delete_end(ap_list *pList)
{
	if(is_empty(pList))
	{
		return(LIST_EMPTY);
	}

	generalised_delete(pList->prev);
	return(SUCCESS);	
}

ap_ret delete_data(ap_list *pList, ap_data existing_data)
{
	ap_node *existing_node = NULL;
	existing_node = search_node(pList, existing_data);
	if(existing_node == NULL)
	{
		return(DATA_NOT_FOUND);
	}
	generalised_delete(existing_node);
	return(SUCCESS);
}

bool find(ap_list *pList, ap_data find_data)
{
	return(search_node(pList, find_data) != NULL);
}	

bool is_empty(ap_list *pList)
{
	return(pList->prev == pList && pList->next == pList);
}

ap_len len(ap_list *pList)
{
	ap_node *run = NULL;
	ap_len len = 0;
	run = pList->next;
	while(run != pList)
	{
		len = len + 1;
		run = run->next;
	}
	return(len);
}

void Display(ap_list *pList)
{
	ap_node *run;
	printf("[Beginning]<->");
	run = pList->next;
	while(run != pList)
	{
		printf("[%d]<->", run->data);
		run = run->next;
	} 
	printf("[End]\n");
}

ap_ret examine_beginning(ap_list *pList, ap_data *pData)
{
	if(is_empty(pList))
	{	
		return(LIST_EMPTY);
	}
	*pData = pList->next->data;
	return(SUCCESS);
}

ap_ret examine_end(ap_list *pList, ap_data *pData)
{
	if(is_empty(pList))
	{	
		return(LIST_EMPTY);
	}
	*pData = pList->prev->data;
	return(SUCCESS);
}

ap_ret examine_and_delete_beginning(ap_list *pList, ap_data *pData)
{
	if(is_empty(pList))
	{	
		return(LIST_EMPTY);
	}
	*pData = pList->next->data;
	generalised_delete(pList->next);
	return(SUCCESS);	
}
ap_ret examine_and_delete_end(ap_list *pList, ap_data *pData)
{
	if(is_empty(pList))
	{	
		return(LIST_EMPTY);
	}
	*pData = pList->prev->data;
	generalised_delete(pList->prev);
	return(SUCCESS);	
}

ap_data *to_array(ap_list *lst, ap_len *p_len)
{
	ap_len lst_len = len(lst);
	ap_data *arr = NULL;
	ap_node *run = NULL;
	int i = 0;

	if (lst_len <= 0)
		return (NULL);

	arr = (ap_data*)xcalloc(lst_len, sizeof(ap_data));

	for (run = lst->next, i = 0; run != lst; run = run->next, ++i)
		arr[i] = run->data;

	*p_len = lst_len;
	return (arr);
}

ap_ret sort(int a[], size_t n)
{
	int i, j, key;

	for (j = 1; j < n; j++)
	{
		key = a[j];
		i = j - 1;
		while (i > -1 && a[i] > key)
		{
			a[i + 1] = a[i];
			i = i - 1;
		}
		a[i + 1] = key;
	}
	return(SUCCESS);
}

ap_ret destroy(ap_list **pList)
{
	ap_node *head_node = NULL;
	ap_node *run = NULL;
	ap_node *run_n = NULL;

	head_node = *pList;
	run = head_node->next;

	while(run != head_node)
	{
		run_n = run->next;
		free(run);
		run = run_n;
	}

	free(head_node);
	*pList = NULL;
	return(SUCCESS);	
}

/*List Auxillary Functions*/
static void generalised_insert(ap_node *pBeg, ap_node *pMid, ap_node *pEnd)
{
	pMid->next = pEnd;
	pMid->prev = pBeg;
	pBeg->next = pMid;
	pEnd->prev = pMid;
}

static void generalised_delete(ap_node *pNode)
{
	pNode->next->prev = pNode->prev;
	pNode->prev->next = pNode->next;
	free(pNode);
	pNode = NULL;
}


static ap_node *search_node(ap_list *pList, ap_data search_data)
{
	ap_node *run = NULL;
	run = pList->next;
	while(run != pList)
	{
		if(run->data == search_data)
		{
			return(run);
		}
		run = run->next;
	}
	return(NULL);
}
static ap_node *get_node(ap_data data)
{
	ap_node *new_node = (ap_node *)xcalloc(1, sizeof(ap_node));
	new_node->data = data;
	return(new_node);
}

/* Auxillary Functions */
static void *xcalloc(size_t units, size_t size_per_units)
{
	void *p = calloc(units, size_per_units);
	return(p);
}
