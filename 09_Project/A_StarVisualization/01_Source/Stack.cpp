//Header File
#include <stdio.h>
#include <stdlib.h>
#include "ServerList.h"
#include "Stack.h"

extern "C" stack_t *create_stack(void)
{
		//variable
		stack_t		*stack;
		stack = create_list();
		return(stack);
		/*
			//code
			stack = get_node(NULL);
			stack->next = stack;
			stack->prev = stack;

			return(stack);
		*/
}

extern "C" ap_ret push(stack_t *stack, ap_data data)
{
		//variable
		ap_ret	ret;

		//code
		ret = insert_end(stack, data);
		return(ret);
		/*
			p_node = get_node(data);
			generalised_insert(stack->prev, p_node, stack);
			return(SUCCESS);
		*/
}

extern "C" ap_ret top(stack_t *stack, ap_data *p_data)
{
		//variable
		ap_ret	ret;

		//code
		ret = examine_beginning(stack, p_data);
		return(ret);
}

extern "C" ap_ret pop(stack_t *stack, ap_data *p_data)
{
		//variable
		ap_ret ret;

		//code
		ret = examine_and_delete_beginning(stack, p_data);
		return(ret);
		/*
			if (stack->next == stack && stack->prev == stack)
			{
				return(STACK_EMPTY);
			}
			*p_data = stack->prev->data;
			generalised_delete(stack->prev);
			return(SUCCESS);
		*/
}

extern "C" bool is_stack_empty(stack_t *stack)
{
		return(is_empty(stack));
}

extern "C" ap_ret destroy_stack(stack_t **pp_stack)
{
		//variable
		ap_ret ret;

		//code
		ret = destroy(pp_stack);
		return(ret);
}

