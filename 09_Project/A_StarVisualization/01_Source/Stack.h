#pragma once
#ifndef _STACK_H

//Headers
#include "ServerList.h"

//typedef
typedef ap_list stack_t;
typedef struct a_star a_star_t;
typedef struct a_star array_t;

//Structure
struct a_star
{
	bool			bVisited = false;
	bool			bObstacle = false;
	float			global_goal;
	float			local_goal;
	int				x;
	int				y;
	struct a_star	*parent;
};

//macros
#define STACK_EMPTY LIST_EMPTY 

//function prototype
extern "C" stack_t *create_stack(void);
extern "C" ap_ret push(stack_t *stack, ap_data data);
extern "C" ap_ret top(stack_t *stack, ap_data *p_data);
extern "C" ap_ret pop(stack_t *stack, ap_data *p_data);
extern "C" bool is_stack_empty(stack_t *stack);
extern "C" ap_ret destroy_stack(stack_t **pp_stack);

#endif // !_STACK_H
