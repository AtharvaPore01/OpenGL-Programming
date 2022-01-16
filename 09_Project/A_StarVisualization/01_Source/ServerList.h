#ifndef _LIST_H 
#define _LIST_H 

#define TRUE			1
#define FALSE			0
#define SUCCESS			1
#define FAILURE			0
#define DATA_NOT_FOUND	1
#define LIST_EMPTY		2

#define	MAP_ROW		28
#define	MAP_COL		31

struct node;
struct a_star;
typedef struct node ap_node;
typedef ap_node ap_list;
typedef int ap_data;
typedef int ap_ret;
typedef int ap_len;

/* Internal Layout */
struct node
{
	ap_data data;
	struct node *prev;
	struct node *next;
};



/* Insterface Functions */

ap_list *create_list(void);

ap_ret insert_beginning(ap_list *, ap_data);
ap_ret insert_end(ap_list *, ap_data);
ap_ret insert_after_data(ap_list *, ap_data, ap_data);
ap_ret insert_before_data(ap_list *, ap_data, ap_data);

ap_ret delete_beginning(ap_list *);
ap_ret delete_end(ap_list *);
ap_ret delete_data(ap_list *, ap_data);

bool find(ap_list *, ap_data);
bool is_empty(ap_list *);
ap_len len(ap_list *);
void Display(ap_list *);

ap_ret examine_beginning(ap_list *, ap_data *);
ap_ret examine_end(ap_list *, ap_data *);
ap_ret examine_and_delete_beginning(ap_list *, ap_data *);
ap_ret examine_and_delete_end(ap_list *, ap_data *);
ap_data *to_array(ap_list *, ap_len *);
ap_ret sort(int a[], size_t);
ap_ret destroy(ap_list **);

/*List Auxillary Functions*/
static void generalised_insert(ap_node *, ap_node *, ap_node *);
static void generalised_delete(ap_node *);
static ap_node *search_node(ap_list *, ap_data);
static ap_node *get_node(ap_data);

/* Auxillary Functions */
static void *xcalloc(size_t, size_t);

#endif /* _LIST_H */ 
