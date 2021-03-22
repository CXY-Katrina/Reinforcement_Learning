/*
	CSC D84 - Unit 3 - Reinforcement Learning
	
	This file contains stubs for implementing the Q-Learning method
	for reinforcement learning as discussed in lecture. You have to
	complete two versions of Q-Learning.
	
	* Standard Q-Learning, based on a full-state representation and
	  a large Q-Table
	* Feature based Q-Learning to handle problems too big to allow
	  for a full-state representation
	    
	Read the assignment handout carefully, then implement the
	required functions below. Sections where you have to add code
	are marked

	**************
	*** TO DO:
	**************

	If you add any helper functions, make sure you document them
	properly and indicate in the report.txt file what you added.
	
	Have fun!

	DO NOT FORGET TO 'valgrind' YOUR CODE - We will check for pointer
	management being done properly, and for memory leaks.

	Starter code: F.J.E. Jan. 16
*/

#include "QLearn.h"

/* Min Prioirty Queue (Lower values indicate higher priority) 
 * form https://www.geeksforgeeks.org/priority-queue-using-linked-list/ 
 * d - data, p - priority
 */

/* Create A New Node for priority queue (only have to do it once) */
Node* create_node(int d, int c, int p) { 
    Node* temp = (Node*)malloc(sizeof(Node)); 
    temp->data = d; 
	  temp->cost = c;
    temp->priority = p; 
    temp->next = NULL; 
    return temp; 
} 

// Return the value at head 
int peek(Node** head) { 
    return (*head)->data; 
} 

int peek_cost(Node** head) { // note cost != priority
    return (*head)->cost; 
}

int peek_priority(Node** head) { // note cost != priority
    return (*head)->priority; 
}

/* Remove node with highest priority (which is lowest value) */
void pop(Node** head) { 
    Node* temp = *head; 
    (*head) = (*head)->next; 
    free(temp); 
} 

/* Add node to priority queue*/
void push(Node** head, int d, int c, int p) { 
    Node* start = (*head);
    Node* temp = create_node(d, c, p); 

	if (start == NULL) {
		(*head) = temp;
		return;
	}

    if ((*head)->priority > p) {         
        temp->next = *head; 
        (*head) = temp; 
    } else { 
        while (start->next != NULL && start->next->priority <= p) { // TO DO: check this (supposed to add prioritites that tie after)
            start = start->next; 
        } 
        temp->next = start->next; 
        start->next = temp; 
    } 
}

/* Returns boolean value corresponding to empty queue or not*/
int is_empty(Node** head) { 
    return (*head) == NULL; 
} 

int get_priority_for_data(Node** head, int data) {
	Node* start = *head;
	
	// traverse till empty 
	while(!is_empty(&start)) {
		if (start->data == data) {
			return start->priority;
		}
		start = start->next;
	}

	return -1; // case when its not found
}

/* Given a reference (pointer to pointer) to the head of a list 
   and a key, deletes the first occurrence of key in linked list */
void delete_node(Node **head_ref, int key) { 
    Node* temp = *head_ref, *prev; 
  
    if (temp != NULL && temp->data == key) { 
        *head_ref = temp->next;   // Changed head 
        free(temp);               // free old head 
        return; 
    } 
  
    while (temp != NULL && temp->data != key) { 
        prev = temp; 
        temp = temp->next; 
    } 
  
    if (temp == NULL) return; 
  
    prev->next = temp->next; 
    free(temp);  // Free memory 
}

void QLearn_update(int s, int a, double r, int s_new, double *QTable)
{
 /*
   This function implementes the Q-Learning update as stated in Lecture. It 
   receives as input a <s,a,r,s'> tuple, and updates the Q-table accordingly.
   
   Your work here is to calculate the required update for the Q-table entry
   for state s, and apply it to the Q-table
     
   The update involves two constants, alpha and lambda, which are defined in QLearn.h - you should not 
   have to change their values. Use them as they are.
     
   Details on how states are used for indexing into the QTable are shown
   below, in the comments for QLearn_action. Be sure to read those as well!
 */
 
  /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/
  // max_a' Q(s', a')
  double max_Q_new_s_new_a_new = MIN;
  double cur;
  for (int i = 0; i < 4; i++) // loop through action {0, 1, 2, 3}
  {
    cur = *(QTable + (4 * s_new) + i);
    if (cur > max_Q_new_s_new_a_new)
    {
      max_Q_new_s_new_a_new = cur;
    }
  }

  // Q(s, a) += alpha * (r + lmabda * max_a' Q(s', a') - Q(s, a))
  *(QTable + (4 * s) + a) += alpha * (r + lambda * max_Q_new_s_new_a_new - *(QTable + (4 * s) + a));
}

int get_random_action(double gr[max_graph_size][4], int i, int j, int size_X)
{
  // Compute the set of valid actions at current state
  int valid_actions[4] = {-1};
  int number_of_valid_actions = 0;
  int a = -1;

  for (int u = 0; u < 4; u++)
  {
    if (gr[i+(j*size_X)][u] == 1) // no wall
    {
      valid_actions[number_of_valid_actions] = u;
      number_of_valid_actions++;
    }
  }

  double random_number = drand48();
  for (int v = 1; v <= number_of_valid_actions; v++)
  {
    if (random_number < (double) v / (double) number_of_valid_actions)
    {
      a = valid_actions[v - 1];
      break;
    }
  }

  return a;
}

int QLearn_action(double gr[max_graph_size][4], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], double pct, double *QTable, int size_X, int graph_size)
{
  /*
     This function decides the action the mouse will take. It receives as inputs
     - The graph - so you can check for walls! The mouse must never move through walls
     - The mouse position
     - The cat position
     - The chees position
     - A 'pct' value in [0,1] indicating the amount of time the mouse uses the QTable to decide its action,
       for example, if pct=.25, then 25% of the time the mouse uses the QTable to choose its action,
       the remaining 75% of the time it chooses randomly among the available actions.
       
     Remember that the training process involves random exploration initially, but as training
     proceeds we use the QTable more and more, in order to improve our QTable values around promising
     actions.
     
     The value of pct is controlled by QLearn_core_GL, and increases with each round of training.
     
     This function *must return* an action index in [0,3] where
        0 - move up
        1 - move right
        2 - move down
        3 - move left

     QLearn_core_GL will print a warning if your action makes the mouse cross a wall, or if it makes
     the mouse leave the map - this should not happen. If you see a warning, fix the code in this
     function!
     
   The Q-table has been pre-allocated and initialized to 0. The Q-table has
   a size of
   
        graph_size^3 x 4
        
   This is because the table requires one entry for each possible state, and
   the state is comprised of the position of the mouse, cat, and cheese. 
   Since each of these agents can be in one of graph_size positions, all
   possible combinations yield graph_size^3 states.
   
   Now, for each state, the mouse has up to 4 possible moves (up, right,
   down, and left). We ignore here the fact that some moves are not possible
   from some states (due to walls) - it is up to the QLearn_action() function
   to make sure the mouse never crosses a wall. 
   
   So all in all, you have a big table.
        
   For example, on an 8x8 maze, the Q-table will have a size of
   
       64^3 x 4  entries
       
       with 
       
       size_X = 8		<--- size of one side of the maze
       graph_size = 64		<--- Total number of nodes in the graph
       
   Indexing within the Q-table works as follows:
   
     say the mouse is at   i,j
         the cat is at     k,l
         the cheese is at  m,n
         
     state = (i+(j*size_X)) + ((k+(l*size_X))*graph_size) + ((m+(n*size_X))*graph_size*graph_size)
     ** Make sure you undestand the state encoding above!
     
     Entries in the Q-table for this state are

     *(QTable+(4*state)+a)      <-- here a is the action in [0,3]
     
     (yes, it's a linear array, no shorcuts with brackets!)
     
     NOTE: There is only one cat and once cheese, so you only need to use cats[0][:] and cheeses[0][:]
   */
  
  /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/  
  // draw a random number c in [0, 1]
  double c = drand48();
  int i = mouse_pos[0][0];
  int j = mouse_pos[0][1];
  int k = cats[0][0];
  int l = cats[0][1];
  int m = cheeses[0][0];
  int n = cheeses[0][1];
  int state = (i+(j*size_X)) + ((k+(l*size_X))*graph_size) + ((m+(n*size_X))*graph_size*graph_size);
  int a = -1;  // action that the mouse will take
  double optimal_value = MIN;
  double cur_value;

  if (c <= pct)
  {
    // uses the QTable to choose its action
    // choose a to be the current know optimal action in Pi(s)
    for (int u = 0; u < 4; u++)
    {
      if (gr[i+(j*size_X)][u] == 1) // no wall
      {
        cur_value = *(QTable+(4*state)+u);
        if (cur_value > optimal_value)
        {
          optimal_value = cur_value;
          a = u;
        }
      }
    }
  }
  else
  {
    // chooses randomly among the available actions
    // choose a random action a from the set of valid actions at state s
    a = get_random_action(gr, i, j, size_X);
  }

  return a;
}

double QLearn_reward(double gr[max_graph_size][4], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size)
{
  /*
    This function computes and returns a reward for the state represented by the input mouse, cat, and
    cheese position. 
    
    You can make this function as simple or as complex as you like. But it should return positive values
    for states that are favorable to the mouse, and negative values for states that are bad for the 
    mouse.
    
    I am providing you with the graph, in case you want to do some processing on the maze in order to
    decide the reward. 
        
    This function should return a maximim/minimum reward when the mouse eats/gets eaten respectively.      
   */

   /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/ 

  // find distances between mouse and cheeses & mouse and cats using BFS
  double distances_bw_mouse_cats[5] = {-1, -1, -1, -1, -1};
  double distances_bw_mouse_cheeses[5] = {-1, -1, -1, -1, -1};

  BFS(gr, mouse_pos, cats, distances_bw_mouse_cats, 1, graph_size + 1, size_X, graph_size);
  BFS(gr, mouse_pos, cheeses, distances_bw_mouse_cheeses, 1, graph_size + 1, size_X, graph_size);

  double d_cat = distances_bw_mouse_cats[0];
  double d_cheese = distances_bw_mouse_cheeses[0];

  double reward = 0;

  if (d_cat == 0)
  {
    return MIN;
  }
  else if (d_cheese == 0)
  {
    return MAX;
  }
  else
  {
    // reward = (1 / cheese_dist) * 50 - (1 / cat_dist) * 5;
    reward = (1 / d_cheese) * 30 - d_cheese + d_cat;

    // one step away from the cheese
    if (d_cheese <= 1) {
      reward += 50;
    }

    // computer number of walls around mouse
    int num_of_walls = count_wall(gr, mouse_pos[0], size_X);

    // only has one way out
    if (num_of_walls == 3) {
      reward -= 50;
    }
  }

  return reward;
}

void feat_QLearn_update(double gr[max_graph_size][4],double weights[25], double reward, int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size)
{
  /*
    This function performs the Q-learning adjustment to all the weights associated with your
    features. Unlike standard Q-learning, you don't receive a <s,a,r,s'> tuple, instead,
    you receive the current state (mouse, cats, and cheese potisions), and the reward 
    associated with this action (this is called immediately after the mouse makes a move,
    so implicit in this is the mouse having selected some action)
    
    Your code must then evaluate the update and apply it to the weights in the weight array.    
   */
  
   /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/        
  double features[25];
  for (int i = 0; i < 25; i++)
  {
    features[i] = 0;
  }

  double Q_s;
  double maxU;
  int maxA;

  // Q(s)
  evaluateFeatures(gr, features, mouse_pos, cats, cheeses, size_X, graph_size);
  Q_s = Qsa(weights, features);
  // Q(s')
  maxQsa(gr,weights,mouse_pos, cats, cheeses, size_X, graph_size, &maxU, &maxA);

  // update each w_i as follows:
  // wi += alpha * (r + lambda * Q(s') - Q(s)) * f_i(s)
  for (int i = 0; i < numFeatures; i++)
  {
    weights[i] += alpha * (reward + lambda * maxU - Q_s) * features[i];
  }
}

int feat_QLearn_action(double gr[max_graph_size][4],double weights[25], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], double pct, int size_X, int graph_size)
{
  /*
    Similar to its counterpart for standard Q-learning, this function returns the index of the next
    action to be taken by the mouse.
    
    Once more, the 'pct' value controls the percent of time that the function chooses an optimal
    action given the current policy.
    
    E.g. if 'pct' is .15, then 15% of the time the function uses the current weights and chooses
    the optimal action. The remaining 85% of the time, a random action is chosen.
    
    As before, the mouse must never select an action that causes it to walk through walls or leave
    the maze.    
   */

  /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/        

  // draw a random number c in [0, 1]
  double c = drand48();
  int a = -1;  // action that the mouse will take
  double maxU;
  int maxA;

  if (c <= pct)
  {
    // uses the QTable to choose its action
    // choose a to be the current know optimal action in Pi(s)
    maxQsa(gr,weights,mouse_pos, cats, cheeses, size_X, graph_size, &maxU, &maxA);
    a = maxA;
  }
  else
  {
    // chooses randomly among the available actions
    // choose a random action a from the set of valid actions at state s
    a = get_random_action(gr, mouse_pos[0][0], mouse_pos[0][1], size_X);
  }

  return a;
}

void evaluateFeatures(double gr[max_graph_size][4],double features[25], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size)
{
  /*
   This function evaluates all the features you defined for the game configuration given by the input
   mouse, cats, and cheese positions. You are free to define up to 25 features. This function will
   evaluate each, and return all the feature values in the features[] array.
   
   Take some time to think about what features would be useful to have, the better your features, the
   smarter your mouse!
   
   Note that instead of passing down the number of cats and the number of cheese chunks (too many parms!)
   the arrays themselves will tell you what are valid cat/cheese locations.
   
   You can have up to 5 cats and up to 5 cheese chunks, and array entries for the remaining cats/cheese
   will have a value of -1 - check this when evaluating your features!
  */

   /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/      
  
  int num_of_cheeses = 0;
  int num_of_cats = 0;

  for (int i = 0; i < 5; i++) {
    if (cats[i][0] != -1) {
      num_of_cats++;
    }
    if (cheeses[i][0] != -1) {
      num_of_cheeses++;
    }
  }

  double distances_bw_mouse_cats[5] = {-1, -1, -1, -1, -1};
  double distances_bw_mouse_cheeses[5] = {-1, -1, -1, -1, -1};
  int max_depth = graph_size + 1;

  double cat_reach_max_depth = BFS(gr, mouse_pos, cats, distances_bw_mouse_cats, num_of_cats, max_depth, size_X, graph_size);
  double cheese_reach_max_depth = BFS(gr, mouse_pos, cheeses, distances_bw_mouse_cheeses, num_of_cheeses, max_depth, size_X, graph_size);

  double d_cat_closest = MAX;
  double d_cheese_closest = MAX;
  int cat_closest_index = -1;
  int cheese_closest_index = -1;
  double d_cat_average = 0;
  double d_cheese_average = 0;

  if (!cat_reach_max_depth)
  {
    calculate_average_closest_distance(distances_bw_mouse_cats, &cat_closest_index, &d_cat_closest, &d_cat_average);
  }
  else
  {
    Manhatten_distance(mouse_pos, cats, distances_bw_mouse_cats);
    calculate_average_closest_distance(distances_bw_mouse_cats, &cat_closest_index, &d_cat_closest, &d_cat_average);
    if (d_cat_closest < max_depth)
    {
      d_cat_closest = max_depth;
    }
  }
  if (!cheese_reach_max_depth)
  {
    calculate_average_closest_distance(distances_bw_mouse_cheeses, &cheese_closest_index, &d_cheese_closest, &d_cheese_average);
  }
  else
  {
    Manhatten_distance(mouse_pos, cheeses, distances_bw_mouse_cheeses);
    calculate_average_closest_distance(distances_bw_mouse_cheeses, &cheese_closest_index, &d_cheese_closest, &d_cheese_average);
    if (d_cheese_closest < max_depth)
    {
      d_cheese_closest = max_depth;
    }
  }

  int num_of_walls = count_wall(gr, cheeses[cheese_closest_index], size_X);

  features[0] = 1 / (d_cheese_closest + 1);
  features[1] = - 1 / (d_cat_closest + 1);
  // if (num_of_walls == 3 && num_of_cheeses == 1) {
  //   features[2] = size_X;
  // } else if (num_of_walls == 3) {
  //   features[2] = size_X / 3;
  // }
  // else
  // {
  //   features[2] = size_X / (num_of_walls + 1);
  // }

  // features[2] = d_cat_average / graph_size;
}

double Qsa(double weights[25], double features[25])
{
  /*
    Compute and return the Qsa value given the input features and current weights
   */

  /***********************************************************************************************
  * TO DO: Complete this function
  ***********************************************************************************************/  
  
  // Q(s) = w1*f1 + w2*f2 + .. + w_25 * f_25
  double Q_s = 0;
  for (int i = 0; i < numFeatures; i++)
  {
    Q_s += weights[i] * features[i];
  }
  return Q_s;
}

// neighbour_adj_index = 0 means its top neighbour, 1 means right ... til 3 
int get_neighbour_index(int current_index, int neighbour_adj_index, int size_X) {
	int x_cord = index_to_x_cooridinate(current_index, size_X);
	int y_cord = index_to_y_cooridinate(current_index, size_X);

	if (neighbour_adj_index == 0) {
		y_cord = y_cord - 1;
	} else if (neighbour_adj_index == 1) {
		x_cord = x_cord + 1;
	} else if (neighbour_adj_index == 2) {
		y_cord = y_cord + 1;
	} else if (neighbour_adj_index == 3) {
		x_cord = x_cord - 1;
	}
	// no need to check bounds here like x_cord must be < 1024 same w y_cord
	// this function is called after we check the connection bw current node and the neighbour
	return cooridinates_to_index(x_cord, y_cord, size_X);
}

void maxQsa(double gr[max_graph_size][4],double weights[25],int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size, double *maxU, int *maxA)
{
 /*
   Given the state represented by the input positions for mouse, cats, and cheese, this function evaluates
   the Q-value at all possible neighbour states and returns the max. The maximum value is returned in maxU
   and the index of the action corresponding to this value is returned in maxA.
   
   You should make sure the function does not evaluate moves that would make the mouse walk through a
   wall. 
  */
 
   /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/  
  
  double features[25] = {0};
  double max_value = MIN;
  double cur_value;
  int a = -2;
  int mouse_index = mouse_pos[0][0] + mouse_pos[0][1] * size_X;
  int mouse_pos_new[1][2];
  double neighbour_index;
  // double qsa[4] = {MIN};

  for (int i = 0; i < 4; i++)
  {
    if (gr[mouse_index][i] == 1) // no wall
    {
      neighbour_index = get_neighbour_index(mouse_index, i, size_X);
      mouse_pos_new[0][0] = index_to_x_cooridinate(neighbour_index, size_X);
      mouse_pos_new[0][1] = index_to_y_cooridinate(neighbour_index, size_X);
      evaluateFeatures(gr, features, mouse_pos_new, cats, cheeses, size_X, graph_size);
      cur_value = Qsa(weights, features);

      // qsa[i] = cur_value;

      if (cur_value > max_value || a == -2)
      {
        max_value = cur_value;
        a = i;
      }
    }
  }

  *maxU=max_value;
  *maxA=a;
  return;
}

/***************************************************************************************************
 *  Add any functions needed to compute your features below 
 *                 ---->  THIS BOX <-----
 * *************************************************************************************************/

int in_dst_pos(int index, int dst_pos[5][2], int size_X)
{
  int output = -1;
  for (int i = 0; i < 5; i++)
  {
    if (index == cooridinates_to_index(dst_pos[i][0], dst_pos[i][1], size_X))
    {
      output = i;
      break;
    }
  }
  return output;
}

int BFS(double gr[max_graph_size][4], int mouse_pos[1][2], int dst_pos[5][2], double distances[5], int num, int depth, int size_X, int graph_size)
{
  int start_node_index = cooridinates_to_index(mouse_pos[0][0], mouse_pos[0][1], size_X);
  int is_in_dst_pos = -1;
  int found = 0;
  int cur_depth = 0;

  // check mouse pos
  is_in_dst_pos = in_dst_pos(start_node_index, dst_pos, size_X);
  if (is_in_dst_pos > -1)
  {
    distances[is_in_dst_pos] = 0;
    found++;
  }

  // initilaize priority queue (graph index, cost) with start location
	Node* priority_queue = NULL;
  priority_queue = create_node(start_node_index, 0 ,0);

  // initialize an array of graph_size indicating whether a node is visited
  int visited[graph_size] = {0};
  visited[start_node_index] = 1;

  // while priorirty queue is not empty or not every dst_pos has been visited
	while (!is_empty(&priority_queue))
  {
    cur_depth++;

    if (found == num)
    {
      break;
    }

    if (cur_depth == depth)
    {
      return 1;
    }

    // lowest cost node
		int current_lowest_node_index = peek(&priority_queue);
		int current_lowest_node_priority = peek_priority(&priority_queue);
		pop(&priority_queue);

    // check neighbours
    for (int i = 0; i < 4; i++) {
      if (gr[current_lowest_node_index][i] == 1) // neighbour doesnt have wall btwn it
      {
        int neighbour_index = get_neighbour_index(current_lowest_node_index, i, size_X);
        if (!visited[neighbour_index]) // neighbour isnt expanded already
        {
          push(&priority_queue, neighbour_index, 0, current_lowest_node_priority + 1); // we dont use cost for BFS, every edge has cost of 1
          visited[neighbour_index] = 1;

          is_in_dst_pos = in_dst_pos(neighbour_index, dst_pos, size_X);
          if (is_in_dst_pos > -1)
          {
            distances[is_in_dst_pos] = current_lowest_node_priority + 1;
            found++;
          }
        }
      }
		}
  }

  // clean up priority queue
	while (!is_empty(&priority_queue)) {
		pop(&priority_queue); 
	}

  return 0;
}

void Manhatten_distance(int mouse_pos[1][2], int dst_pos[5][2], double distances[5])
{
  for (int i = 0; i < 5; i++)
  {
    distances[i] = abs(mouse_pos[0][0] - dst_pos[i][0]) + abs(mouse_pos[0][1] - dst_pos[i][1]);
  }
}

void calculate_average_closest_distance(double distances[5], int *closest_index, double *d_closest, double *d_average)
{
  double cur_d_closest = MAX;
  double cur_d_average = 0;
  int cur_index_closest = -1;
  int num = 0;

  for (int i = 0; i < 5; i++)
  {
    if (distances[i] > -1)
    {
      num++;
      cur_d_average += distances[i];
      if (distances[i] < cur_d_closest)
      {
        cur_d_closest = distances[i];
        cur_index_closest = i;
      }
    }
  }
  *d_average = cur_d_average / (double) num;
  *d_closest = cur_d_closest;
  *closest_index = cur_index_closest;
}

int count_wall(double gr[max_graph_size][4], int pos[2], int size_X)
{
  int num = 0;
  int index = pos[0] + (pos[1] * size_X);

  for (int i = 0; i < 4; i++) {
    if (gr[index][i] == 0) {
      num++;
    }
  }

  return num;
}
