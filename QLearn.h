/*
	CSC D84 - Unit 3 - Reinforcement Learning

	This file contains the API function headers for your assignment.
	Please pay close attention to the function prototypes, and
	understand what the arguments are.

	Stubs for implementing each function are to be found in QLearn.c,
	along with clear ** TO DO markers to let you know where to add code.

	You are free to add helper functions within reason. But you must
	provide a prototype *in this file* as well as the implementation
	in the .c program file.

	Starter by: F.J.E., Jan. 2016
*/

#ifndef __QLearn_header

#define __QLearn_header

// Generally needed includes
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<malloc.h>
#include<string.h>

#define alpha .01			// Learning rate for standard Q-Learning
#define lambda .5			// Discount rate for future rewards
#define max_graph_size 32*32

#define numFeatures 2			// UPDATE THIS to be the number of features you have

#define MIN -1000000
#define MAX 1000000

// Function prototypes for D84 - Unit 3 - Reinforcement Learning
void QLearn_update(int s, int a, double r, int s_new, double *QTable);
int QLearn_action(double gr[max_graph_size][4], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], double pct, double *QTable, int size_X, int graph_size);
double QLearn_reward(double gr[max_graph_size][4], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size);

void feat_QLearn_update(double gr[max_graph_size][4],double weights[25], double reward, int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size);
int feat_QLearn_action(double gr[max_graph_size][4],double weights[25], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], double pct, int size_X, int graph_size);
void evaluateFeatures(double gr[max_graph_size][4],double features[25], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size);
double Qsa(double weights[25], double features[25]);
void maxQsa(double gr[max_graph_size][4],double weights[25],int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size, double *maxU, int *maxA);

void expensiveFeature1(double gr[max_graph_size][4], int path[max_graph_size][2],int start_x, int start_y, int (*goalCheck)(int x, int y, int pos[5][2]), int pos[5][2], int s_type, int *l, int size_X);
int checkForGoal(int x, int y, int pos[5][2]);

// If you need to add any function prototypes yourself, you can do so *below* this line.
typedef struct node { 
    int data; 
	int cost; // actual cost to get to that node - g(n)
    int priority; // f(n) = g(n) + h(n)
    struct node* next; 
} Node; 

int inline cooridinates_to_index(int x, int y, int size_X) {
	return x + (y * size_X);
}

int inline index_to_x_cooridinate(int index, int size_X) {
	return index % size_X;
}

int inline index_to_y_cooridinate(int index, int size_X) {
	return index / size_X;
}

int get_random_action(double gr[max_graph_size][4], int i, int j, int size_X);
int get_neighbour_index(int current_index, int neighbour_adj_index, int size_X);
int in_dst_pos(int index, int dst_pos[5][2], int size_X);
int BFS(double gr[max_graph_size][4], int mouse_pos[1][2], int dst_pos[5][2], double distances[5], int num, int depth, int size_X, int graph_size);
void Manhatten_distance(int mouse_pos[1][2], int dst_pos[5][2], double distances[5]);
void calculate_average_closest_distance(double distances[5], int *closest_index, double *d_closest, double *d_average);
int count_wall(double gr[max_graph_size][4], int pos[2], int size_X);
#endif

