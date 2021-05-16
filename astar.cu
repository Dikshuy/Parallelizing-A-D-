#include<bits/stdc++.h>

#define NODES 400  //size of graph will be NODES*NODES     
#define THREADS_BLOCK 256

typedef float data;      

// initial graph with no weights 
void init(data* graph, int num_NODES) {
    int i, j;

    for (i = 0; i < num_NODES; i++) {
        for (j = 0; j < num_NODES; j++) {           
            graph[i*num_NODES + j] = (data)0;
        }
    }
}

// randomly allocating edge weights (g)
void edges(data* graph, int* edge_count, int num_NODES) {
    int nearestNode(data* node_dist, int* visited_node, int num_NODES);

    for (int i = 1; i < num_NODES; i++) {
        int rand_vertex = (rand() % i);
        data weight = (rand() % 500) + 1;             
        graph[rand_vertex*num_NODES + i] = weight;   
        graph[i*num_NODES + rand_vertex] = weight;
        edge_count[i] += 1;                             
        edge_count[rand_vertex] += 1;
    }
}

// finding the minimum using a single thread in a single block 
__global__ void nearestNode(data* node_dist, int* visited_node, int* global_closest, int num_NODES) {
    data dist = 100000;
    int node = -1;
    int i;

    for (i = 0; i < num_NODES; i++) {
        if ((node_dist[i] < dist) && (visited_node[i] != 1)) {
            dist = node_dist[i];
            node = i;
        }
    }

    global_closest[0] = node;
    visited_node[node] = 1;
}

// finding out the heuristic using euclidean distance
int heuristicFunction(int visited_node, int end_node){
    static int xd, yd, d;
    int visited_node_x = visited_node/NODES;
    int visited_node_y = visited_node%NODES;
    int end_node_x = end_node/NODES;
    int end_node_y = end_node%NODES;
    xd = visited_node_x-end_node_x;
    yd = visited_node_y-end_node_y;

    d = static_cast<int>(sqrt(xd*xd+yd*yd));
    return(d);
}


__global__ void cudasync(data* graph, data* node_dist, int* parent_node, int* visited_node, int* global_closest) {
    int next = blockIdx.x*blockDim.x + threadIdx.x;   
    int source = global_closest[0];

    data edge = graph[source*NODES + next];
    data new_dist = node_dist[source] + edge;

    if ((edge != 0) && (visited_node[next] != 1) && (new_dist < node_dist[next])) {
        node_dist[next] = new_dist;
        parent_node[next] = source;
    }
    __syncthreads();
}



int main() {
    srand(42);  

    int size = NODES*NODES*sizeof(data);             
    int int_array = NODES*sizeof(int);                         
    int data_array = NODES*sizeof(data);                      
    data* graph = (data*)malloc(size);                  
    data* node_dist = (data*)malloc(data_array);                  
    int* parent_node = (int*)malloc(int_array);                      
    int* edge_count = (int*)malloc(int_array);                      
    int* visited_node = (int*)malloc(int_array);             
    int* pn_matrix = (int*)malloc(int_array);  
    data* dist_matrix = (data*)malloc(data_array);
    data* cost_function = (data*)malloc(data_array);

    data* gpu_graph;
    data* gpu_node_dist;
    int* gpu_parent_node;
    int* gpu_visited_node;
    cudaMalloc((void**)&gpu_graph, size);
    cudaMalloc((void**)&gpu_node_dist, data_array);
    cudaMalloc((void**)&gpu_parent_node, int_array);
    cudaMalloc((void**)&gpu_visited_node, int_array);

    int* closest_vertex = (int*)malloc(sizeof(int));
    int* gpu_closest_vertex;
    closest_vertex[0] = -1;
    cudaMalloc((void**)&gpu_closest_vertex, (sizeof(int)));
    cudaMemcpy(gpu_closest_vertex, closest_vertex, sizeof(int), cudaMemcpyHostToDevice);

    init(graph, NODES);               
    edges(graph, edge_count, NODES);    

    int origin = (rand() % NODES);               
    printf("Origin vertex: %d\n", origin);

    int goal = (rand() % NODES);                 
    printf("Goal vertex: %d\n", goal);

 
    int version = 1;
    cudaEvent_t start, stop;              
    float elapsed_exec;                             
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    node_dist[origin] = 0;                                

    cudaMemcpy(gpu_graph, graph, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_node_dist, node_dist, data_array, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_parent_node, parent_node, int_array, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_visited_node, visited_node, int_array, cudaMemcpyHostToDevice);

    dim3 gridNearest(1, 1, 1);
    dim3 blockNearest(1, 1, 1);

    dim3 gridRelax(NODES / THREADS_BLOCK, 1, 1);
    dim3 blockRelax(THREADS_BLOCK, 1, 1);           

    cudaEventRecord(start);
    for (int i = 0; i < NODES; i++) {
        nearestNode <<<gridNearest, blockNearest>>>(gpu_node_dist, gpu_visited_node, gpu_closest_vertex, NODES);                 
        cudasync <<<gridRelax, blockRelax>>>(gpu_graph, gpu_node_dist, gpu_parent_node, gpu_visited_node, gpu_closest_vertex); 
    }
    printf("optimal path found!!");
    cudaEventRecord(stop);
    
    cudaMemcpy(node_dist, gpu_node_dist, data_array, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent_node, gpu_parent_node, int_array, cudaMemcpyDeviceToHost);
    cudaMemcpy(visited_node, gpu_visited_node, int_array, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NODES; i++) { 
        int end_dist = heuristicFunction(parent_node[i], goal);             
        pn_matrix[version*NODES + i] = parent_node[i];
        dist_matrix[version*NODES + i] = node_dist[i];
        cost_function[version*NODES + i] = node_dist[i]+end_dist; 
    }

    //free memory
    cudaFree(gpu_graph);
    cudaFree(gpu_node_dist);
    cudaFree(gpu_parent_node);
    cudaFree(gpu_visited_node);

    cudaEventElapsedTime(&elapsed_exec, start, stop);        //elapsed execution time
    printf("\n\nComputational time in CUDA (ms): %7.9f\n", elapsed_exec);
}