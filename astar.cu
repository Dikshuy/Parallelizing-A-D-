#include<bits/stdc++.h>

#define NODES 800
#define THREADS_BLOCK 512

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


// __global__ costFunction(data* node_dist, int* visited_node, int* global_closest, int num_NODES, ){

// }


__global__ void cudaRelax(data* graph, data* node_dist, int* parent_node, int* visited_node, int* global_closest) {
    int next = blockIdx.x*blockDim.x + threadIdx.x;
    int source = global_closest[0];
    data edge = graph[source*NODES + next];
    data new_dist = node_dist[source] + edge;

    if ((edge != 0) && (visited_node[next] != 1) && (new_dist < node_dist[next])) {
        node_dist[next] = new_dist;
        parent_node[next] = source;
    }
}



int main() {

    srand(42);

    __global__ void nearestNode(data* node_dist, int* visited_node, int* global_closest, int num_NODES);
    __global__ void cudaRelax(data* graph, data* node_dist, int* parent_node, int* visited_node, int* source);

    //declare variables and allocate memory
    __global__ void nearestNode(data* node_dist, int* visited_node, int* global_closest, int num_NODES);
    __global__ void cudaRelax(data* graph, data* node_dist, int* parent_node, int* visited_node, int* source);

    //declare variables and allocate memory
    int graph_size = NODES*NODES*sizeof(data);             //memory in B required by adjacency matrix representation of graph
    int int_array = NODES*sizeof(int);                         //memory in B required by array of vertex IDs. NODES have int IDs.
    int data_array = NODES*sizeof(data);                      //memory in B required by array of vertex distances (depends on type of data used)
    data* graph = (data*)malloc(graph_size);                  //graph itself
    data* node_dist = (data*)malloc(data_array);                  //distances from source indexed by node ID
    int* parent_node = (int*)malloc(int_array);                      //previous nodes on SP indexed by node ID
    int* edge_count = (int*)malloc(int_array);                      //number of edges per node indexed by node ID
    int* visited_node = (int*)malloc(int_array);                      //pseudo-bool if node has been visited indexed by node ID
    int *pn_matrix = (int*)malloc((1)*int_array);    //matrix of parent_node arrays (one per each implementation)
    data* dist_matrix = (data*)malloc((1)*data_array);

    //CUDA mallocs
    data* gpu_graph;
    data* gpu_node_dist;
    int* gpu_parent_node;
    int* gpu_visited_node;
    cudaMalloc((void**)&gpu_graph, graph_size);
    cudaMalloc((void**)&gpu_node_dist, data_array);
    cudaMalloc((void**)&gpu_parent_node, int_array);
    cudaMalloc((void**)&gpu_visited_node, int_array);
    //for closest vertex
    int* closest_vertex = (int*)malloc(sizeof(int));
    int* gpu_closest_vertex;
    closest_vertex[0] = -1;
    cudaMalloc((void**)&gpu_closest_vertex, (sizeof(int)));
    cudaMemcpy(gpu_closest_vertex, closest_vertex, sizeof(int), cudaMemcpyHostToDevice);

    init(graph, NODES);
    edges(graph, edge_count, NODES);

    // int i;                                          //iterator
    int origin = (rand() % NODES);               //starting vertex
    printf("Origin vertex: %d\n", origin);

    int goal = (rand() % NODES);                 // ending vertex
    printf("Goal vertex: %d\n", goal);


    int version = 1;
    cudaEvent_t exec_start, exec_stop;              //timer for execution only
    float elapsed_exec;                             //elapsed time
    cudaEventCreate(&exec_start);
    cudaEventCreate(&exec_stop);

    node_dist[origin] = 0;                                  //start distance is zero; ensures it will be first pulled out

    cudaMemcpy(gpu_graph, graph, graph_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_node_dist, node_dist, data_array, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_parent_node, parent_node, int_array, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_visited_node, visited_node, int_array, cudaMemcpyHostToDevice);

    //Min:  One thread checks for closest vertex. Ideally there would be multiple threads working in
    //  parallel, but due to compiler issues with prallelized-reduction functions this is being used as a backup.
    dim3 gridMin(1, 1, 1);
    dim3 blockMin(1, 1, 1);

    //Relax: Each thread is responsible for relaxing from a shared, given vertex
    //  to one other vertex determined by the ID of the thread. Since each thread handles
    //  a different vertex, there's no RaW or WaR data hazards; all that's needed is a
   dim3 gridRelax(NODES / THREADS_BLOCK, 1, 1);
    dim3 blockRelax(THREADS_BLOCK, 1, 1);

    cudaEventRecord(exec_start);
    for (int i = 0; i < NODES; i++) {
        nearestNode <<<gridMin, blockMin>>>(gpu_node_dist, gpu_visited_node, gpu_closest_vertex, NODES);                 //find min
        cudaRelax <<<gridRelax, blockRelax>>>(gpu_graph, gpu_node_dist, gpu_parent_node, gpu_visited_node, gpu_closest_vertex); //relax
    }
    cudaEventRecord(exec_stop);

    //save data in PN, ND matrices
    cudaMemcpy(node_dist, gpu_node_dist, data_array, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent_node, gpu_parent_node, int_array, cudaMemcpyDeviceToHost);
    cudaMemcpy(visited_node, gpu_visited_node, int_array, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NODES; i++) {                //record resulting parent array and node distance
        pn_matrix[version*NODES + i] = parent_node[i];
        dist_matrix[version*NODES + i] = node_dist[i];
    }

  //free memory
    cudaFree(gpu_graph);
    cudaFree(gpu_node_dist);
    cudaFree(gpu_parent_node);
    cudaFree(gpu_visited_node);

    //calculate elapsed time
    cudaEventElapsedTime(&elapsed_exec, exec_start, exec_stop);        //elapsed execution time
    printf("\n\nCUDA Time (ms): %7.9f\n", elapsed_exec);
}
