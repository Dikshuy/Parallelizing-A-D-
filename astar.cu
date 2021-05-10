#include<bits/stdc++.h> 
using namespace std; 

#define ROW 200
#define COL 200

#define THREADS_PER_BLOCK 8

typedef pair<int, int> Pair; 
typedef pair<double, pair<int, int>> pPair; 

struct cell {
	int parent_i, parent_j; 
	// f = g + h 
	double f, g, h; 
}; 

bool isValid(int row, int col, int H, int W) { 
	// returns true if row number and column number is in range 
	return (row >= 0) && (row < H) && (col >= 0) && (col < W); 
} 

__global__ void isValid( int *row, int *col, int *H, int *W, bool *answer)
{ 
	// returns true if row number and column number is in range 
	answer = (row >= 0) && (row < H) && (col >= 0) && (col < W); 
} 

// 0 represents occupied cell, 1 represents free cell and 2 represents the cell in the optimal path
bool isUnBlocked(int grid[][COL], int row, int col) { 
	// returns true if the cell is not blocked else false 
	if (grid[row][col] == 1 || grid[row][col] == 2) 
		return (true); 
	else
		return (false); 
} 

__global__ void isUnBlocked(int *grid[][COL], int *row, int *col, bool *answer) { 
	// returns true if the cell is not blocked else false 
	if (grid[row][col] == 1 || grid[row][col] == 2)
	{ 
		answer = true; 
	}
	else
	{
		answer = false; 
	}
}

bool isDestination(int row, int col, Pair dest) { 
	if (row == dest.first && col == dest.second) 
		return (true); 
	else
		return (false); 
} 

__global__ isDestination(int *row, int *col, Pair *dest, bool *answer) { 
	if (row == dest.first && col == dest.second) 
		answer = true; 
	else
		answer = false; 
} 
 
double calculateHValue(int row, int col, Pair dest) {  
	return ((double)sqrt ((row-dest.first)*(row-dest.first) + (col-dest.second)*(col-dest.second))); 
} 

__global__ calculateHValue(int row, int col, Pair dest) {  
	answer =  ((double)sqrt ((row-dest.first)*(row-dest.first) + (col-dest.second)*(col-dest.second))); 
} 

__global__ void tracePath(cell *cellDetails[][COL], Pair *dest, vector<Pair> *solvec) { 

	int row = dest.first; 
	int col = dest.second; 

	stack<Pair> Path; 

	while (!(cellDetails[row][col].parent_i == row && cellDetails[row][col].parent_j == col )) { 
		Path.push(make_pair (row, col)); 
		int temp_row = cellDetails[row][col].parent_i; 
		int temp_col = cellDetails[row][col].parent_j; 
		row = temp_row; 
		col = temp_col; 
	} 

	Path.push (make_pair (row, col)); 
	while (!Path.empty()) 
	{ 
		Pair p = Path.top(); 
		Path.pop();
		solvec->push_back(p);
	} 
} 

__global__ void Do(*parent_i,*parent_j, *I, *J, *dest, *cellDetails, *sol, *closedList, *H, *W)

{

int i = I[threadIdx.x]
int j = I[threadIdx.x]
double gNew, hNew, fNew;

isValid<<<1,1>>> (i, j, H, W, ans1);
isDestination<<<1,1>>> (i, j, dest,ans2);
isUnBlocked<<<1,1>>> (grid, i, j, ans3);

if (ans1 == true) {
			if (ans2 == true) {  
				cellDetails[i][j].parent_i = parent_i; 
				cellDetails[i][j].parent_j = parent_j;  
				tracePath (cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			} 
			else if (closedList[i][j] == false &&  ans3== true) 
			{ 
				gNew = cellDetails[i][j].g + 1.0; 
				calculateHValue<<<1,1>>> (i, j, dest, hNew); 
				fNew = gNew + hNew;  
				if (cellDetails[i][j].f == FLT_MAX || cellDetails[i][j].f > fNew) 
				{ 
					openList.insert( make_pair(fNew, make_pair(i-1, j)));
					cellDetails[i][j].f = fNew; 
					cellDetails[i][j].g = gNew; 
					cellDetails[i][j].h = hNew; 
					cellDetails[i][j].parent_i = i; 
					cellDetails[i][j].parent_j = j; 
				} 
			} 
		} 
}

void aStarSearch(int grid[][COL], int H, int W, Pair src, Pair dest, vector<Pair> *sol) {

	if (isValid (src.first, src.second, H, W) == false) { 
		printf ("Source is invalid"); 
		return; 
	} 
	if (isValid (dest.first, dest.second, H, W) == false) { 
		printf ("Destination is invalid"); 
		return; 
	} 

	// blocking check
	if (isUnBlocked(grid, src.first, src.second) == false) { 
		printf ("Source is blocked"); 
		return; 
	} 
	if (isUnBlocked(grid, dest.first, dest.second) == false) { 
		printf ("Destination is blocked"); 
		return; 
	} 

	// if the destination cell is the same as source cell 
	if (isDestination(src.first, src.second, dest) == true) { 
		printf ("We are already at the destination"); 
		return; 
	} 
	
	bool closedList[ROW][COL], *dev_closedList; 
	memset(closedList, false, sizeof (closedList)); 
 
	cell cellDetails[ROW][COL], *dev_cellDetails; 

	int i, j, *parent_i, *parent_j; 

	for (i=0; i<ROW; i++) { 
		for (j=0; j<COL; j++) { 
			cellDetails[i][j].f = FLT_MAX; 
			cellDetails[i][j].g = FLT_MAX; 
			cellDetails[i][j].h = FLT_MAX; 
			cellDetails[i][j].parent_i = -1; 
			cellDetails[i][j].parent_j = -1; 
		} 
	} 

	i = src.first, j = src.second; 
	cellDetails[i][j].f = 0.0; 
	cellDetails[i][j].g = 0.0; 
	cellDetails[i][j].h = 0.0; 
	cellDetails[i][j].parent_i = i; 
	cellDetails[i][j].parent_j = j; 

	set<pPair> openList; 
	openList.insert(make_pair (0.0, make_pair (i, j)));
	bool foundDest = false; 

	while (!openList.empty()) 

	{ 
		pPair p = *openList.begin(); 
		openList.erase(openList.begin()); 
		i = p.second.first; 
		j = p.second.second; 
		closedList[i][j] = true; 
	
	/* 
		Generating all the 8 successor of this cell 
			N.W   N   N.E 
			  \   |   / 
			   \  |  / 
			W----Cell----E 
			   /  |  \ 
			  /   |   \ 
			S.W   S   S.E 
		Cell-->Popped Cell (i, j) 
		N --> North	 (i-1, j) 
		S --> South	 (i+1, j) 
		E --> East	 (i, j+1) 
		W --> West		 (i, j-1) 
		N.E--> North-East (i-1, j+1) 
		N.W--> North-West (i-1, j-1) 
		S.E--> South-East (i+1, j+1) 
		S.W--> South-West (i+1, j-1)*/

		
		double gNew, hNew, fNew; 

		int I[8][8];
		int J[8][8];
		int *dev_I;
		int *dev_J;
		int *dev_H;
		int *dev_W;
		int *dev_sol;
		int *closedList;

		I = [i-1 i+1 i i i-1 i-1 i+1 i+1 ];
		J = [j j j+1 j-1 j+1 j-1 j+1 j-1];

		int size = 8*sizeof( int );

		cudaMalloc((void**) &I, size);
		cudaMalloc((void**) &J, size);
		cudaMalloc((void**) &parent_j, sizeof( i ));
		cudaMalloc((void**) &parent_j, sizeof( j ));
		cudaMalloc((void**) &dev_dest, sizeof( dest ));
		cudaMalloc((void**) &dev_cellDetails, sizeof( cellDetails ));
		cudaMalloc((void**) &dev_sol, sizeof( dev_sol));
		cudaMalloc((void**) &dev_closedList, sizeof( closedList ));
		cudaMalloc((void**) &dev_H, sizeof ( H ));
		cudaMalloc((void**) &dev_W, sizeof ( W ));

		cudaMemcpy( dev_I, I, sizeof( I ), cudaMemcpyHostToDevice);
		cudaMemcpy( dev_J, J, sizeof( J ), cudaMemcpyHostToDevice);
		cudaMemcpy( parent_j, i , sizeof( i ), cudaMemcpyHostToDevice);
		cudaMemcpy( parent_j , j , sizeof( j ), cudaMemcpyHostToDevice);
		cudaMemcpy( dev_dest, dest , sizeof ( dest ), cudaMemcpyHostToDevice);
		cudaMemcpy( dev_cellDetails , cellDetails , sizeof( cellDetails ), cudaMemcpyHostToDevice);
		cudaMemcpy( dev_sol, sol , sizeof( sol ), cudaMemcpyHostToDevice);
		cudaMemcpy( dev_closedList, closedList , sizeof( closedList ), cudaMemcpyHostToDevice);
		cudaMemcpy( dev_H, H, sizeof( H ) , cudaMemcpyHostToDevice);
		cudaMemcpy( dev_W, W, sizeof( W ), cudaMemcpyHostToDevice);

		Do<<< 1 , THREADS_PER_BLOCK>>> (parent_i,parent_j, I, J, dev_dest, dev_cellDetails, dev_sol, dev_closedList, dev_H, dev_W);

		cudaMemcpy(cellDetails, dev_cellDetails, sizeof( cellDetails ), cudaMemcpyDeviceTo);
		cudaMemcpy(sol, dev_sol, sizeof( cellDetails ), cudaMemcpyDeviceTo);
		cudaMemcpy(closedList, dev_closedList, sizeof( cellDetails ), cudaMemcpyDeviceTo);
	}

	if (foundDest == false) 
		printf("Failed to find the Destination Cell, I guess there is no possible path!!"); 

	return; 
} 

int main() 
{ 	
	vector<Pair> solution_path;
	fstream map;
	fstream position;
	ofstream myfile;
	string Xdimension, Ydimension, filename1, filename2;
	filename1 = "map.txt";
	filename2 = "position.txt";
	map.open(filename1.c_str());
	position.open(filename2.c_str());
	
	map >> Xdimension;
	map >> Ydimension;
	int height = stoi(Xdimension);
	int width = stoi(Ydimension);
 
	int sx,sy,dx,dy;
	position>>sx;
	position>>sy;
	position>>dx;
	position>>dy;

	int grid[ROW][COL];
	int inp_count = 0;
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
				if(map >> grid[i][j]){
					inp_count++;	
				}else{
					break;
				}
		}
	}

	if(inp_count < height*width){
		cout<<"There is some problem with the input file"<<endl;
	}
	else{
		cout<<"Your map is perfect"<<endl;
		
		Pair src = make_pair(sx, sy); 
		Pair dest = make_pair(dx, dy);
		aStarSearch(grid, height, width, src, dest, &solution_path);
		cout<<endl;
		myfile.open ("solution.txt");
  		for(int k=0; k<solution_path.size(); k++){
			myfile <<solution_path[k].first<<" ";
		}
		myfile << "\n";
		for(int k=0; k<solution_path.size(); k++){
			myfile <<solution_path[k].second<<" ";
		}
		myfile << "\n";
		myfile.close();
	}
	
	return(0); 
}

