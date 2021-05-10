#include<bits/stdc++.h> 
#include<omp.h>

using namespace std; 

#define ROW 200
#define COL 200

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

// 0 represents occupied cell, 1 represents free cell and 2 represents the cell in the optimal path
bool isUnBlocked(int grid[][COL], int row, int col) { 
	// returns true if the cell is not blocked else false 
	if (grid[row][col] == 1 || grid[row][col] == 2) 
		return (true); 
	else
		return (false); 
} 

bool isDestination(int row, int col, Pair dest) { 
	if (row == dest.first && col == dest.second) 
		return (true); 
	else
		return (false); 
} 
 
double calculateHValue(int row, int col, Pair dest) {  
	return ((double)sqrt ((row-dest.first)*(row-dest.first) + (col-dest.second)*(col-dest.second))); 
} 

void tracePath(cell cellDetails[][COL], Pair dest, vector<Pair> *solvec) { 
	printf ("optmial path Found :)"); 
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

	return; 
} 

void aStarSearch(int grid[][COL], int H, int W, Pair src, Pair dest, vector<Pair> *sol) {
	// validation check 
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
	
	bool closedList[ROW][COL]; 
	memset(closedList, false, sizeof (closedList)); 
 
	cell cellDetails[ROW][COL]; 

	int i, j; 

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

		//----------- 1st Successor (North) ------------ 

		if (isValid(i-1, j, H, W) == true) {
			if (isDestination(i-1, j, dest) == true) {  
				cellDetails[i-1][j].parent_i = i; 
				cellDetails[i-1][j].parent_j = j;  
				tracePath (cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			} 
			else if (closedList[i-1][j] == false && isUnBlocked(grid, i-1, j) == true) 
			{ 
				gNew = cellDetails[i][j].g + 1.0; 
				hNew = calculateHValue (i-1, j, dest); 
				fNew = gNew + hNew;  
				if (cellDetails[i-1][j].f == FLT_MAX || cellDetails[i-1][j].f > fNew) 
				{ 
					openList.insert( make_pair(fNew, make_pair(i-1, j)));
					cellDetails[i-1][j].f = fNew; 
					cellDetails[i-1][j].g = gNew; 
					cellDetails[i-1][j].h = hNew; 
					cellDetails[i-1][j].parent_i = i; 
					cellDetails[i-1][j].parent_j = j; 
				} 
			} 
		} 

		//----------- 2nd Successor (South) ------------ 

		if (isValid(i+1, j, H, W) == true) { 
			if (isDestination(i+1, j, dest) == true) {  
				cellDetails[i+1][j].parent_i = i; 
				cellDetails[i+1][j].parent_j = j;  
				tracePath(cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			} 
			else if (closedList[i+1][j] == false && isUnBlocked(grid, i+1, j) == true) { 
				gNew = cellDetails[i][j].g + 1.0; 
				hNew = calculateHValue(i+1, j, dest); 
				fNew = gNew + hNew; 
				if (cellDetails[i+1][j].f == FLT_MAX || cellDetails[i+1][j].f > fNew) { 
					openList.insert( make_pair (fNew, make_pair (i+1, j)));
					cellDetails[i+1][j].f = fNew; 
					cellDetails[i+1][j].g = gNew; 
					cellDetails[i+1][j].h = hNew; 
					cellDetails[i+1][j].parent_i = i; 
					cellDetails[i+1][j].parent_j = j; 
				} 
			} 
		} 

		//----------- 3rd Successor (East) ------------ 

		if (isValid (i, j+1, H, W) == true) { 
			if (isDestination(i, j+1, dest) == true) { 
				cellDetails[i][j+1].parent_i = i; 
				cellDetails[i][j+1].parent_j = j; 
				tracePath(cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			} 
			else if (closedList[i][j+1] == false && isUnBlocked (grid, i, j+1) == true) { 
				gNew = cellDetails[i][j].g + 1.0; 
				hNew = calculateHValue (i, j+1, dest); 
				fNew = gNew + hNew; 
				if (cellDetails[i][j+1].f == FLT_MAX || cellDetails[i][j+1].f > fNew) { 
					openList.insert( make_pair(fNew, make_pair (i, j+1)));
					cellDetails[i][j+1].f = fNew; 
					cellDetails[i][j+1].g = gNew; 
					cellDetails[i][j+1].h = hNew; 
					cellDetails[i][j+1].parent_i = i; 
					cellDetails[i][j+1].parent_j = j; 
				} 
			} 
		} 

		//----------- 4th Successor (West) ------------ 
 
		if (isValid(i, j-1, H, W) == true) {
			if (isDestination(i, j-1, dest) == true) { 
				cellDetails[i][j-1].parent_i = i; 
				cellDetails[i][j-1].parent_j = j; 
				tracePath(cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			} 
			else if (closedList[i][j-1] == false && isUnBlocked(grid, i, j-1) == true) { 
				gNew = cellDetails[i][j].g + 1.0; 
				hNew = calculateHValue(i, j-1, dest); 
				fNew = gNew + hNew;  
				if (cellDetails[i][j-1].f == FLT_MAX || cellDetails[i][j-1].f > fNew) { 
					openList.insert( make_pair (fNew, make_pair (i, j-1)));
					cellDetails[i][j-1].f = fNew; 
					cellDetails[i][j-1].g = gNew; 
					cellDetails[i][j-1].h = hNew; 
					cellDetails[i][j-1].parent_i = i; 
					cellDetails[i][j-1].parent_j = j; 
				} 
			} 
		} 

		//----------- 5th Successor (North-East) ------------ 

		if (isValid(i-1, j+1, H, W) == true) {  
			if (isDestination(i-1, j+1, dest) == true) { 
				cellDetails[i-1][j+1].parent_i = i; 
				cellDetails[i-1][j+1].parent_j = j; 
				tracePath (cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			} 
			else if (closedList[i-1][j+1] == false && isUnBlocked(grid, i-1, j+1) == true) { 
				gNew = cellDetails[i][j].g + 1.414; 
				hNew = calculateHValue(i-1, j+1, dest); 
				fNew = gNew + hNew; 
				if (cellDetails[i-1][j+1].f == FLT_MAX || cellDetails[i-1][j+1].f > fNew) { 
					openList.insert( make_pair (fNew, make_pair(i-1, j+1)));
					cellDetails[i-1][j+1].f = fNew; 
					cellDetails[i-1][j+1].g = gNew; 
					cellDetails[i-1][j+1].h = hNew; 
					cellDetails[i-1][j+1].parent_i = i; 
					cellDetails[i-1][j+1].parent_j = j; 
				} 
			} 
		} 

		//----------- 6th Successor (North-West) ------------ 

		if (isValid (i-1, j-1, H, W) == true) {  
			if (isDestination (i-1, j-1, dest) == true) { 
				cellDetails[i-1][j-1].parent_i = i; 
				cellDetails[i-1][j-1].parent_j = j; 
				tracePath (cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			}
			else if (closedList[i-1][j-1] == false && isUnBlocked(grid, i-1, j-1) == true) { 
				gNew = cellDetails[i][j].g + 1.414; 
				hNew = calculateHValue(i-1, j-1, dest); 
				fNew = gNew + hNew;  
				if (cellDetails[i-1][j-1].f == FLT_MAX || cellDetails[i-1][j-1].f > fNew) { 
					openList.insert( make_pair (fNew, make_pair (i-1, j-1))); 
					cellDetails[i-1][j-1].f = fNew; 
					cellDetails[i-1][j-1].g = gNew; 
					cellDetails[i-1][j-1].h = hNew; 
					cellDetails[i-1][j-1].parent_i = i; 
					cellDetails[i-1][j-1].parent_j = j; 
				} 
			} 
		} 

		//----------- 7th Successor (South-East) ------------ 

		if (isValid(i+1, j+1, H, W) == true) { 
			if (isDestination(i+1, j+1, dest) == true) { 
				cellDetails[i+1][j+1].parent_i = i; 
				cellDetails[i+1][j+1].parent_j = j;  
				tracePath (cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			} 
			else if (closedList[i+1][j+1] == false && isUnBlocked(grid, i+1, j+1) == true) { 
				gNew = cellDetails[i][j].g + 1.414; 
				hNew = calculateHValue(i+1, j+1, dest); 
				fNew = gNew + hNew; 
				if (cellDetails[i+1][j+1].f == FLT_MAX || cellDetails[i+1][j+1].f > fNew) { 
					openList.insert(make_pair(fNew, make_pair (i+1, j+1)));
					cellDetails[i+1][j+1].f = fNew; 
					cellDetails[i+1][j+1].g = gNew; 
					cellDetails[i+1][j+1].h = hNew; 
					cellDetails[i+1][j+1].parent_i = i; 
					cellDetails[i+1][j+1].parent_j = j; 
				} 
			} 
		} 

		//----------- 8th Successor (South-West) ------------ 

		if (isValid (i+1, j-1, H, W) == true) { 
			if (isDestination(i+1, j-1, dest) == true) { 
				cellDetails[i+1][j-1].parent_i = i; 
				cellDetails[i+1][j-1].parent_j = j;  
				tracePath(cellDetails, dest, sol); 
				foundDest = true; 
				return; 
			} 
			else if (closedList[i+1][j-1] == false && (grid, i+1, j-1) == true) { 
				gNew = cellDetails[i][j].g + 1.414; 
				hNew = calculateHValue(i+1, j-1, dest); 
				fNew = gNew + hNew; 
				if (cellDetails[i+1][j-1].f == FLT_MAX || cellDetails[i+1][j-1].f > fNew) { 
					openList.insert(make_pair(fNew, make_pair(i+1, j-1)));
					cellDetails[i+1][j-1].f = fNew; 
					cellDetails[i+1][j-1].g = gNew; 
					cellDetails[i+1][j-1].h = hNew; 
					cellDetails[i+1][j-1].parent_i = i; 
					cellDetails[i+1][j-1].parent_j = j; 
				} 
			} 
		} 
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
		#pragma omp parallel for schedule(static) private(j)
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
