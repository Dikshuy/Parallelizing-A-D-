#include <iostream>
#include <fstream>
#include <iomanip>
#include <queue>
#include <string>
#include <math.h>

using namespace std; 

#define ROW 10
#define COL 10

const int n=10; // horizontal size of the map
const int m=10; // vertical size size of the map
static int map[n][m];
static int closed_nodes_map[n][m]; // map of closed (tried-out) nodes
static int open_nodes_map[n][m]; // map of open (not-yet-tried) nodes
static int dir_map[n][m]; // map of directions
const int dir=4; // number of possible directions to go at any position
// if dir==4
static int dx[dir]={1, 0, -1, 0};
static int dy[dir]={0, 1, 0, -1};
// // if dir==8
// static int dx[dir]={1, 1, 0, -1, -1, -1, 0, 1};
// static int dy[dir]={0, 1, 1, 1, 0, -1, -1, -1};

class node
{
    // current position
    int xPos;
    int yPos;
    // total distance already travelled to reach the node
    int level;
    // priority=level+remaining distance estimate
    int priority;  // smaller: higher priority

    public:
        node(int xp, int yp, int d, int p) 
            {xPos=xp; yPos=yp; level=d; priority=p;}
    
        int getxPos() const {return xPos;}
        int getyPos() const {return yPos;}
        int getLevel() const {return level;}
        int getPriority() const {return priority;}

        void updatePriority(const int & xDest, const int & yDest)
        {
             priority=level+estimate(xDest, yDest)*10; //A*
        }

        // give better priority to going strait instead of diagonally
        void nextLevel(const int & i) // i: direction
        {
             level+=(dir==8?(i%2==0?10:14):10);
        }
        
        // Estimation function for the remaining distance to the goal.
        const int & estimate(const int & xDest, const int & yDest) const
        {
            static int xd, yd, d;
            xd=xDest-xPos;
            yd=yDest-yPos;         

            // Euclidian Distance
            d=static_cast<int>(sqrt(xd*xd+yd*yd));

            // Manhattan distance
            //d=abs(xd)+abs(yd);
            
            // Chebyshev distance
            //d=max(abs(xd), abs(yd));

            return(d);
        }
};

// Determine priority (in the priority queue)
bool operator<(const node & a, const node & b)
{
  return a.getPriority() > b.getPriority();
}

// A-star algorithm.
// The route returned is a string of direction digits.
string pathFind( const int & xStart, const int & yStart, 
                 const int & xFinish, const int & yFinish )
{
    static priority_queue<node> pq[2]; // list of open (not-yet-tried) nodes
    static int pqi; // pq index
    static node* n0;
    static node* m0;
    static int i, j, x, y, xdx, ydy;
    static char c;
    pqi=0;

    // reset the node maps
    for(y=0;y<m;y++)
    {
        for(x=0;x<n;x++)
        {
            closed_nodes_map[x][y]=0;
            open_nodes_map[x][y]=0;
        }
    }

    // create the start node and push into list of open nodes
    n0=new node(xStart, yStart, 0, 0);
    n0->updatePriority(xFinish, yFinish);
    pq[pqi].push(*n0);
    open_nodes_map[x][y]=n0->getPriority(); // mark it on the open nodes map

    // A* search
    while(!pq[pqi].empty())
    {
        // get the current node w/ the highest priority
        // from the list of open nodes
        n0=new node( pq[pqi].top().getxPos(), pq[pqi].top().getyPos(), 
                     pq[pqi].top().getLevel(), pq[pqi].top().getPriority());

        x=n0->getxPos(); y=n0->getyPos();

        pq[pqi].pop(); // remove the node from the open list
        open_nodes_map[x][y]=0;
        // mark it on the closed nodes map
        closed_nodes_map[x][y]=1;

        // quit searching when the goal state is reached
        //if((*n0).estimate(xFinish, yFinish) == 0)
        if(x==xFinish && y==yFinish) 
        {
            // generate the path from finish to start
            // by following the directions
            string path="";
            while(!(x==xStart && y==yStart))
            {
                j=dir_map[x][y];
                c='0'+(j+dir/2)%dir;
                path=c+path;
                x+=dx[j];
                y+=dy[j];
            }

            // garbage collection
            delete n0;
            // empty the leftover nodes
            while(!pq[pqi].empty()) pq[pqi].pop();           
            return path;
        }

        // generate moves (child nodes) in all possible directions
        for(i=0;i<dir;i++)
        {
            xdx=x+dx[i]; ydy=y+dy[i];

            if(!(xdx<0 || xdx>n-1 || ydy<0 || ydy>m-1 || map[xdx][ydy]==1 
                || closed_nodes_map[xdx][ydy]==1))
            {
                // generate a child node
                m0=new node( xdx, ydy, n0->getLevel(), 
                             n0->getPriority());
                m0->nextLevel(i);
                m0->updatePriority(xFinish, yFinish);

                // if it is not in the open list then add into that
                if(open_nodes_map[xdx][ydy]==0)
                {
                    open_nodes_map[xdx][ydy]=m0->getPriority();
                    pq[pqi].push(*m0);
                    // mark its parent node direction
                    dir_map[xdx][ydy]=(i+dir/2)%dir;
                }
                else if(open_nodes_map[xdx][ydy]>m0->getPriority())
                {
                    // update the priority info
                    open_nodes_map[xdx][ydy]=m0->getPriority();
                    // update the parent direction info
                    dir_map[xdx][ydy]=(i+dir/2)%dir;

                    // replace the node
                    // by emptying one pq to the other one
                    // except the node to be replaced will be ignored
                    // and the new node will be pushed in instead
                    while(!(pq[pqi].top().getxPos()==xdx && 
                           pq[pqi].top().getyPos()==ydy))
                    {                
                        pq[1-pqi].push(pq[pqi].top());
                        pq[pqi].pop();       
                    }
                    pq[pqi].pop(); // remove the wanted node
                    
                    // empty the larger size pq to the smaller one
                    if(pq[pqi].size()>pq[1-pqi].size()) pqi=1-pqi;
                    while(!pq[pqi].empty())
                    {                
                        pq[1-pqi].push(pq[pqi].top());
                        pq[pqi].pop();       
                    }
                    pqi=1-pqi;
                    pq[pqi].push(*m0); // add the better node instead
                }
                else delete m0; // garbage collection
            }
        }
        delete n0; // garbage collection
    }
    return ""; // no route found
}

int main(){
      
    // time_t t;
    // srand(time(NULL));	
	fstream maps;
	fstream position;
	ofstream myfile;
	string Xdimension, Ydimension, filename1, filename2;
	filename1 = "map.txt";
	filename2 = "position.txt";
	maps.open(filename1.c_str());
	position.open(filename2.c_str());

	maps >> Xdimension;
	maps >> Ydimension;
	int height = stoi(Xdimension);
	int width = stoi(Ydimension);

	int start_x;
    int start_y;
    int dest_x;
    int dest_y;
	vector<int> solution_path_x;
	vector<int> solution_path_y;
 
	
	position>>start_x;
	position>>start_y;
	position>>dest_x;
	position>>dest_y;

	int map[height][width];
	int inp_count = 0;
  
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
				if(maps >> map[i][j]){
					inp_count++;	
				}else{
					break;
				}
		}
	}
	int x=0;
	int y=0;

	cout<<"************** generating map *************************"<<endl;
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			cout<<map[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	cout<<"************** map generated *************************"<<endl;

	if(inp_count < height*width){
		cout<<"There is some problem with the input file"<<endl;
	}
	else{
		cout<<"Your map is perfect"<<endl;
// int dim = 6;
//   int map[ROW][COL];
// 	int *results = (int *) malloc(dim * dim * sizeof(int));
//   srand((unsigned) time(&t));
// 	double p = 0.5;

// 	for (int i = 0; i < dim; i++){
//       for(int j=0;j<dim; j++)
//       {
// 		map[i][j] = p < (double)rand()/(double)(RAND_MAX);
		
// 	}
//   }

//   for (int i=0;i<dim;i++){
// 	    	for (int j=0; j<dim;j++){
// 	    		cout<<map[i][j]<<" ";
// 	    	}
//         cout<<endl;
// 	    }

     

//   //# print_board(map, dim);	
// 	//1s are spaces you can walk on

//   int start_x=0,start_y=0,dest_x=0,dest_y=0;
// 	int startPoint = -1;
// 	while (startPoint == -1){
// 		int rand1 = rand() % dim;
//     int rand2 = rand() % dim;
// 		if (map[rand1][rand2] == 1)
// 			start_x= rand1;
//       start_y= rand2;
//       startPoint=start_x*dim+start_y;
//       // cout<<endl<<"Hi_Start_Point_Here"<<start_x<<" "<<start_y<<" "<<startPoint<<endl;
//       	}
	
// 	int endPoint = -1;
// 	while (endPoint == -1){
// 		int rand1 = rand() % dim;
//     int rand2 = rand() % dim;
// 		if (map[rand1][rand2] == 1)
// 			dest_x= rand1;
//       dest_y= rand2;
//       endPoint=dest_x*dim+dest_y;
//       if(endPoint==startPoint) {endPoint=-1;}
//        // cout<<endl<<"Hi_End_Point_Here "<<endPoint<<endl;
		
// 	}

		cout<<"starting coordinates: "<<start_x<<" "<<start_y;
		cout<<endl;
		cout<<"destination coordinates: "<<dest_x<<" "<<dest_y;
		cout<<endl;

		// aStarSearch(grid, height, width, src, dest, &solution_path);
		string route = pathFind(start_x, start_y, dest_x, dest_y);
		cout<<"**************************************";
		cout<< route;
		cout<<"**************************************";
		cout<<endl;
		if(route.length()>0){
      // cout<<"hello world  from if"<<endl;
	        int j; char c;
	        x=start_x;
	        y=start_y;
	        // map[x][y]=2;
	        for(int i=0;i<route.length();i++)
	        {
	            c =route.at(i);
	            j=atoi(&c); 
	            x=x+dx[j];
	            y=y+dy[j];
	            map[x][y]=2;
	        }
	        // map[x][y]=2;
	    
	        // // display the map with the route
	        // for(int y=0;y<COL;y++)
	        // {
	        //     for(int x=0;x<ROW;x++)
	        //         if(map[x][y]==0)
	        //             cout<<".";
	        //         else if(map[x][y]==1)
	        //             cout<<"O"; //obstacle
	        //         else if(map[x][y]==2)
	        //             cout<<"S"; //start
	        //         else if(map[x][y]==3)
	        //             cout<<"R"; //route
	        //         else if(map[x][y]==4)
	        //             cout<<"F"; //finish
	        //     cout<<endl;
	        // }
	    }

	    for (int i=0;i<ROW;i++){
	    	for (int j=0; j<COL;j++){
	    		cout<<map[i][j]<<" ";
	    	}
        cout<<endl;
	    }

		myfile.open ("solution.txt");
		// cout<<"lol"<<endl;
		for (int i=0; i<height; i++){
			for(int j=0; j<width; j++){
				if (map[i][j]==2){
					solution_path_x.push_back(i);
					solution_path_y.push_back(j);
				}
			}
		}
  		for(int k=0; k<solution_path_x.size(); k++){
  			// cout<<solution_path_x[k]<<" ";
			myfile <<solution_path_x[k]<<" ";
		}
		cout<<endl;
		myfile << "\n";
		for(int k=0; k<solution_path_y.size(); k++){
			// cout<<solution_path_y[k]<<" ";
			myfile <<solution_path_y[k]<<" ";
		}
		myfile << "\n";
		cout<<endl;
		myfile.close();
	}
    return 0;
}