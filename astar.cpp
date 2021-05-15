#include <iostream>
#include <fstream>
#include <iomanip>
#include <queue>
#include <string>
#include <math.h>

using namespace std; 

#define ROW 60
#define COL 60

static int map[ROW][COL];
static int closed_nodes_map[ROW][COL]; // map of closed (tried-out) nodes
static int open_nodes_map[ROW][COL]; // map of open (not-yet-tried) nodes
static int dir_map[ROW][COL]; // map of directions
const int dir=4; 
static int dx[dir]={1, 0, -1, 0};
static int dy[dir]={0, 1, 0, -1};

class node
{
    int xPos;
    int yPos;
    int level;
    int priority;

    public:
        node(int xp, int yp, int d, int p) 
            {xPos=xp; yPos=yp; level=d; priority=p;}
    
        int getxPos() const {return xPos;}
        int getyPos() const {return yPos;}
        int getLevel() const {return level;}
        int getPriority() const {return priority;}

        void updatePriority(const int & xDest, const int & yDest)
        {
             priority=level+estimate(xDest, yDest)*10; //f=g+h
        }

        void nextLevel(const int & i)
        {
             level+=10;
        }
        
        const int & estimate(const int & xDest, const int & yDest) const
        {
            static int xd, yd, d;
            xd=xDest-xPos;
            yd=yDest-yPos;         

            d=static_cast<int>(sqrt(xd*xd+yd*yd));
            return(d);
        }
};

bool operator<(const node & a, const node & b)
{
  return a.getPriority() > b.getPriority();
}

string pathFind( const int & xStart, const int & yStart, const int & xFinish, const int & yFinish )
{
    static priority_queue<node> pq[2]; 
    static int pqi;
    static node* n0;
    static node* m0;
    static int i, j, x, y, xdx, ydy;
    static char c;
    pqi=0;

    for(y=0;y<COL;y++)
    {
        for(x=0;x<ROW;x++)
        {
            closed_nodes_map[x][y]=0;
            open_nodes_map[x][y]=0;
        }
    }

    n0=new node(xStart, yStart, 0, 0);
    n0->updatePriority(xFinish, yFinish);
    pq[pqi].push(*n0);
    open_nodes_map[x][y]=n0->getPriority(); 

    // A* search
    while(!pq[pqi].empty())
    {
        // get the current node w/ the highest priorityvfrom the list of open nodes
        n0=new node( pq[pqi].top().getxPos(), pq[pqi].top().getyPos(), 
                     pq[pqi].top().getLevel(), pq[pqi].top().getPriority());

        x=n0->getxPos(); y=n0->getyPos();

        pq[pqi].pop(); // remove the node from the open list
        open_nodes_map[x][y]=0;
        // mark it on the closed nodes map
        closed_nodes_map[x][y]=1;

        // quit searching when the goal state is reached
        if(x==xFinish && y==yFinish) 
        {
            // generate the path from finish to start by following the directions
            string path="";
            while(!(x==xStart && y==yStart))
            {
                j=dir_map[x][y];
                c='0'+(j+dir/2)%dir;
                path=c+path;
                x+=dx[j];
                y+=dy[j];
            }
            delete n0;
            while(!pq[pqi].empty()) pq[pqi].pop();           
            return path;
        }

        for(i=0;i<dir;i++)
        {
            xdx=x+dx[i]; ydy=y+dy[i];
            if(!(xdx<0 || xdx>ROW-1 || ydy<0 || ydy>COL-1 || map[xdx][ydy]==1 || closed_nodes_map[xdx][ydy]==1))
            {
                m0=new node( xdx, ydy, n0->getLevel(), n0->getPriority());
                m0->nextLevel(i);
                m0->updatePriority(xFinish, yFinish);
                if(open_nodes_map[xdx][ydy]==0)
                {
                    open_nodes_map[xdx][ydy]=m0->getPriority();
                    pq[pqi].push(*m0);
                    dir_map[xdx][ydy]=(i+dir/2)%dir;
                }
                else if(open_nodes_map[xdx][ydy]>m0->getPriority())
                {
                    open_nodes_map[xdx][ydy]=m0->getPriority();
                    dir_map[xdx][ydy]=(i+dir/2)%dir;
                    while(!(pq[pqi].top().getxPos()==xdx && pq[pqi].top().getyPos()==ydy))
                    {                
                        pq[1-pqi].push(pq[pqi].top());
                        pq[pqi].pop();       
                    }
                    pq[pqi].pop();  
                    if(pq[pqi].size()>pq[1-pqi].size()) pqi=1-pqi;
                    while(!pq[pqi].empty())
                    {                
                        pq[1-pqi].push(pq[pqi].top());
                        pq[pqi].pop();       
                    }
                    pqi=1-pqi;
                    pq[pqi].push(*m0); 
                }
                else delete m0;
            }
        }
        delete n0; 
    }
    return "";
}

int main(){
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
	int height = stoi(Xdimension); //height = COL
	int width = stoi(Ydimension);  //width = ROW

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
	cout<<"************************* generated map *************************"<<endl;
	for(int j=0;j<COL;j++){
		for(int i=0;i<ROW;i++){
			cout<<map[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	
	if(inp_count < ROW*COL){
		cout<<"There is some problem with the input file"<<endl;
	}
	else{
		cout<<"Your map is perfect"<<endl;

		cout<<"starting coordinates: "<<start_x<<" "<<start_y;
		cout<<endl;
		cout<<"destination coordinates: "<<dest_x<<" "<<dest_y;
		cout<<endl;

		string route = pathFind(start_x, start_y, dest_x, dest_y);
		cout<<"Directions: 0->Right, 1->South, 3->Left, 4-> North"<<endl;
		cout<<"**************************************"<<endl;
		cout<< route<<endl;
		cout<<"**************************************"<<endl;
		cout<<endl;
		if(route.length()>0){
	        int j; char c;
	        int x=start_x;
	        int y=start_y;
	        map[x][y]=2;
	        for(int i=0;i<route.length();i++)
	        {
	            c =route.at(i);
	            j=atoi(&c); 
	            x=x+dx[j];
	            y=y+dy[j];
	            map[x][y]=2;
	        }
	        map[x][y]=2;
	    }

	    for (int j=0;j<COL;j++){
	    	for (int i=0; i<ROW;i++){
	    		cout<<map[i][j]<<" ";
	    	}
        cout<<endl;
	    }

		myfile.open ("solution.txt");
		for (int j=0; j<COL; j++){
			for(int i=0; i<ROW; i++){
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