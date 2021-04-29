#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace std;

const int HEIGHT = 20, WIDTH = 60;
const float D = 1, D2 = 1.4;

const char map[HEIGHT][WIDTH] = {
	"...#...............................#..........#............",
	"...#...............................#.......................",
	"...################### #############..........#............",
	"...#...............................################ #######",
	"...#..............#................#.......................",
	"..................#................#.......................",
	"...#..............##################.......................",
	"...#..............#................#........########.......",
	"...####### ##############.##...............................",
	"...#...........#...........#.#######.......................",
	"...#...........#...........#.......#.......................",
	"...#...........#...........#.......#.......................",
	"...#...........#...........############## #################",
	"...#.......................#................#..............",
	"...#...........#...........#................#..............",
	"...################### #####................#..............",
	"...#......#................######### #.....................",
	"...#......#..........#####.#.........#### ###..............",
	"...#.................#.....#.........#......#..............",
	"...#......#..........#.....#.........#......#.............."
};

class CELL{
    public:
        int x, y;
        char sym;
        bool isLet;
        int camefrom;
        float f,g,h;            // f = g(step cost)+h(heuristic distance)

    void calculate_f(){
        f = g+h;
    }

    CELL() {};

	CELL(int X, int Y, bool state = true) :x(X), y(Y), isLet(state) { };
     
    void setPos(int X, int Y){
        x = X;
        y = Y;
    }

    bool operator==(CELL obj){
        return this->x == obj.x && this->y == obj.y;
    }

    vector <CELL> getnearby(CELL area[][WIDTH]){
        vector <CELL> neighbors;
        for (int i=0; i<HEIGHT; i++){
            for (int j=0; j<WIDTH; j++){
                if (abs(area[i][j].x - this->x) <= 1 && abs(area[i][j].y - this->y) <= 1 && !(area[i][j] == *this)){
                    neighbors.push_back(area[i][j]);
                }
            }
        }
        return neighbors;
    }
};

// Heuristic Euclidean function
float distance(CELL start, CELL finish) {
	return sqrt(pow(finish.x - start.x, 2) + pow(finish.y - start.y, 2));
}

vector <CELL> astar(CELL start, CELL target, CELL area[HEIGHT][WIDTH]){
    vector <CELL> returnPath;
    vector <CELL> openSet;
    int i,j;

    for (i=0; i<HEIGHT; i++){
        for (j=0; j<WIDTH; j++){
            area[i][j].setPos(j,i);
            area[i][j].g = (float)INT64_MAX;
            area[i][j].h = (float)INT64_MAX;

            if (map[i][j]=='#'){
                area[i][j].isLet = false;
                area[i][j].sym = "#";
            }
            else{
                area[i][j].isLet = true;
                area[i][j].sym = ".";
            }
        }
    }

    openSet.push_back(start);
    openSet[0].g = 0;
    openSet[0].f = distance(openSet[0], area[target.y][target.x]);
    openSet[0].camefrom = -1;

    vector <CELL> path;
    int k = 0;

    while (openSet.size() != 0){
        CELL current;
        int minF = 0;

        for (i=0; i< openSet.size(); i++){
            if (openSet[minF].F > openSet[i].f){
                minF = i;
            }
        }
        current = openSet[minF];
        path.push_back(current);

        if (current == target){
            cout << "path found!"<<endl;
            break;
        }

        vector <CELL>::iterator iter = find(openSet.begin(), openSet.end(), current);
        openSet.erase(iter);

        vector <CELL> neighbors = current.getnearby(area);

        for (i=0; i<neighbors.size(); i++){
            float g_tmp;
            if (abs(current.x - neighbors[i].x) == 1 && abs(current.y - neighbors[i].y) == 1)
                g_tmp = current.g + D2;
            else
                g_tmp = current.g + D;

            if (g_tmp < neighbors[i].g && neighbors[i].isLet == true)
                neighbors[i].camefrom = k;
                neighbors[i].g = g_tmp;
                neighbors[i].h = ceil(distance(neighbors[i], target));
                neighbors[i].calculate_f();

                if (find(openSet.begin(), openSet.end(), neighbors[i]) == openSet.end())
                    openSet.push_back(neighbors[i]);
        }
    }
    k++;
    CELL current = openSet[openSet.size() - 1];

	do {
		current = path[current.camefrom];
		returnPath.push_back(current);
	} while (current.camefrom != -1);

	return returnPath;
}