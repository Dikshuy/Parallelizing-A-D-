#include <iostream>
#include <queue>
#include <limits>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <sstream>

double Inf = 10e9;
using namespace std;

struct vertex{
    int x, y;
    float k1;
    float k2;
    vertex(int, int, float, float);
    vertex() : x(0), y(0), k1(0), k2(0) {}
};
vertex::vertex(int p_x, int p_y, float p_k1, float p_k2) : x{p_x}, y{p_y}, k1{p_k1}, k2{p_k2} {}
//Goal & start
vertex s_goal(5, 5, 0, 0);
vertex s_start(0, 0, 0, 0);
int km = 0;

//constraints
#define grid_s_x 1000
#define grid_s_y 1000

double rhs[grid_s_x][grid_s_y];
double g[grid_s_x][grid_s_y];

bool GRID[grid_s_x][grid_s_y];

bool Ukey[grid_s_x][grid_s_y];
float Ukey_k1[grid_s_x][grid_s_y];
float Ukey_k2[grid_s_x][grid_s_y];

struct compare{ 
    bool operator()(const vertex &a, const vertex &b){
        if (a.k1 > b.k1){
            return 1;
        }
        else if ((a.k1 == b.k1)){
            if (a.k2 > b.k2)
                return 1;
            else
                return 0;
        }
        else
            return 0;
    }
};

bool isVertexEqual(vertex v1, vertex v2){
    if (v1.x == v2.x && v1.y == v2.y)
    {
        return 1;
    }
    return 0;
}

typedef priority_queue<vertex, vector<vertex>, compare> m_priority_queue; //Min Priority Queue

m_priority_queue U;

void showpq(m_priority_queue gq){
    m_priority_queue g = gq;
    while (!g.empty())
    {
        vertex c_v = g.top();
        g.pop();
    }
    cout << '\n';
}

double h(vertex s1, vertex s2){
    return sqrt(pow((s1.x - s2.x), 2) + pow((s1.y - s2.y), 2));
}

bool isInQueue(vertex s){
    if (Ukey[s.x][s.y] == 1){
        return 1;
    }
    return 0;
}

void pushToQueue(vertex s){
    U.push(s);
    Ukey[s.x][s.y] = 1;
    Ukey_k1[s.x][s.y] = s.k1;
    Ukey_k2[s.x][s.y] = s.k2;
}

bool isCostLower(vertex b, vertex a)
{
    if (a.k1 > b.k1){
        return 1;
    }
    else if (a.k1 == b.k1){
        if (a.k2 > b.k2)
            return 1;
        else
            return 0;
    }
    else
        return 0;
}

vertex CalculateKey(vertex s){
    double k1 = min(g[s.x][s.y], rhs[s.x][s.y]) + h(s_start, s) + km;
    double k2 = min(g[s.x][s.y], rhs[s.x][s.y]);

    s.k1 = k1;
    s.k2 = k2;
    return s;
}

void UpdateVertex(vertex u){
    if (u.x < 0 || u.x > grid_s_x || u.y < 0 || u.y > grid_s_y){
        return;
    }

    if (!isVertexEqual(u, s_goal)){
        double c[3][3];

        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= 2; i++){
            for (int j = 0; j <= 2; j++){
                if (abs(i - 1) != abs(j - 1)){
                    c[i][j] = g[u.x + i - 1][u.y + j - 1] + 1 + GRID[u.x + i - 1][u.y + j - 1] * Inf;
                    if (u.x + i - 1 > grid_s_x || u.y + j - 1 > grid_s_y || u.x + i - 1 < 0 || u.y + j - 1 < 0 || GRID[u.x + i - 1][u.y + j - 1] == 1)
                    {
                        c[i][j] == Inf;
                    }
                }
            }
        }

        rhs[u.x][u.y] = min(min(c[1][2], c[2][1]), min(c[1][0], c[0][1]));
    }
    u = CalculateKey(u);
    if (isInQueue(u)){
        Ukey[u.x][u.y] = 0; 
    }
    if (rhs[u.x][u.y] != g[u.x][u.y]){
        pushToQueue(u);
    }
}

bool isGhost(vertex s)
{
    if (Ukey[s.x][s.y] == 1 && Ukey_k1[s.x][s.y] == s.k1 && Ukey_k2[s.x][s.y] == s.k2){
        return 0;
    }
    return 1;
}

void pop(){
    vertex s = U.top();
    Ukey[s.x][s.y] = 0;
    U.pop();
}

vertex TopKey(){
    if (U.size() == 0)
        return vertex(-1, -1, Inf, Inf);

    vertex temp = U.top();

    while (isGhost(temp))
    {
        pop(); //pop unwanted ones
        if (U.size() == 0)
            return vertex(-1, -1, Inf, Inf);
        temp = U.top();
    }
    return temp; //return top most vertex which isn't a ghost
}

void ComputeShortestPath(){
    while (isCostLower(TopKey(), CalculateKey(s_start)) ||
           rhs[s_start.x][s_start.y] != g[s_start.x][s_start.y])
    {

        vertex k_old = TopKey();
        pop();
        vertex u = k_old;

        if (isCostLower(k_old, CalculateKey(u)))
        {
            u = CalculateKey(u);
            pushToQueue(u);
        }
        else if (g[u.x][u.y] > rhs[u.x][u.y])
        {
            g[u.x][u.y] = rhs[u.x][u.y];

            #pragma omp parallel for collapse(2)
            for (int i = 0; i <= 2; i++)
            {
                for (int j = 0; j <= 2; j++)
                {
                    if (abs(i - 1) != abs(j - 1))
                    {
                        UpdateVertex(vertex(u.x + i - 1, u.y + j - 1, 0, 0));
                    }
                }
            }
        }
        else
        {
            g[u.x][u.y] = Inf;
            cout << " => else" << endl;

            #pragma omp parallel for collapse(2)
            for (int i = 0; i <= 2; i++)
            {
                for (int j = 0; j <= 2; j++)
                {
                    if (abs(i - 1) != abs(j - 1) || (i == 1 && j == 1))
                    {
                        UpdateVertex(vertex(u.x + i - 1, u.y + j - 1, 0, 0));
                    }
                }
            }
        }
    }
}
void fillGRID()
{
    string line;
    ifstream textfile("in.in");

    int i = 0;
    while (getline(textfile, line))
    {
        std::stringstream ss(line);

        int j = 0;

        for (int x; ss >> x; j++)
        {
            GRID[j][i] = x;
            if (ss.peek() == ',')
                ss.ignore();
        }
        i++;
    }
    GRID[1][1] = 1;
    GRID[2][1] = 1;
    GRID[3][1] = 1;
    GRID[2][2] = 1;
    GRID[2][3] = 1;
    textfile.close();
}

int main()
{
    omp_set_num_threads(8);
    //Initialize
    fillGRID();
    /*   cout << "Successfully loaded GRID" << endl;

    for (int k = 0; k < 50; k++)
    {
        for (int m = 0; m < 50; m++)
        {
            cout << GRID[k][m] << " ";
        }
        cout << endl;
    }*/

    km = 0;

#pragma omp parallel for
    for (int i = 0; i < grid_s_x; i++)
#pragma omp parallel for
        for (int j = 0; j < grid_s_y; j++)
        {
            rhs[i][j] = Inf;
            g[i][j] = Inf;
        }
    rhs[s_goal.x][s_goal.y] = 0;

    s_goal = CalculateKey(s_goal);
    pushToQueue(s_goal);

    ComputeShortestPath();
    showpq(U);

    /*   cout << "Successfully loaded rhs" << endl;

    for (int k = 0; k < 25; k++)
    {
        for (int m = 0; m < 25; m++)
        {
            cout << g[k][m] << "\t";
        }
        cout << endl;
    }*/
    return 0;
}