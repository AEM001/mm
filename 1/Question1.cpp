#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>

using namespace std;

const int N = 100; // 路口总数
const int INF = INT_MAX;

// 邻接矩阵，matrix[i][j]表示从路口i+1到j+1的限速（0表示无连接）
int matrix[N][N] = { 0 };
int mat1[10][9] = {
    {90,40,40,40,40,40,40,40,40},
    {60,90,90,40,40,60,60,40,40},
    {40,60,60,60,60,60,40,40,40},
    {40,60,90,90,90,60,60,60,60},
    {60,60,60,60,60,60,60,60,40},
    {60,60,90,90,90,90,90,90,60},
    {40,40,60,60,40,40,40,60,40},
    {60,60,40,40,40,40,90,90,90},
    {120,120,120,120,40,40,90,90,90},
    {60,60,60,60,40,60,60,60,60}
};
int mat2[9][10] = {
    {60,90,60,60,40,40,40,60,60,120},
    {40,60,60,60,40,90,60,60,60,120},
    {40,60,60,60,120,90,60,40,40,120},
    {40,60,90,60,120,60,60,60,40,120},
    {60,60,90,60,120,60,60,60,40,40},
    {60,40,90,60,120,60,40,40,40,40},
    {40,40,90,40,120,40,40,40,90,40},
    {60,60,90,40,40,60,60,60,90,60},
    {60,60,90,60,60,60,90,60,60,90},
};

// 初始化邻接矩阵，假设水平限速60，垂直限速80，部分为高速公路120
void initMatrix() {
    for (int i = 0; i < 10; ++i) 
        for (int j = 0; j < 9; ++j) {
            matrix[10 * i + j][10 * i + j + 1] = mat1[i][j];
            matrix[10 * i + j + 1][10 * i + j] = mat1[i][j];
            matrix[10 * j + i][10 * (j + 1) + i] = mat2[j][i];
            matrix[10 * (j + 1) + i][10 * j + i] = mat2[j][i];
        }
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            if (matrix[i][j] == 40)matrix[i][j] = 41.9750;
            else if (matrix[i][j] == 60)matrix[i][j] = 38.4917;
            else if (matrix[i][j] == 90)matrix[i][j] = 38.4198;
            else if (matrix[i][j] == 120)matrix[i][j] = 63.4198;
        }
    }
}

void dijkstra(int start, int end) {
    vector<double> dist(N, INF);    // 最短时间
    vector<int> prev(N, -1);        // 前驱节点
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

    // 起点编号转换为0-based索引
    start--;
    dist[start] = 0.0;
    pq.push({ 0.0, start });

    while (!pq.empty()) {
        double time = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (u == end - 1) break; // 到达终点

        if (time > dist[u]) continue;

        // 遍历所有可能的邻居
        for (int v = 0; v < N; ++v) {
            if (matrix[u][v] == 0) continue; // 无连接

            // 计算通过当前路段的时间
            double speed = matrix[u][v];
            //double new_time = time + 50.0 / speed; // 时间 = 距离50km / 限速
            double new_time = time + speed;
            if (new_time < dist[v]) {
                dist[v] = new_time;
                prev[v] = u;
                pq.push({ new_time, v });
            }
        }
    }

    // 回溯路径
    vector<int> path;
    int current = end - 1; // 转换为0-based索引
    while (current != -1) {
        path.push_back(current + 1); // 转换回1-based编号
        current = prev[current];
    }
    reverse(path.begin(), path.end());

    // 输出结果
    //cout << "最短时间: " << dist[end - 1] << " 小时" << endl;
    cout << "最少费用: " << dist[end - 1] << " 元" << endl;
    cout << "路径: ";
    for (int node : path) {
        cout << node << " ";
    }
    cout << endl;
}

int main() {
    initMatrix(); // 初始化邻接矩阵
    dijkstra(1, 100); // 计算从1到100的最短路径
    return 0;
}