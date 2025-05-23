#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>

using namespace std;

const int MAX_NODES = 101; // 节点编号1-100
int speed_limit[MAX_NODES][MAX_NODES] = { 0 };
const double T1 = 12;   //确定一个上限 
vector<int>Path_1 = { 1, 2, 12, 13, 14, 24, 25, 35, 45, 55, 56, 57, 58, 59, 69, 79, 89, 90, 100 };

struct Label {
    int node;
    double time;
    double cost;
    Label* parent;
    double ratio; // 从父节点到当前节点的超速比例

    Label(int n, double t, double c, Label* p, double r)
        : node(n), time(t), cost(c), parent(p), ratio(r) {}

    // 优先队列比较函数（按费用升序）
    bool operator>(const Label& other) const {
        return cost > other.cost;
    }
};

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

void init_speed_limit() {
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 9; ++j) {
            speed_limit[10 * i + j + 1][10 * i + j + 2] = mat1[i][j];
            speed_limit[10 * i + j + 2][10 * i + j + 1] = mat1[i][j];
            speed_limit[10 * j + i + 1][10 * (j + 1) + i + 1] = mat2[j][i];
            speed_limit[10 * (j + 1) + i + 1][10 * j + i + 1] = mat2[j][i];
        }
}

// 获取邻接节点
vector<int> get_neighbors(int u) {
    vector<int> neighbors;
    if (u % 10 != 0) neighbors.push_back(u + 1); // 右
    if (u <= 90) neighbors.push_back(u + 10);     // 上
    return neighbors;
}

int fine_amount(int v_max_limit, double x_ratio) {
    if (x_ratio < 0.0001)return 0;
    double ratio = x_ratio;
    int v_class = v_max_limit;
    if (v_class == 40) {
        if (ratio <= 0.20001)return 50;
        if (ratio <= 0.50001)return 100;
        if (ratio <= 0.70001)return 300;
        return 500;
    }
    if (v_class == 60) {
        if (ratio <= 0.20001)return 100;
        if (ratio <= 0.50001)return 150;
        if (ratio <= 0.70001)return 500;
        return 1000;
    }
    if (v_class == 90) {
        if (ratio <= 0.20001)return 150;
        if (ratio <= 0.50001)return 200;
        if (ratio <= 0.70001)return 1000;
        return 1500;
    }
    if (v_class == 120) {
        if (ratio <= 0.50001)return 200;
        if (ratio <= 0.70001)return 1500;
        return 2000;
    }
    return 0;
}

// 计算罚款期望
double calc_expected_fine(double r, int s) {
    double p_single;
    if (r < 0.20001)p_single = 0.7;
    else if (r < 0.50001)p_single = 0.9;
    else p_single = 0.99;
    double p_detect_fixed;
    if (s >= 90)p_detect_fixed = p_single;
    else p_detect_fixed = 0;
    double p_detect_mobile = 20.0 / 180 * p_single;
    
    double p_det = 1 - (1 - p_detect_fixed) * (1 - p_detect_mobile);
    return p_det * fine_amount(s, r);
}

void solve_problem2() {
    for (int i = 0; ; ++i) {
        double TimeLimit = T1 - i * 0.001;
        if (TimeLimit < 6.3725)break;
        cout << "TimeLimit = " << TimeLimit << '\t';

        init_speed_limit();
        const int total_roads = 180; // 总路段数
        vector<double> r_options = { 0.0,0.2,0.5,0.7 }; // 离散的超速选项

        // 优先队列（最小堆）
        auto cmp = [](Label* a, Label* b) { return a->cost > b->cost; };
        priority_queue<Label*, vector<Label*>, decltype(cmp)> pq(cmp);
        unordered_map<int, vector<Label*>> labels;

        // 初始化起点
        Label* start = new Label(1, 0.0, 0.0, nullptr, 0.0);
        labels[1].push_back(start);
        pq.push(start);

        while (!pq.empty()) {
            Label* current = pq.top();
            pq.pop();

            if (current->node == 100) continue; // 终点处理在最后

            for (int v : get_neighbors(current->node)) {
                int s = speed_limit[current->node][v];
                if (s == 0) continue; // 无效路段

                for (double r : r_options) {
                    if (r > 0.7) continue;

                    double v_actual = s * (1 + r);
                    double t_segment = 50.0 / v_actual;
                    double new_time = current->time + t_segment;

                    if (new_time > TimeLimit) continue;

                    // 汽油费
                    double petrol = 7.76 * (0.0625 * v_actual + 1.875) * 0.5;
                    // 过路费
                    double toll = (s == 120) ? 25.0 : 0.0;
                    // 罚款期望
                    double fine = calc_expected_fine(r, s);
                    // 餐饮费用
                    double food = 20 * t_segment;

                    double new_cost = current->cost + petrol + toll + fine + food;

                    Label* new_label = new Label(v, new_time, new_cost, current, r);

                    // 剪枝
                    bool dominated = false;
                    auto& v_labels = labels[v];
                    for (auto it = v_labels.begin(); it != v_labels.end();) {
                        Label* existing = *it;
                        if (existing->time <= new_time && existing->cost <= new_cost) {
                            dominated = true;
                            break;
                        }
                        if (new_label->time <= existing->time && new_label->cost <= existing->cost) {
                            it = v_labels.erase(it);
                        }
                        else {
                            ++it;
                        }
                    }

                    if (!dominated) {
                        v_labels.push_back(new_label);
                        pq.push(new_label);
                    }
                    else {
                        delete new_label;
                    }
                }
            }
        }

        // 提取最优路径
        Label* best = nullptr;
        double min_cost = numeric_limits<double>::max();
        double total_time = 0.0; 
        for (Label* lbl : labels[100]) {
            if (lbl->cost < min_cost) {
                min_cost = lbl->cost;
                best = lbl;
                total_time = lbl->time; // 保存最优时间
            }
        }

        // 重建路径和超速比例
        vector<int> path;
        vector<double> ratios;
        while (best != nullptr) {
            path.push_back(best->node);
            if (best->parent) ratios.push_back(best->ratio);
            best = best->parent;
        }
        reverse(path.begin(), path.end());
        reverse(ratios.begin(), ratios.end());

        // 输出结果（包含总费用和总时间）
        cout << "Path: ";
        for (int node : path) cout << node << " ";
        cout << '\t';
        if (Path_1 == path)cout << "PATH1";
        cout << endl;
        /*
        cout << "\nSpeed Ratios: ";
        for (double r : ratios) cout << fixed << setprecision(4) << r << " ";
        cout << "\nTotal Cost: " << fixed << setprecision(4) << min_cost;
        cout << "\nTotal Time: " << fixed << setprecision(4) << total_time << endl; // 新增时间输出
        */
    }
}

int main() {
    solve_problem2();
    return 0;
}