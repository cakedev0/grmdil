#include <bits/stdc++.h>
 
using namespace std;
 
#define DEBUG 0
 
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
#define REP(i, n) FOR(i, 0, n)
#define TRACE(x) if(DEBUG) cout << #x _ "=" _ x << endl
#define TRACE_VEC(v) if(DEBUG){ cout << #v << _ "="; for(auto e : v) cout _ e; cout << endl; }
#define _ << " " <<
#define endl "\n"

struct Edge{
  int32_t u, v;
  double d;
  bool operator<(const Edge & other){ return d < other.d; }
  Edge(int32_t u, int32_t v, double d) : u(u), v(v), d(d) {}
  Edge() {}
  void in(){ cin >> u >> v >> d; }
  void trace(){ if(DEBUG) cout << u _ v _ d << endl; }
};
 
struct UnionSet{
  vector<pair<int32_t, int32_t> > sets;
  void init(int32_t n){ REP(x, n) sets.push_back({x, 0}); }
  int find(int32_t x){
    if(sets[x].first == x) return x;
    else return sets[x].first = find(sets[x].first);
  }
  void union_both(int32_t x, int32_t y){
    int32_t parentX = find(x), parentY = find(y);
    int32_t rankX = sets[parentX].second, rankY = sets[parentY].second;
    if(parentX == parentY) return;
    else if(rankX < rankY) sets[parentX].first = parentY;
    else sets[parentY].first = parentX;
    if(rankX == rankY) sets[parentX].second++;
  }
};

const int N_MAX = 3000000;
const int N_CLASS = 6;

int n, m;
double P[N_MAX][N_CLASS];
double E[N_MAX][N_CLASS];
double mass[N_CLASS];
vector<Edge> edges;
vector<pair<int, double> > neighs[N_MAX];
int label[N_MAX];
bool viewed[N_MAX];
int degree[N_MAX];
int label_counts[N_MAX][N_CLASS];
double alpha =  0.01;
double beta = 0.02;
double theta = 0.5;
int n_repeat = 1;

double f(double p, int c){
    return - p / pow(mass[c], theta);
}

double g(double d){
    return alpha / (beta + d);
}

void compute_E(int u){
    REP(c, N_CLASS){
        double ene = f(P[u][c], c);
        for(auto p : neighs[u]){
            int v = p.first;
            if(viewed[v]){
                double gd = g(p.second);
                double e_min = 1e9;
                REP(cp, N_CLASS){
                    double e_cp = c == cp ? E[v][cp] : gd + E[v][cp];
                    e_min = min(e_cp, e_min);
                }
                ene += e_min;
            }
        }
        E[u][c] = ene;
    }
}

void backtrack_E(int u){
    int c = 0;
    double gd = 0;
    for(auto p : neighs[u]){
        int v = p.first;
        if(viewed[v]){
            c = label[v];
            gd = g(p.second);
        }
    }

    double e_min = 1e9;
    REP(cp, N_CLASS){
        double e_cp = c == cp ? E[u][cp] : gd + E[u][cp];
        if(e_cp < e_min){
            e_min = e_cp;
            label[u] = cp;
        }
    }
}

bool root[N_MAX];

int32_t main(int argc, char *argv[]){
    ios_base::sync_with_stdio(0);
    scanf("%d %d\n", &n, &m);
    REP(i, n){
        REP(j, N_CLASS) scanf("%lf", &P[i][j]);
        degree[i] = 0;
        viewed[i] = false;
    }
    
    REP(c, N_CLASS) REP(i, n){
        mass[c] += P[i][c];
    }
    double total_mass = 0;
    REP(c, N_CLASS) total_mass += mass[c];
    REP(c, N_CLASS) mass[c] /= total_mass;

    int u, v;
    double d;
    REP(i, m){
        scanf("%d %d %lf", &u, &v, &d);
        edges.push_back(Edge(u, v, d));
    }

    if(argc > 1) alpha = atof(argv[1]);
    if(argc > 2) beta = atof(argv[2]);
    if(argc > 3) n_repeat = atoi(argv[3]);
    if(argc > 4) srand(atoi(argv[4]));
    queue<int> Q;
    
    REP(k_rand, n_repeat){
        random_shuffle(edges.begin(), edges.end());
        REP(u, n){
            neighs[u].clear();
            viewed[u] = false;
            degree[u] = 0; 
        }

        UnionSet U;
        U.init(n);
        for(auto e : edges){
            if(U.find(e.u) != U.find(e.v)){
                U.union_both(e.u, e.v);
                neighs[e.u].push_back({e.v, e.d});
                neighs[e.v].push_back({e.u, e.d});
                degree[e.u] += 1;
                degree[e.v] += 1;
            }
        }
        
        /*
        fill(root, root+n, false);
        REP(u, n) root[U.find(u)] = true;
        int n_roots = 0;
        REP(u, n) n_roots += root[u];
        printf("n_roots: %d\n", n_roots);
        */

        // FORWARD PASS
        REP(u, n){
            if(degree[u] <= 1) Q.push(u);
        }

        vector<int> order;
        while(!Q.empty()){
            u = Q.front();
            order.push_back(u);
            Q.pop();
            for(auto p : neighs[u]){
                v = p.first;
                if(!viewed[v]){
                    degree[v]--;
                    if(degree[v] == 1) Q.push(v);
                }
            }
            compute_E(u);
            viewed[u] = true;
        }

        // BACKWARD PASS
        reverse(order.begin(), order.end());
        fill(viewed, viewed+n, false);
        for(auto u : order){
            backtrack_E(u);
            viewed[u] = true;
        }

        REP(u, n){
            label_counts[u][label[u]] += 1;
        }
    }


    // OUTPUT
    if(!DEBUG){
        REP(u, n){
            int max_c = 0;
            REP(c, N_CLASS){
                if(label_counts[u][c] > label_counts[u][max_c]){
                    max_c = c;
                }
            }
            printf("%d\n", max_c + 1);
        }
    }
}


 
 
 
 
 
 