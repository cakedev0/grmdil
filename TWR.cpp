#include <bits/stdc++.h>

#define FOR(i, n) for(int i=0; (i) < (int)(n); ++(i))

using namespace std;

const int MAX_N = 3*1000*1000;
const int NB_CLASS = 6;
const int MAX_ITER = 40;
const double ALPHA = 500;
const double BETA = 0.5;
const double EPS = 1E-3;

double P[MAX_N][NB_CLASS];

struct Edge {
    int u, v;
    double d;
    bool taken;
    Edge(int u, int v, double d, bool taken=false) :
        u(u), v(v), d(d), taken(taken) {}
    int other(int x) {
        return u+v-x;
    }
};

struct Graph {
    vector<vector<int>> graph;
    vector<Edge> edges;
    Graph(int n, int m) {
        graph.resize(n);
        int u, v; double d;
        FOR(i, m) {
            scanf("%d%d%lf", &u, &v, &d);
            graph[u].push_back(edges.size());
            graph[v].push_back(edges.size());
            edges.push_back({u, v, d});
        }
    }
    int operator ()(int s, int e) {
        return edges[e].other(s);
    }
};

struct NodeOrder {
    int n;
    vector<int> index, inv;
    bool reversed;
    NodeOrder(int n) : n(n) {
        reversed = false;
        index.resize(n); // Mean Chain size: 2.53
        inv.resize(n);
        FOR(i, n)
            index[i] = i;
        int tmp;
        FOR(i, n) {
            tmp = random() % (n-i);
            swap(index[i], index[i+tmp]);
            inv[index[i]] = i;
        }
    }
    int operator [](int i) const {
        return reversed ? inv[n-i-1] : inv[i];
    }
    bool less(int a, int b) const {
        return reversed ^ (index[a] < index[b]);
    }
    void reverse() {
        reversed = !reversed;
    }
};

struct Message {
    Graph& G;
    vector<pair<vector<double>, vector<double>>> M;
    Message(int m, Graph& G) : G(G) {
        M.resize(m, pair<vector<double>, vector<double>>({
                    vector<double>(NB_CLASS, 0),
                    vector<double>(NB_CLASS, 0)
                }));
    }
    vector<double>& operator ()(int s, int e) {
        auto& m = (s == G.edges[e].u) ? M[e].first : M[e].second;
        return m;
    }
};

struct Theta {
    int n, m;
    double cst;
    vector<vector<double>> node;
    vector<vector<vector<double>>> edge;
    Theta(int n, int m) : n(n), m(m) {
        node.resize(n, vector<double>(NB_CLASS));
        edge.resize(m, vector<vector<double>>(NB_CLASS,
                    vector<double>(NB_CLASS)));
    }
    void init(Graph& G) {
        // theta_const
        cst = 0;
        // theta_s
        double eta = 0;
        double eta_c[NB_CLASS];
        FOR(i, n) FOR(c, NB_CLASS) {
            eta += P[i][c];
            eta_c[c] += P[i][c];
        }
        FOR(i, n) {
            FOR(c, NB_CLASS)
                node[i][c] = - sqrt(eta/eta_c[c]) * P[i][c];
        }
        // theta_st
        FOR(e, m) {
            FOR(c, NB_CLASS) FOR(cp, NB_CLASS) if(c != cp)
                edge[e][c][cp] = ALPHA / (BETA + G.edges[e].d);
        }
    }
};

struct ChainSet {
    vector<vector<int>> chains, edges;
    vector<double> Ps, Pst;
    vector<double> np, nm;
    NodeOrder& order;
    ChainSet(int n, int m, NodeOrder& order, Graph& G) : order(order) {
        Ps.resize(n);
        Pst.resize(m);
        np.resize(n);
        nm.resize(n);
        //double tot0 = 0, tot1 = 0;
        FOR(_i, G.graph.size()) {
            int i = order.inv[_i];
            for(auto e: G.graph[i]) if(!G.edges[e].taken) {
                assert(order.less(i, G(i, e)));
                chains.push_back({i});
                edges.push_back({});
                addChain(i, e, G);
                //tot0++; tot1 += chains[chains.size()-1].size();
                set<int> S;
                for(auto s: chains[chains.size()-1])
                    S.insert(s);
                for(auto s: S)
                    ++Ps[s];
                for(auto e: edges[edges.size()-1])
                    ++Pst[e];
            }
        }
        //printf("%lf\n", tot1/tot0);
    }
    void reverseOrder() {
        order.reverse();
    }
    void addChain(int u, int e, Graph& G) {
        np[G(u, e)]++;
        nm[u]++;
        G.edges[e].taken = true;
        int v = G(u, e);
        chains[chains.size()-1].push_back(v);
        edges[chains.size()-1].push_back(e);
        for(auto ep: G.graph[v])
            if(e != ep && !G.edges[ep].taken
                    && order.less(v, G(v, ep))) {
                addChain(v, ep, G);
                break;
            }
    }
    double gamma(int s, int e) const {
        return 1 / max(np[s], nm[s]);
        //return Ps[s] / Pst[e];
    }
};

int n, m;

int main(int argc, char** argv) {
    srand(42);
    scanf("%d%d", &n, &m);
    FOR(i, n) {
        FOR(j, NB_CLASS)
            scanf("%lf", &P[i][j]);
    }
    Graph G(n, m);
    NodeOrder I(n);
    ChainSet T(n, m, I, G);
    Theta bar(n, m), hat(n, m);
    bar.init(G);
    double Ebound, delta;
    double Elast = -numeric_limits<double>::infinity();
    Message M(m, G);
    
    FOR(it, MAX_ITER) { // stopping criterion
        // Step 1
        Ebound = bar.cst;
        FOR(_i, n) { int s = I[_i];
            delta = numeric_limits<double>::infinity();
            FOR(i, NB_CLASS) {
                hat.node[s][i] = bar.node[s][i];
                for(auto e: G.graph[s])
                    hat.node[s][i] += M(G(s, e), e)[i];
                delta = min(delta, hat.node[s][i]);
            }
            FOR(i, NB_CLASS)
                hat.node[s][i] -= delta;
            Ebound += delta;
        //}
        int t;
        //FOR(e, m) {
            for(auto e: G.graph[s]) if(I.less(s, G(s, e))) {
                t = G(s, e);
                //if(I.less(G.edges[e].u, G.edges[e].v)) {
                //    s = G.edges[e].u;
                //    t = G.edges[e].v;
                //} else {
                //    s = G.edges[e].v;
                //    t = G.edges[e].u;
                //}
                vector<double>& Mst = M(s, e);
                vector<double>& Mts = M(t, e);
                delta = numeric_limits<double>::infinity();
                FOR(k, NB_CLASS) {
                    Mst[k] = numeric_limits<double>::infinity();
                    FOR(j, NB_CLASS)
                        Mst[k] = min(Mst[k], (T.gamma(s, t) * hat.node[s][j]
                                    - Mts[j]) + bar.edge[e][j][k]);
                    delta = min(delta, Mst[k]);
                }
                FOR(k, NB_CLASS)
                    Mst[k] -= delta;
                Ebound += delta;
            }
        }
        // Step 2
        I.reverse();

        // Stopping Criterion:
        // max_iter
        fprintf(stderr, "%lf\n", Ebound);
        if(Ebound - Elast < EPS)
            break;
        Elast = Ebound;
    }

    I.reversed = false;
    vector<int> x(n, -1), y(n, -1);
    double score, newScore;
    FOR(_i, n) { int s = I[_i];
        score = numeric_limits<double>::infinity();
        FOR(c, NB_CLASS) {
            newScore = hat.node[s][c];
            for(auto e: G.graph[s]) if(I.less(G(s, e), s)) {
                newScore += hat.edge[e][x[G(s, e)]][c];
            }
            if(newScore < score) {
                score = newScore;
                x[s] = c;
            }
        }
        score = -numeric_limits<double>::infinity();
        FOR(c, NB_CLASS) {
            newScore = P[s][c];
            if(newScore > score) {
                score = newScore;
                y[s] = c;
            }
        }
    }
    //FOR(i, n) printf("%d %d\n", x[i]+1, y[i]+1);
    FOR(i, n) printf("%d\n", x[i]+1);

    return 0;
}





