import gurobipy as gp
from gurobipy import GRB as GRB
import networkx as nx
from itertools import combinations
from time import time


class MinMaxCorrelationClusteringSolver:

    def __init__(self, g: nx.Graph, binary: bool = False,
                 lazy_triangle: bool = False, suppress_log: bool = False,
                 add_partial_optimality_constraints: bool = False,
                 method=GRB.METHOD_BARRIER):

        self.graph = g
        self.model = gp.Model()

        if suppress_log:
            self.model.Params.LogToConsole = 0

        self.model.setParam("method", method)

        # add variables
        self.edge_vars = self.model.addVars([e for e in combinations(self.graph.nodes, 2)], lb=0, ub=1,
                                            vtype=GRB.BINARY if binary else GRB.CONTINUOUS, name="edge")
        for (u, v), var in self.edge_vars.items():
            self.edge_vars[v, u] = var
        self.disagreement_vars = self.model.addVars(self.graph.nodes, lb=0, ub=self.graph.number_of_nodes() - 1,
                                                    name="disagreement")
        self.max_disagreement_var = self.model.addVar(0, self.graph.number_of_nodes() - 1, obj=1,
                                                      name="max_disagreement")

        # add constraints that asserts that max_disagreement_var is at least the maximum over all disagreements
        for var in self.disagreement_vars.values():
            self.model.addConstr(var <= self.max_disagreement_var, "max_disagreement")

        # add constraints that correctly compute the disagreement per node
        self.disagreement_inequalities_added = False
        self.add_disagreement_inequalities()

        # add triangle inequalities
        self.triangle_inequalities_added = False
        if lazy_triangle:
            self.model.Params.lazyConstraints = 1
            if not binary:
                self.model.addVar(0, 1, 0, GRB.BINARY, "dummy")
        else:
            self.add_triangle_inequalities()

        self.quadratic_bound_vars = None
        if add_partial_optimality_constraints:
            self.add_partial_optimality_constraints()

    def add_triangle_inequalities(self):
        for u, v in combinations(self.graph.nodes, 2):
            for w in self.graph.nodes():
                if w in [u, v]:
                    continue
                self.model.addConstr(self.edge_vars[(u, v)] <= self.edge_vars[u, w] + self.edge_vars[v, w],
                                     "triangle")
        self.triangle_inequalities_added = True

    def add_partial_optimality_constraints(self):
        # let y be the variable that describes the maximal disagreement of the clustering
        # For each pair of nodes uv we introduce a quadratic variable z_uv = x_uv * y
        max_deg = max([self.graph.degree(n) for n in self.graph.nodes])
        self.quadratic_bound_vars = self.model.addVars(
            [e for e in combinations(self.graph.nodes, 2)],
            lb=0, ub=max_deg-1, vtype=GRB.CONTINUOUS, name="quadratic_bound")

        for (u, v), var in self.quadratic_bound_vars.items():
            # linearize z = x_uv * y
            self.model.addConstr(var <= (max_deg - 1) * self.edge_vars[u, v], "quadratic_bound_lin_1")
            self.model.addConstr(var <= self.max_disagreement_var, "quadratic_bound_lin_2")
            self.model.addConstr(var >= self.max_disagreement_var - (max_deg - 1) * (1 - self.edge_vars[u, v]),
                                 "quadratic_bound_lin_3")
            # add the constraint (1 - x_uv) |N_u \triangle N_v| <= 2 y - 2 (x_uv * y)
            n_u = set(self.graph.neighbors(u))
            n_u.add(u)
            n_v = set(self.graph.neighbors(v))
            n_v.add(v)
            sym_diff = len(n_u.symmetric_difference(n_v))
            self.model.addConstr((1-self.edge_vars[(u, v)]) * sym_diff <= 2 * self.max_disagreement_var - 2 * var,
                                 "must_cut")
            intersection = len(n_u.intersection(n_v))
            self.model.addConstr(self.edge_vars[u, v] * intersection <= 2 * var,
                                 "must_join")

    def add_disagreement_inequalities(self):
        for n in self.graph.nodes:
            ineq = gp.quicksum(self.edge_vars[n, v] if self.graph.has_edge(n, v) else
                               (1 - self.edge_vars[n, v]) for v in self.graph.nodes if n != v) \
                   <= self.disagreement_vars[n]
            self.model.addConstr(ineq, f"disagreement_[{n}]")
        self.disagreement_inequalities_added = True

    def callback(self, _, where):
        if where == GRB.callback.MIPSOL:
            self.callback_mipsol()

    def callback_mipsol(self):
        if self.triangle_inequalities_added:
            return
        edge_values = self.model.cbGetSolution(self.edge_vars)
        g = nx.Graph()
        g.add_nodes_from(self.graph.nodes)
        g.add_edges_from([e for e, x_e in edge_values.items() if x_e < 1/2])
        node_labels = {}
        for i, comp in enumerate(nx.connected_components(g)):
            for n in comp:
                node_labels[n] = i
        counter = 0
        for (u, v), x_e in edge_values.items():
            if x_e > 1/2 and node_labels[u] == node_labels[v]:
                for n in g.neighbors(u):
                    if g.has_edge(v, n):
                        self.model.cbLazy(self.edge_vars[(u, v)] <= self.edge_vars[(u, n)] + self.edge_vars[(v, n)])
                        counter += 1
                        break
        print("added", counter, "triangle inequalities")

    def solve(self):

        def callback(*args):
            self.callback(*args)

        self.model.optimize(callback)

        return self.model.getAttr("X", self.edge_vars), \
            self.model.getAttr("X", [self.max_disagreement_var])[0]


def solve_lp(edges):
    graph = nx.Graph()
    for u, v in edges:
        graph.add_edge(u, v)
    mmcc_lp = MinMaxCorrelationClusteringSolver(graph, binary=False, suppress_log=True)
    t_0 = time()
    _, lp_bound = mmcc_lp.solve()
    t = time() - t_0
    return lp_bound, t

