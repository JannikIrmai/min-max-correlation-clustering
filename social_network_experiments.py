from mmcc_approximation import MinMaxCorrelationClustering
from DMN import dmn_python
import data_loading
from itertools import product
from time import time
from mmcc_solver import solve_lp


def main():

    facebook = True
    if facebook:
        numbers = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    else:
        numbers = [0, 1, 2, 3]

    # arguments for the greedy joining algorithm A
    chosen_kwargs = {
        "allow_cluster_joining": True,
        "allow_increase": False,
        "max_dis_largest_degree": True,
        "neighbor_smallest_degree": True,
        "require_worst_improvement": False,
        "symm_diff_factor": 1
    }

    # combinations of arguments for the greedy joining algorithm A*
    arguments = {
        "allow_cluster_joining": [True],
        "allow_increase": [True, False],
        "max_dis_largest_degree": [True, False],
        "neighbor_smallest_degree": [True, False],
        "require_worst_improvement": [False],
        "symm_diff_factor": [0, 1, 10000]
    }
    argument_names = list(arguments.keys())
    argument_values = [arguments[name] for name in argument_names]
    greedy_kwargs = []
    for args in product(*argument_values):
        greedy_kwargs.append({name: val for name, val in zip(argument_names, args)})

    max_num_nodes_lp = 250
    max_num_nodes_dmn_python = 15000

    if max_num_nodes_lp > 0:
        # call solver once such that the license print out is before table header
        solve_lp([[0, 1]])
        print()

    # print the header of the table
    print(r"  ID        |V|        |E|   deg CLB    LP     A    A*  DMN    t_CLB      t_LP     t_A    t_A*    "
          r"t_DMN t_DMN++")

    for graph_id in numbers:
        # load the edges of the graph
        if facebook:
            edges = data_loading.load_facebook(graph_id)
        else:
            if graph_id == 0:
                edges = data_loading.load_lastfm()
            elif graph_id == 1:
                edges = data_loading.load_ca_hep_ph()
            elif graph_id == 2:
                edges = data_loading.load_ca_hep_th()
            else:
                edges = data_loading.load_youtube()

        mmcc = MinMaxCorrelationClustering(edges)
        max_deg = mmcc.max_degree()

        t_before_clb = time()
        if mmcc.num_nodes() < 100000:
            bound = mmcc.bound()
        else:
            bound = -1
        clb_time = time() - t_before_clb

        if mmcc.num_nodes() < 100000:
            t_before_dmn = time()
            dmn, _ = mmcc.dmn(0.7, 0.7)
            t_dmn = time() - t_before_dmn
        else:
            dmn = 0
            t_dmn = 0

        if mmcc.num_nodes() < max_num_nodes_dmn_python:
            dmn_p, t_dmn_python = dmn_python(edges, 0.7, 0.7)
            assert dmn == dmn_p
        else:
            t_dmn_python = 0

        t_before_a_star = time()
        dis_holger = []
        for kwargs in greedy_kwargs:
            dis, clustering = mmcc.greedy_joining(**kwargs)
            dis_holger.append(dis)
        t_a_star = time() - t_before_a_star

        min_dis = min(dis_holger)

        if mmcc.num_nodes() < max_num_nodes_lp:
            lp_bound, lp_time = solve_lp(edges)
        else:
            lp_bound = None
            lp_time = None

        t_before_greedy = time()
        dis, _ = mmcc.greedy_joining(**chosen_kwargs)
        t_greedy = time() - t_before_greedy

        print(f"{graph_id:>4} {mmcc.num_nodes():>10} {mmcc.num_edges():>10} {max_deg:>5} "
              f"{bound:>3} {f'{lp_bound:.2f}' if lp_bound else '-':>5} "
              f"{dis:>5} {min_dis:>5} {dmn:>4} "
              f"{clb_time*1000:>8.2f} {f'{lp_time*1000:.1f}' if lp_time else '-':>9} "
              f"{t_greedy*1000:>7.2f} {t_a_star*1000:>7.2f} "
              f"{t_dmn_python*1000:>8.2f} {t_dmn*1000:>7.2f}")


if __name__ == "__main__":
    main()
