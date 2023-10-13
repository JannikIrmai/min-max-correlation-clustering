import numpy as np
import matplotlib.pyplot as plt
import DMN
from mmcc_approximation import MinMaxCorrelationClustering
from time import time
from itertools import product
from DMN import dmn_python
from mmcc_solver import solve_lp


def plot_quartile_curve(values, t, ax, color=None, name: str = None):

    median = np.median(values, axis=1)
    quart_1 = np.quantile(values, 1/4, axis=1)
    quart_3 = np.quantile(values, 3/4, axis=1)

    min_val = np.quantile(values, 1 / 10, axis=1)
    max_val = np.quantile(values, 9 / 10, axis=1)

    ax.plot(t, median, color=color, label=name)
    ax.fill_between(t, quart_1, quart_3, fc=color, alpha=0.3, ec=None)
    ax.plot(t, min_val, color=color, ls="dashed", lw=0.5)
    ax.plot(t, max_val, color=color, ls="dashed", lw=0.5)


def get_synth_instance(n: int, k: int, f: int):
    """
    :param n: number of nodes
    :param k: number of clusters (must divide n)
    :param f: number of flips
    :return:
    """
    assert n % k == 0

    c = n // k  # cluster size
    adj_mat = np.zeros((n, n), dtype=int)
    for i in range(k):
        adj_mat[i*c:(i+1)*c, i*c:(i+1)*c] = 1

    source, target = np.meshgrid(range(n), range(n))
    edges = np.array([source.flatten(), target.flatten()]).T
    edges = edges[edges[:, 0] < edges[:, 1]]
    flips = np.random.choice(range(edges.shape[0]), replace=False, size=f)

    for (u, v) in edges[flips]:
        adj_mat[u, v] = adj_mat[u, v] == 0
        adj_mat[v, u] = adj_mat[v, u] == 0

    return adj_mat


def dmn(adj_mat: np.ndarray, r1: float = 0.7, r2: float = 0.7):
    distances, l_t_vals, neighbors_r, neighbors_r2, clock, frac_val = DMN.exact(adj_mat, r1, r2)
    clustering, cluster_clock = DMN.cluster(distances, l_t_vals, neighbors_r, neighbors_r2, r1, r2)
    degrees = DMN.DegreeDist(adj_mat)
    _, alg_obj_val, _ = DMN.LocalObj(adj_mat, clustering, degrees, np.inf)

    return alg_obj_val


def main():
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

    n = 100
    k = 10
    num_flips = np.arange(0, 1001, 50, dtype=int)
    num_seeds = 10

    compute_lp_bound = False

    dmn_dis = []
    dmn_t = []
    dmn_python_dis = []
    dmn_python_t = []
    greedy_dis = []
    greedy_hyper_dis = []
    greedy_t = []
    greedy_hyper_t = []
    bound = []
    bound_t = []
    lp_bound = []
    lp_bound_t = []

    for f in num_flips:
        dmn_dis.append([])
        dmn_python_dis.append([])
        dmn_t.append([])
        dmn_python_t.append([])
        greedy_dis.append([])
        greedy_hyper_dis.append([])
        greedy_t.append([])
        greedy_hyper_t.append([])
        bound.append([])
        bound_t.append([])
        lp_bound.append([])
        lp_bound_t.append([])
        for seed in range(num_seeds):
            print(f"\rf = {f}, seed = {seed}", end="", flush=True)
            np.random.seed(seed)
            adj_mat = get_synth_instance(n, k, f)

            edges = np.argwhere(adj_mat)
            edges = edges[edges[:, 0] < edges[:, 1]]
            mmcc = MinMaxCorrelationClustering(edges)

            t_0 = time()
            dmn_dis[-1].append(mmcc.dmn(0.7, 0.7)[0])
            dmn_t[-1].append(time()-t_0)

            dis, t = dmn_python(edges, 0.7, 0.7)
            dmn_python_dis[-1].append(dis)
            dmn_python_t[-1].append(t)

            t_0 = time()
            bound[-1].append(mmcc.bound())
            bound_t[-1].append(time() - t_0)
            t_0 = time()
            greedy_dis[-1].append(mmcc.greedy_joining(**chosen_kwargs)[0])
            greedy_t[-1].append(time() - t_0)

            all_dis = []
            t_0 = time()
            for kwargs in greedy_kwargs:
                dis, clustering = mmcc.greedy_joining(**kwargs)
                all_dis.append(dis)
            greedy_hyper_dis[-1].append(min(all_dis))
            greedy_hyper_t[-1].append(time() - t_0)

            if compute_lp_bound:
                this_lp_bound, lp_time = solve_lp(edges)
                lp_bound[-1].append(this_lp_bound)
                lp_bound_t[-1].append(lp_time)


    fig, ax = plt.subplots(2, sharex=True)
    ax[1].set_xlabel(r"$f$")
    ax[0].set_ylabel("disagreement")
    ax[1].set_ylabel(r"$t$[s]")

    plot_quartile_curve(dmn_dis, num_flips, ax[0], color="tab:blue", name=f"DMN/DMN$^{{++}}$")
    # plot_quartile_curve(dmn_python_dis, num_flips, ax[0], color="tab:cyan", name="DMN")
    plot_quartile_curve(greedy_dis, num_flips, ax[0], color="tab:red", name=f"$\mathcal{{A}}$")
    plot_quartile_curve(greedy_hyper_dis, num_flips, ax[0], color="darkred", name=f"$\mathcal{{A}}^*$")
    plot_quartile_curve(bound, num_flips, ax[0], color="tab:green", name="CLB")
    if compute_lp_bound:
        plot_quartile_curve(lp_bound, num_flips, ax[0], color="tab:orange", name="LP")

    plot_quartile_curve(dmn_t, num_flips, ax[1], color="tab:blue", name=f"DMN$^{{++}}$")
    plot_quartile_curve(dmn_python_t, num_flips, ax[1], color="tab:cyan", name=f"DMN")
    plot_quartile_curve(greedy_t, num_flips, ax[1], color="tab:red", name=f"$\mathcal{{A}}$")
    plot_quartile_curve(greedy_hyper_t, num_flips, ax[1], color="darkred", name=f"$\mathcal{{A}}^*$")
    plot_quartile_curve(bound_t, num_flips, ax[1], color="tab:green", name="CLB")
    if compute_lp_bound:
        plot_quartile_curve(lp_bound_t, num_flips, ax[1], color="tab:orange", name="LP")

    ax[0].legend()
    ax[1].set_yscale('log')
    ax[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

