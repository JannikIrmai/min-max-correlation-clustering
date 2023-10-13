#include <vector>
#include <array>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <queue>
#include <numeric>

/**
 * @brief This methods computes the union of two sorted containers a and b. 
 * The union in computed in place in the container a
 * 
 * @tparam CONTAINER 
 * @param a 
 * @param b 
 */
template<class CONTAINER>
void inplace_union(CONTAINER& a, CONTAINER& b)
{
    size_t mid = a.size();
    a.reserve(a.size() + b.size());
    // insert the contents of b to the end of a
    a.insert(a.end(), b.begin(), b.end());
    // sort the merged container in place
    std::inplace_merge(a.begin(), a.begin() + mid, a.end());
    // remove duplicate elements
    a.erase(std::unique(a.begin(), a.end()), a.end());
}


template<class CONTAINER>
size_t intersection_size(const CONTAINER& a, const CONTAINER& b){
    auto it_a = a.begin();  // iterator over the neighbors of i
    auto it_b = b.begin();  // iterator over the neighbors of j
    size_t i = 0;
    while ((it_a != a.end()) && (it_b != b.end()))
    {
        if (*it_a == *it_b)
        {
            ++i;
            ++it_a;
        } else if (*it_a < *it_b) {
            ++it_a;
        } else {
            ++it_b;
        }
    }
    return i;
}


class AdjacencyGraph
{
public:
    AdjacencyGraph(size_t n): adjacency_(n) {}

    void contract (size_t u, size_t v)
    {
        for (size_t w : adjacency_[v])
        {
            adjacency_[w].erase(v);
            adjacency_[w].insert(u);
        }
        adjacency_[u].insert(adjacency_[v].begin(), adjacency_[v].end());
        adjacency_[v].clear();
    }

    bool is_edge(size_t u, size_t v) const
    {
        if (adjacency_[u].size() <= adjacency_[v].size())
            return adjacency_[u].count(v);
        else
            return adjacency_[v].count(u);
    }

    void add_edge(size_t u, size_t v)
    {
        adjacency_[u].insert(v);
        adjacency_[v].insert(u);
    }

    const std::set<size_t>& neighbors(size_t u) const
    {
        return adjacency_[u];
    }

private: 
    std::vector<std::set<size_t>> adjacency_;
};


template<class NODE>
class MinMaxCorrelationClustering
{
public:

    typedef std::vector<std::array<NODE, 2>> EDGES;

    MinMaxCorrelationClustering(const EDGES& edges)
    {
        build_graph(edges);
    }

    void print_graph(bool by_index = true)
    {
        for (size_t i = 0; i < graph_.size(); ++i)
        {
            if (by_index)
                std::cout << i << " : ";
            else
                std::cout << idx2node_[i] << " : ";

            for (size_t j_idx = 0; j_idx < graph_[i].size(); ++j_idx)
            {
                size_t j = graph_[i][j_idx];
                if (by_index)
                    std::cout << j << " ";
                else 
                    std::cout << idx2node_[j] << " ";
            }
            std::cout << "\n";
        }
    }

    size_t num_nodes() const
    {
        return graph_.size();
    }

    size_t num_edges() const
    {
        size_t e = 0;
        for (size_t i = 0; i < graph_.size(); ++i)
            e += graph_[i].size();
        return e / 2;
    }

    size_t max_degree()
    {
        size_t d = 0;
        for (size_t i = 0; i < graph_.size(); ++i)
            if (graph_[i].size() > d)
                d = graph_[i].size();
        return d;
    }

    /**
     * @brief This is an alternative implementation of the combinatorial lower bound without a bisection algorithm.
     *  Instead a lower bound candidate d is initialized to the number of nodes and iteratively decrease by one
     *  until the lower bounding technique implies that there cannot exists a partition with maximum disagreement d.
     *  While this results in a linear number of iterations (compared to a logarithmic number for the other implementation),
     *  this implementation has the advantage that information can be shared between iterations. In contrast to this,
     *  in the other implementation the bound criterion needs to be reevaluated in each iteration from scratch.
     * 
     * @return size_t 
     */
    size_t combinatorial_lower_bound_alt()
    {
        // compute the intersection sizes for all pairs of nodes
        std::map<std::pair<size_t, size_t>, size_t> intersection_sizes = all_intersection_sizes();

        // vector where the entry at position i maps to all pairs of nodes whose neighborhoods intersect in i nodes.        
        std::vector<std::vector<std::pair<size_t, size_t>>> intersection_size_to_pairs(graph_.size() + 1);
        // vector where the entry at position i maps to all pairs of nodes whose neighborhoods have symmetric difference of i
        std::vector<std::vector<std::pair<size_t, size_t>>> symm_diff_to_pairs(graph_.size() + 1);

        for (auto it = intersection_sizes.begin(); it != intersection_sizes.end(); ++it)
        {
            intersection_size_to_pairs[it->second].push_back(it->first);
            // compute the symmetric difference of the neighborhoods of u and v as
            // |N_u| + |N_v| - 2 * |N_u \cap N_v|
            // The +2 in the line below is to account for the fact that we consider a node to be part of its own neighborhood
            size_t symm_diff = graph_[it->first.first].size() + graph_[it->first.second].size() + 2 - 2 * it->second;
            symm_diff_to_pairs[symm_diff].push_back(it->first);
        }

        // add pairs of nodes that have intersection 0 to sym diff
        // todo: avoid the quadratic memory need
        for (size_t i = 0; i < graph_.size(); ++i)
            for (size_t j = i+1; j < graph_.size(); ++j)
                if (intersection_sizes.count({i, j}) == 0)
                    symm_diff_to_pairs[graph_[i].size() + graph_[j].size() + 2].push_back({i, j});

        // start with the singleton clustering
        std::vector<std::vector<size_t>> clustering(graph_.size());
        std::vector<size_t> node2cluster(graph_.size());
        for (size_t i = 0; i < graph_.size(); ++i)
        {
            clustering[i] = {i};
            node2cluster[i] = i;
        }
        // vector that flags all clusters that are updated 
        AdjacencyGraph must_cut_graph(graph_.size());

        // For every partition Pi with maximum disagreement d and for all pairs u, v \in V it holds that
        //  (a)  |N_u \cap N_v| > 2*d       =>  [u]_Pi  = [v]_Pi
        //  (b)  |N_u \triangle N_v| > 2*d  =>  [u]_Pi != [v]_Pi 
        // instead of testing ""> 2*d" we test ">= rhs" with rhs >= 2*d+1
        for (size_t rhs = graph_.size(); rhs > 0; --rhs)
        {
            size_t d = (rhs - 1) / 2;
            // join the clusters of all pairs of nodes whose neighborhoods intersect in rhs nodes
            for (auto& edge : intersection_size_to_pairs[rhs])
            {
                size_t u = edge.first;
                size_t v = edge.second;
                // join the clusters that contain u and v
                if (node2cluster[u] == node2cluster[v])  // u and v are already in the same cluster
                    continue;
                if (must_cut_graph.is_edge(node2cluster[u], node2cluster[v]))
                    return d + 1;
                if (clustering[u].size() < clustering[v].size())
                    std::swap(u, v);  // swap so that the smaller cluster is joined to the larger
                size_t cu = node2cluster[u];
                size_t cv = node2cluster[v];
                
                // by joining cu and cv, we contract the nodes corresponding to cu and cv in the must cut graph
                must_cut_graph.contract(cu, cv);

                // add the cluster of v to the cluster of u
                inplace_union(clustering[cu], clustering[cv]);
                // update the node2cluster vector for all nodes in the cluster cv
                for (size_t w : clustering[cv])
                    node2cluster[w] = cu;
                clustering[cv].clear();
            }

            // add must cuts between all nodes whose neighborhoods have a symmetric difference greater than rhs
            for (auto& edge : symm_diff_to_pairs[rhs])
            {
                size_t u = edge.first;
                size_t v = edge.second;
                if (node2cluster[u] == node2cluster[v])
                    return d+1;
                must_cut_graph.add_edge(node2cluster[u], node2cluster[v]);
            }

            // compute the disagreement bound
            for (size_t u = 0; u < graph_.size(); ++u)
            {
                size_t dis = 0;
                // increase the disagreement by one for each neighbor of u that must be in a different cluster than u
                for (size_t v : graph_[u])
                    if (must_cut_graph.is_edge(node2cluster[u], node2cluster[v]))
                        ++dis;
                // increase the disagreement by one for each node in the cluster of u that is not a neighbor of u
                dis += clustering[node2cluster[u]].size() - intersection_size(graph_[u], clustering[node2cluster[u]]) - 1;
                // terminate if disagreement is to large
                if (dis > d)
                    return d + 1;
            }
        }
        return 0;
    }

    /**
     * @brief This method implements a bisection algorithm for computing the combinatorial lower bound for the min max 
     *  correlation clustering problem.
     * 
     * @return size_t 
     */
    size_t combinatorial_lower_bound()
    {
        // compute the size of the intersection of the neighborhoods of all pairs of nodes
        std::vector<std::vector<size_t>> intersection_sizes(graph_.size(), std::vector<size_t>(graph_.size(), 0));
        for (size_t i = 0; i < graph_.size(); ++i)
        {
            for (auto it1 = graph_[i].begin(); it1 != graph_[i].end(); ++it1)
            {
                intersection_sizes[i][*it1] += 2;
                for (auto it2 = it1 + 1; it2 != graph_[i].end(); ++it2) 
                {
                    ++intersection_sizes[*it1][*it2];
                    ++intersection_sizes[*it2][*it1];
                }
            }
        }

        int lower_bound = -1;  // highest known value for which there cannot exists a clustering that achieves that disagreement
        size_t upper_bound = 0;  // lowest known value for which there might exists a clustering that achieves that disagreement
        for (size_t u = 0; u < graph_.size(); ++u)
            if (graph_[u].size() > upper_bound)
                upper_bound = graph_[u].size();

        // start of the bisection algorithm
        while (lower_bound + 1 < upper_bound)
        {
            size_t bound = (lower_bound + upper_bound) / 2;
            // compute the connected components of the graph whose edges are all pairs
            // of nodes whose neighborhoods intersect in at least bound nodes
            std::vector<size_t> components(graph_.size(), 0);
            std::queue<size_t> queue;
            size_t comp_idx = 0;
            for (size_t u = 0; u < graph_.size(); ++u)
            {
                if (components[u] > 0)
                    continue;  // u was already visited
                ++comp_idx;
                components[u] = comp_idx;
                queue.push(u);
                while (!queue.empty())
                {
                    size_t v = queue.front();
                    queue.pop();
                    for (size_t w = 0; w < graph_.size(); ++w)
                    {
                        if (components[w] > 0)
                            continue;  // w was already visited
                        if (intersection_sizes[v][w] <= 2*bound)
                            continue;
                        components[w] = comp_idx;
                        queue.push(w);
                    }
                }
            }
            // store the connected components in clusters
            std::vector<std::vector<size_t>> clustering(comp_idx);
            for (size_t u = 0; u < graph_.size(); ++u)
                clustering[components[u] - 1].push_back(u);
            
            bool any_dis_too_large = false;
            for (size_t c = 0; c < clustering.size(); ++c)
            {
                std::vector<size_t> cluster_complement;
                for (size_t other_c = 0; other_c < clustering.size(); ++other_c)
                {
                    if (other_c == c)
                        continue;
                    bool any_symm_diff_large = false;
                    for (size_t u : clustering[c])
                    {
                        for (size_t v : clustering[other_c])
                        {
                            size_t symm_diff = graph_[u].size() + graph_[v].size() + 2 - 2 * intersection_sizes[u][v];
                            if (symm_diff > 2 * bound)
                            {
                                any_symm_diff_large = true;
                                break;
                            }
                        }
                        if (any_symm_diff_large)
                            break;
                    }
                    if (any_symm_diff_large)
                        inplace_union(cluster_complement, clustering[other_c]);
                }
                // compute the disagreement bound for all nodes in the cluster c
                for (auto u : clustering[c])
                {
                    size_t dis = intersection_size(graph_[u], cluster_complement) 
                        + clustering[c].size() - intersection_size(graph_[u], clustering[c]) - 1;
                    if (dis > bound)
                    {
                        any_dis_too_large = true;
                        break;
                    }
                }
                if (any_dis_too_large)
                    break;
            }
            if (any_dis_too_large)
                lower_bound = bound;
            else
                upper_bound = bound;
        }
        return upper_bound;
    }

    /**
     * @brief This method implements the greedy joining algorithm:
     *  1. start with singleton clustering
     *  2. Select node w with largest disagreement
     *  3. Select a neighbor v of w such that 
     *      a) joining the clusters of v and w improves the current solution
     *      b) the intersection of the neighborhoods of v and w is maximal among all v that satisfy a)
     *  4. If such a node v exists, join the clusters of v and w and proceed with step 2. Otherwise, terminate
     * 
     * In this algorithm there are several arbitrary choices one can make. The arguments of this methods allow to
     * specify different choices.
     * 
     * @param allow_cluster_joining Flag that indicates whether two clusters can be joined if neither of them is a 
     *  singleton cluster. This is purely for performance reasons as joining non singleton clusters may be computationally more expensive.
     * @param allow_increase Flag that indicates whether a join is allowed if a vertex whose disagreement was less than the maximum
     *  disagreement becomes equal to the current maximum disagreement.
     * @param max_dis_largest_degree Flag that indicates whether in step 2 among all nodes with largest disagreement the nodes
     *  with largest degree shall be selected. Otherwise the node with smallest degree is detected. 
     *  Further ties are resolved by selecting the node with the smallest id first.
     * @param neighbor_smallest_degree Flag that indicates whether among all neighbors of the worst node w that satisfy a) and b) the one
     *  with the smallest degree shall be chosen. Otherwise the one with the largest degree is chosen. In case of further ties, 
     *  the node with the smallest id is chosen.
     * @param require_worst_improvement Flag that indicates whether it is required that the disagreement of the worst node improves
     *  in each iteration.
     * @param symm_diff_factor Instead of sorting neighborhoods just by intersection size we sort it by intersection size minus symmetric
     *  difference. Where the symmetric difference is multiplied by symm_diff_factor. I.e. if symm_diff_factor = 0, we just sort by intersection
     *  size. The greater symm_diff_factor is, the more symm_diff_factor is considered.
     * @return std::pair<size_t, std::vector<std::vector<NODE>>> 
     *  This method returns the disagreement of the computed partition as well as the partition.
     */
    std::pair<size_t, std::vector<std::vector<NODE>>> greedy_joining(
        bool allow_cluster_joining = true,
        bool allow_increase = true,
        bool max_dis_largest_degree = true,
        bool neighbor_smallest_degree = true,
        bool require_worst_improvement = true,
        double symm_diff_factor = 0
    )
    {
        // As we start with the singleton clustering, the disagreement of a node is equal to its degree.
        std::vector<size_t> disagreement(graph_.size());
        std::vector<size_t> node2cluster(graph_.size());
        std::vector<std::vector<size_t>> clustering(graph_.size());
        for (size_t i = 0; i < graph_.size(); ++i)
        {
            disagreement[i] = graph_[i].size();
            node2cluster[i] = i;
            clustering[i] = {i};
        }

        std::vector<int> disagreement_change(graph_.size());
        auto update_disagreement_join = [&clustering, &node2cluster, &disagreement_change, &graph_ = this->graph_] (size_t c1, size_t c2)
        {
            // if the cluster c1 contains more nodes than the cluster c2, swap the clusters
            if (clustering[c1].size() > clustering[c2].size())
                std::swap(c1, c2);

            // for each node in c1 and c2 increase the disagreement by the size of the other cluster
            // based on the assumption that the node is not connected to any node in the other cluster.
            for (size_t v : clustering[c1])
                disagreement_change[v] = clustering[c2].size();
            for (size_t v : clustering[c2])
                disagreement_change[v] = clustering[c1].size();
            
            // decrease the disagreements of all nodes by accounting for edges between the clusters
            for (size_t v : clustering[c1])
            {
                // for each node v in the cluster c1, iterate over its neighbors nv.
                // If the neighbor nv is in the cluster c2, decrease the disagreement of 
                // v and nv by two.
                auto nv_it = graph_[v].begin();
                auto w_it = clustering[c2].begin();
                while (nv_it != graph_[v].end() && w_it != clustering[c2].end())
                {
                    if (*nv_it == *w_it) {
                        disagreement_change[*nv_it] -= 2;
                        disagreement_change[v] -= 2;
                        ++nv_it;
                    }
                    else if (*nv_it < *w_it) {
                        ++nv_it;
                    }
                    else {
                        ++w_it;
                    }
                }
            }
        };

        // This function is used selecting the next node with largest disagreement.
        auto priority_compare = [&graph_ = this->graph_, &disagreement, &max_dis_largest_degree] (size_t i, size_t j)
        {
            if (disagreement[i] == disagreement[j]) {
                if (graph_[i].size() == graph_[j].size()){
                    return i < j;
                }
                if (max_dis_largest_degree){
                    return graph_[i].size() > graph_[j].size();
                }
                return graph_[i].size() < graph_[j].size();
            } 
            else {
                return disagreement[i] > disagreement[j];
            }
        };
        
        std::set<size_t, decltype(priority_compare)> queue(priority_compare);
        for (size_t i = 0; i < graph_.size(); ++i)
            queue.insert(i);

        auto pop_queue = [&queue] ()
        {
            auto it = queue.begin();
            size_t i = *it;
            queue.erase(it);
            return i;
        };

        // neighborhoods sorted by intersection size
        std::vector<std::vector<size_t>> sorted_neighborhoods(graph_.size());
        std::vector<bool> neighborhood_computed(graph_.size(), false);

        // fill priority queue with nodes where the priority is given by the disagreement 
        size_t max_dis = 0;
        while (!queue.empty())
        {
            size_t w = pop_queue();
            max_dis = disagreement[w];

            if (!neighborhood_computed[w])
            {
                neighborhood_computed[w] = true;
                // For each neighbor v of w compute the size of the intersection of the neighborhoods of w and v
                std::vector<size_t> intersection_sizes = compute_neighborhood_intersections_of_neighbors(w);
                std::vector<double> sort_criterion(intersection_sizes.size());
                for (size_t i = 0; i < graph_[w].size(); ++i){
                    sort_criterion[i] = intersection_sizes[i] - symm_diff_factor * (2 + graph_[w].size() + graph_[graph_[w][i]].size() - 2 * intersection_sizes[i]);
                }


                // sort the neighbors by intersection size
                std::vector<size_t> idx(graph_[w].size());
                std::iota(idx.begin(), idx.end(), 0);
                std::stable_sort(idx.begin(), idx.end(),
                    [&sort_criterion, &w, &graph_ = this->graph_, &neighbor_smallest_degree](size_t i, size_t j) { 
                        if (sort_criterion[i] == sort_criterion[j]){
                            size_t u = graph_[w][i];
                            size_t v = graph_[w][j];
                            if (graph_[u].size() == graph_[v].size()){
                                return u > v;
                            }
                            if (neighbor_smallest_degree)
                                return graph_[u].size() > graph_[v].size();
                            return graph_[u].size() < graph_[v].size();
                        }
                        return sort_criterion[i] < sort_criterion[j]; 
                    });
                sorted_neighborhoods[w].resize(graph_[w].size());
                for (size_t i = 0; i < graph_[w].size(); ++i)
                    sorted_neighborhoods[w][i] = graph_[w][idx[i]];
            }

            bool improved = false;
            while (sorted_neighborhoods[w].size() > 0)
            {
                size_t v = sorted_neighborhoods[w][sorted_neighborhoods[w].size() - 1];
                sorted_neighborhoods[w].pop_back();
                // if v and w are already in the same cluster, continue
                size_t cv = node2cluster[v];
                size_t cw = node2cluster[w];
                if (cv == cw)
                    continue;
                // if cluster joining is not allowed and both cv and cv contain more than one element, continue
                if ((!allow_cluster_joining) && (clustering[cw].size() > 1) && (clustering[cv].size() > 1))
                    continue;
                // update the disagreement of the nodes in the clusters of v and w if the clusters were joined
                update_disagreement_join(cv, cw);
                
                // flag that indicates whether the join is valid. A joins is valid if the disagreements
                // of the nodes in the joined cluster is not to large (depends on input flags)
                bool valid_join = disagreement[w] + disagreement_change[w] + require_worst_improvement <= max_dis;  
                for (size_t c : {cv, cw}){
                    for (size_t u : clustering[c]){
                        size_t new_dis = disagreement[u] + disagreement_change[u];
                        if (new_dis > max_dis)
                            valid_join = false;
                        if (!allow_increase && (new_dis == max_dis) && (disagreement[u] < max_dis))
                            valid_join = false;
                        if (!valid_join)
                            break;
                    }
                    if (!valid_join)
                            break;
                }
                if (!valid_join)
                    continue;

                // join the clusters of v and w
                // std::cout << "joining " << w << " " << v << "\n";
                inplace_union(clustering[cw], clustering[cv]);
                // update the node2cluster vector for all nodes in the cluster cv
                for (size_t u : clustering[cv])
                    node2cluster[u] = cw;
                clustering[cv] = {};
                // update the priority queue
                for (size_t vv : clustering[cv]) {
                    if (disagreement_change[vv] != 0) {
                        queue.erase(vv);
                        disagreement[vv] += disagreement_change[vv];
                        queue.insert(vv);
                    }
                }
                for (size_t ww : clustering[cw]){
                    if (disagreement_change[ww] != 0) {
                        queue.erase(ww);
                        disagreement[ww] += disagreement_change[ww];
                        queue.insert(ww);
                    }
                }
                improved = true;
                break;
            }
            if (!improved)
                break;
        }

        // extract the clustering in terms of original node ids
        std::map<size_t, size_t> cluster_map;
        size_t cluster_idx = 0;
        std::vector<std::vector<NODE>> node_clustering;
        for (size_t i = 0; i < graph_.size(); ++i)
        {
            size_t c = node2cluster[i];
            if (cluster_map.count(c) == 0)
            {
                cluster_map[c] = cluster_idx;
                node_clustering.push_back({});
                ++cluster_idx;
            }
            node_clustering[cluster_map[c]].push_back(idx2node_[i]);
        }
        return {max_dis, node_clustering};
    }

    /**
     * @brief This method implements the algorithm of Davies et al. (2023).
     *  In comparison to the original implementation, we do not compute the semi-metric 
     *  by squaring the adjacency matrix. Instead, we exploit the fact that in practical
     *  instances the graph of positive edges is rather sparse. We only compute
     *  the semi metric for those pairs of nodes for which it is not 1, i.e. those pairs
     *  of nodes who have at least one common neighbor.
     * 
     * @param r1 
     * @param r2 
     * @return std::pair<size_t, std::vector<std::vector<NODE>>> 
     */
    std::pair<size_t, std::vector<std::vector<NODE>>> dmn(double r1, double r2)
    {
        if (r1 <= 0 || r2 > r1 || 1 <= r2)
            throw std::runtime_error("Radii invalid! It must hold that 0 < r1 <= r2 < 1.");

        std::map<std::pair<size_t, size_t>, size_t> intersection_sizes = all_intersection_sizes();
        std::vector<double> l_values(graph_.size(), 0);
        std::vector<std::vector<size_t>> balls1(graph_.size());
        std::vector<std::vector<double>> balls1_semi_metric(graph_.size());
        std::vector<std::vector<size_t>> balls2(graph_.size());

        for (size_t i = 0; i < graph_.size(); ++i)
        {
            balls1[i].push_back(i);
            balls1_semi_metric[i].push_back(0);
            balls2[i].push_back(i);
        }

        for (auto& it : intersection_sizes)
        {
            size_t u = it.first.first;
            size_t v = it.first.second;
            double denom = graph_[u].size() + 1 + graph_[v].size() + 1 - it.second;
            double semi_metric = 1 - (double)it.second / denom;

            if (semi_metric <= r1)
            {
                l_values[u] += r1 - semi_metric;
                l_values[v] += r1 - semi_metric;
                balls1[u].push_back(v);
                balls1_semi_metric[u].push_back(semi_metric);
                balls1[v].push_back(u);
                balls1_semi_metric[v].push_back(semi_metric);
            }
            if (semi_metric <= r2)
            {
                balls2[u].push_back(v);
                balls2[v].push_back(u);
            }
        }

        std::vector<size_t> clustering(graph_.size(), 0);
        size_t num_not_clustered = graph_.size();
        size_t cluster_idx = 0;
        while (num_not_clustered > 0)
        {
            ++cluster_idx;
            // select the node with the largest l-value
            size_t u = std::distance(l_values.begin(), std::max_element(l_values.begin(), l_values.end()));
            // assert that u is not already assigned to a cluster
            if (clustering[u] != 0)
                throw std::runtime_error("node with largest l-value is already clustered!");

            for (size_t v : balls2[u])
            {
                if (clustering[v] != 0)
                    continue;

                l_values[v] = -1;  // v is now clustered, so it should not be clustered again
                --num_not_clustered;
                clustering[v] = cluster_idx;
                // adjust the l-values of all nodes in the r1 ball of v
                for (size_t i = 0; i < balls1[v].size(); ++i)
                    if (clustering[balls1[v][i]] == 0) 
                        l_values[balls1[v][i]] -= r1 - balls1_semi_metric[v][i];
            }
        }
        
        // compute the clusters (once in terms of node indices and once in terms of original node names)
        std::vector<std::vector<size_t>> idx_clustering(cluster_idx);
        std::vector<std::vector<NODE>> node_clustering(cluster_idx);
        for (size_t i = 0; i < graph_.size(); ++i)
        {
            idx_clustering[clustering[i]-1].push_back(i);
            node_clustering[clustering[i]-1].push_back(idx2node_[i]);
        }

        // compute the maximum disagreement
        size_t max_dis = 0;
        for (size_t i = 0; i < graph_.size(); ++i)
        {   
            // compute disagreement of node i with its cluster
            size_t dis = graph_[i].size() + idx_clustering[clustering[i]-1].size() - 1 - 2 * intersection_size(graph_[i], idx_clustering[clustering[i]-1]);
            if (dis > max_dis)
                max_dis = dis;
        }

        return {max_dis, node_clustering};
    }

private:
    std::vector<NODE> idx2node_;
    std::vector<std::vector<size_t>> graph_;

    void build_graph(const EDGES& edges)
    {
        // compute the degree of each node
        std::map<NODE, size_t> degree_map;
        for (const std::array<NODE, 2>& edge : edges)
            for (const size_t& v : edge)
                ++degree_map[v];
        
        // sort the nodes by descending degree where ties are resolved by comparing the original node value
        std::vector<std::pair<NODE, size_t>> node_degree_pairs;
        node_degree_pairs.reserve(degree_map.size());
        for (auto it = degree_map.begin(); it != degree_map.end(); ++it)
            node_degree_pairs.push_back({it->first, it->second});

        std::sort(node_degree_pairs.begin(), node_degree_pairs.end(), 
            [=](std::pair<NODE, size_t>& a, std::pair<NODE, size_t>& b)
            {
                if (a.second > b.second)
                    return true;
                else if (a.second == b.second)
                    return a.first < b.first;
                else
                    return false;
            }
        );

        // map nodes to indices enumerated according to the sorted node order
        idx2node_.resize(node_degree_pairs.size());
        std::map<NODE, size_t> node2idx;
        for (size_t i = 0; i < node_degree_pairs.size(); ++i)
        {
            idx2node_[i] = node_degree_pairs[i].first;
            node2idx[node_degree_pairs[i].first] = i;
        }

        // build graph based on node indices
        graph_ = std::vector<std::vector<size_t>>(idx2node_.size());
        for (const std::array<NODE, 2>& edge : edges)
        {
            size_t i = node2idx[edge[0]];
            size_t j = node2idx[edge[1]];
            graph_[i].push_back(j);
            graph_[j].push_back(i);
        }

        // sort the neighborhoods of all vertices in ascending index order
        for (std::vector<size_t>& neighborhood : graph_)
            std::sort(neighborhood.begin(), neighborhood.end());
    }

    std::vector<size_t> compute_neighborhood_intersections_of_neighbors(size_t i)
    {
        // fill the intersection sizes of i with its neighbors with 2s because
        // the neighborhoods of i and a neighbor j intersect at least in {i, j}.
        std::vector<size_t> intersections(graph_[i].size(), 2);
        // iterate over all neighbors j of i and compute the intersection size
        for (size_t j_idx = 0; j_idx < graph_[i].size(); ++j_idx)
        {
            size_t j = graph_[i][j_idx];
            intersections[j_idx] += intersection_size(graph_[i], graph_[j]);
        }
        return intersections;
    }

    /**
     * @brief This method computes the size of the intersection of the neighborhoods
     * of all pairs of nodes whose neighborhoods intersect in at least one node
     * 
     * @return std::map<std::pair<size_t, size_t>, size_t> 
     */
    std::map<std::pair<size_t, size_t>, size_t> all_intersection_sizes()
    {
        std::map<std::pair<size_t, size_t>, size_t> intersection_size;
        for (size_t i = 0; i < graph_.size(); ++i)
        {
            for (auto it1 = graph_[i].begin(); it1 != graph_[i].end(); ++it1)
            {
                if (i < *it1)
                    intersection_size[{i, *it1}] += 2;
                for (auto it2 = it1 + 1; it2 != graph_[i].end(); ++it2) 
                {
                    ++intersection_size[{*it1, *it2}];
                }
            }
        }
        return intersection_size;
    }
};

