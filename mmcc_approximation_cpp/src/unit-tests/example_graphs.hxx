#include <vector>
#include <array>


namespace example_graphs
{

typedef std::vector<std::array<size_t, 2>> EDGES;

EDGES triangle_with_antenna() {
    return {{3, 7}, {7, 1}, {3, 1}, {7, 8}, {8, 9}, {8, 4}};
}

EDGES y(){
    return {{0, 2}, {0, 4}, {0, 5}, {1, 2}, {1, 3}};
}


EDGES fish(){
    return {{0, 1}, {0, 2}, {0, 3}, {1, 4}, {2, 4}, {3, 4}, {4, 5}, {4, 6}, {5, 6}};
}

EDGES empty(){
    return {};
}

EDGES five_cycle_plus_triangle(){
    return {{0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 4}, {3, 5}, {4, 5}};
}

EDGES mini_fish(){
    return {{0, 2}, {0, 3}, {0, 4}, {1, 3}, {1, 5}, {2, 4}, {3, 4}, {3, 5}};
}

EDGES complete(size_t n){
    EDGES edges;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i+1; j < n; ++j)
            edges.push_back({i, j});
    return edges;
}


} // namespace example_graphs
