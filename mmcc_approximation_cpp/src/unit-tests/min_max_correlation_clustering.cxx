#include <min_max_correlation_clustering.hxx>
#include "example_graphs.hxx"


#define ASSERT(condition) { if(!(condition)){ std::cerr << "ASSERT FAILED: " << #condition << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; } }


void assert_bound_correct(const example_graphs::EDGES& edges, size_t expected_bound)
{
    MinMaxCorrelationClustering<size_t> mmcc(edges);
    size_t bound = mmcc.combinatorial_lower_bound();
    ASSERT(bound == expected_bound);
}

void test_bound()
{
    for (size_t n = 0; n < 5; ++n)
        assert_bound_correct(example_graphs::complete(n), 0);
    assert_bound_correct(example_graphs::triangle_with_antenna(), 1);
    assert_bound_correct(example_graphs::y(), 1);
    assert_bound_correct(example_graphs::fish(), 3);
    assert_bound_correct(example_graphs::five_cycle_plus_triangle(), 1);
    assert_bound_correct(example_graphs::mini_fish(), 2);
}

void assert_greedy_joining_correct(const example_graphs::EDGES& edges, size_t expected_disagreement)
{
    MinMaxCorrelationClustering<size_t> mmcc(edges);
    auto result = mmcc.greedy_joining();
    ASSERT(result.first == expected_disagreement);
}

void test_greedy_joining()
{
    for (size_t n = 0; n < 5; ++n)
        assert_greedy_joining_correct(example_graphs::complete(n), 0);
    assert_greedy_joining_correct(example_graphs::triangle_with_antenna(), 1);
    assert_greedy_joining_correct(example_graphs::y(), 2);
    assert_greedy_joining_correct(example_graphs::fish(), 3);
    assert_greedy_joining_correct(example_graphs::five_cycle_plus_triangle(), 2);
    assert_greedy_joining_correct(example_graphs::mini_fish(), 2);
}


int main()
{
    test_bound();
    test_greedy_joining();
    return 0;
}