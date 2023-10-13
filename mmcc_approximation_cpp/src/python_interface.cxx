#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <min_max_correlation_clustering.hxx>


namespace py = pybind11;

template<class NODE>
MinMaxCorrelationClustering<NODE> init_mmcc(
    const std::vector<std::array<size_t, 2>>& edges
) {
    return MinMaxCorrelationClustering<NODE>(edges);
}


// wrap as Python module
PYBIND11_MODULE(mmcc_approximation, m)
{
    py::class_<MinMaxCorrelationClustering<size_t>>(m, "MinMaxCorrelationClustering")
        .def(py::init<>(&init_mmcc<size_t>))
        .def("bound", &MinMaxCorrelationClustering<size_t>::combinatorial_lower_bound)
        .def("greedy_joining", &MinMaxCorrelationClustering<size_t>::greedy_joining,
                py::arg("allow_cluster_joining") = true, 
                py::arg("allow_increase") = true, 
                py::arg("max_dis_largest_degree") = true, 
                py::arg("neighbor_smallest_degree") = true, 
                py::arg("require_worst_improvement") = true,
                py::arg("symm_diff_factor") = 0)
        .def("num_nodes", &MinMaxCorrelationClustering<size_t>::num_nodes)
        .def("num_edges", &MinMaxCorrelationClustering<size_t>::num_edges)
        .def("max_degree", &MinMaxCorrelationClustering<size_t>::max_degree)
        .def("dmn", &MinMaxCorrelationClustering<size_t>::dmn); 
}