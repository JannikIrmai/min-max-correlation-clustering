# Approximation Algorithms for min max correlation clustering (MMCC)

This repository contains algorithms for the min max correlation clustering problem that are published [here](https://proceedings.mlr.press/v238/s-g-heidrich24a.html).

## Structure

The directory `mmcc_approximation_cpp` contains `c++` implementations of the algorithms 
as well as a python interface for those implementations. 
The python interface is based on [pybind11](https://pybind11.readthedocs.io/en/stable/index.html).
The python files in the top level directory contain code for reproducing the experiments from the article.

## Installation

To install the `c++` implementation of the MMCC approximation algorithms as a python package, execute:
```
pip install wheel
pip install pybind11
pip install ./mmcc_approximation_cpp
```
or simply execute
```
pip install -r requirements.txt
```
to install all required packages at once.

## Algorithms

In the `MinMaxCorrelationClustering` class, three algorithms are implemented:
- The method `bound` computes the combinatorial lower bound for the MMCC problem.
- The method `greedy_joining` implements a greedy joining heuristic for the MMCC problem.
- The method `dmh` computes an approximation for the MMCC problem with the algorithm presented in Davies et al. (2023) 


## Usage
After installing the algorithms as python package the algorithms can be used as follows
```python
from mmcc_approximation import MinMaxCorrelationClustering
edges = [[1, 2], [2, 3], [7, 11]]
mmcc = MinMaxCorrelationClustering(edges)

bound = mmcc.bound()
disagreement, clustering = mmcc.greedy_joining()
```

## Experiments

Complete source code for reproducing the experiments in the article is provided in `social_network_experiments.py` and
in `synthetic_experiment.py`.

### Data
The graph data for the social network experiments can be downloaded from [here](https://snap.stanford.edu/data/).
Place the data into a `data` directory that is on the same level as the `social_network_experiments.py` file.


## Cite this work

```
@InProceedings{heidrich2024a,
  title = {A 4-Approximation Algorithm for Min Max Correlation Clustering},
  author = {S.G. Heidrich, Holger and Irmai, Jannik and Andres, Bjoern},
  booktitle = {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  year = {2024},
  url = {https://proceedings.mlr.press/v238/s-g-heidrich24a.html}
}
```
