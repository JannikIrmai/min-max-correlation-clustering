#!/usr/bin/env python
# coding: utf-8

# The code in this file is from the supplementary material of:

# Sami Davies, Benjamin Moseley, and Heather Newman. Fast combinatorial algorithms for min max correlation clustering.
# In Proceedings of the 40th International Conference on Machine Learning, 2023.

# In[12]:


import numpy as np
import math
from numpy.linalg import matrix_power
import gurobipy as grb
import time

import networkx as nx
from util import sort_nodes_by_degree


# In[15]:


#Computing the correlation metric distances
#input: positive adjacency matrix, radii r and r2 for rounding algorithm 
#output: correlation metric distances, L_0 values, r and r2 neighborhoods, time, fractional cost
def exact(pos_adj_mx, r, r2):
    t0 = time.time()
    n = np.shape(pos_adj_mx)[0]
    if not(np.array_equal(np.diagonal(pos_adj_mx), np.ones([n]))):
           raise Exception('Diagonal not all 1s')
    #initialize a dictionary that stores the L_t values of each vertex
    L_t_vals = {}
    #initialize the dictionaries that store the r and r2 neighborhoods of each vertex
    neighborsR = {}
    neighborsR2 = {}
    for k in range(n):
        L_t_vals.update({k: 0})
        neighborsR.update({k: []})
        neighborsR2.update({k: []})
    neg_adj_mx = np.subtract(np.ones([n,n]), pos_adj_mx)
    #for each pair of nodes, compute common positive neighborhood 
    pos_len_2 = matrix_power(pos_adj_mx, 2) 
    #compute the vector of positive degrees
    pos_degrees = np.matmul(pos_adj_mx, np.ones(n))

    #initialize the correlation metric distances
    distances = np.zeros([n,n])
    #compute exact correlation metric using sizes of common positive and negative neighborhoods
    for u in range(n):
        for w in range(n-u-1):
            v = u + w + 1
            distances[u][v] = 1 - ((pos_len_2[u][v])/(pos_degrees[u] + pos_degrees[v] - pos_len_2[u][v]))
            distances[v][u] = distances[u][v]
            #add to r2-neighborhood 
            if distances[u][v] <= r2:
                neighborsR2[u].append(v)
                neighborsR2[v].append(u)
                #update L_t values and add to r-neighborhood
                if distances[u][v] <= r:
                    L_t_vals[u] = L_t_vals[u] + r - distances[u][v]
                    neighborsR[u].append(v)
                    L_t_vals[v] = L_t_vals[v] + r - distances[u][v]
                    neighborsR[v].append(u)
    t1 = time.time()
    clock = t1 - t0
    
    #for analysis only: compute the fractional cost of the correlation metric
    frac_values = []
    for v in range(n):
        tot = 0
        neg_deg = n - np.dot(np.ones([n]), pos_adj_mx[v])
        for w in range(n):
            if pos_adj_mx[w,v] == 0:
                tot = tot + 1 - distances[v][w]
            else:
                tot = tot + distances[v][w]
        frac_values.append(tot)
    frac_val = max(frac_values)

    return distances, L_t_vals, neighborsR, neighborsR2, clock, frac_val 


# In[1]:


#KMZ Phase 2 (Rounding Algorithm)
#input: distances, L_0 values, R and R2 neighborhoods, radii r and r2 
#output: set of clusters (as a list of lists) and time 
def cluster(distances, L_t_vals, neighborsR, neighborsR2, r, r2):
    t0 = time.time()
    #store the clusters in this list
    clustering = []
    n = np.shape(distances)[0]
    #yet unclustered vertices
    num_unclustered = n
    #indicator list indexed by the vertices: 1 if unclustered, 0 if clustered 
    V_t = np.ones([n])  
    while num_unclustered > 0: 
        #find the vertex maximizing L_t
        max_key = max(L_t_vals, key=L_t_vals.get)
        #initialize new cluster with maximizing vertex
        cluster = [max_key]
        #cut out the new cluster
        for j in range(len(neighborsR2[max_key])):
            #check whether this R2-neighbor of the maximizing vertex has been clustered yet
            if V_t[neighborsR2[max_key][j]] == 1:
                #if not, add to present cluster
                cluster.append(neighborsR2[max_key][j])
        #append the new cluster to the list of clusters
        clustering.append(cluster)
        #update the number of unclustered vertices remaining 
        num_unclustered = num_unclustered - len(cluster)
        #update L_t
        for k in range(len(cluster)):
            #remove L_t values for clustered vertices
            del L_t_vals[cluster[k]]
            #mark clustered vertices as clustered
            V_t[cluster[k]] = 0
            #update remaining L_t values 
            for key in L_t_vals:
                if distances[cluster[k]][key] <= r:
                    L_t_vals[key] = L_t_vals[key] - (r - distances[cluster[k]][key])
    t1 = time.time()
    clock = t1 - t0
    return clustering, clock 


# In[17]:


#input: positive adjacency matrix, clustering (as a list of lists), vector of positive degrees, p in lp norm
#output: vector of disagreements, objective value (max disagreements at a vertex), maximizing vertex
def LocalObj(pos_adj_mx, clustering, pos_degrees, norm):
    n = len(pos_degrees)
    disag_vector = np.zeros(n)
    num_clusters = len(clustering)
    for i in range(num_clusters):
        clus = clustering[i]
        for j in range(len(clus)):
            pos_disag = pos_degrees[clus[j]]
            neg_disag = 0
            for k in range(len(clus)):
                if pos_adj_mx[clus[j]][clus[k]] == 1:
                    pos_disag = pos_disag - 1 
                else:
                    neg_disag = neg_disag + 1
            disag_vector[clus[j]] = pos_disag + neg_disag 
    alg_obj_val = np.linalg.norm(disag_vector, norm)
    if norm == math.inf:
        obj_vx = np.argmax(disag_vector)
    else:
        obj_vx = math.inf 
    return disag_vector, alg_obj_val, obj_vx


# In[18]:


#input: positive adjacency matrix
#output: vector of positive degrees
def DegreeDist(pos_adj_mx):
    n = np.shape(pos_adj_mx)[0]
    degrees = np.dot(pos_adj_mx, np.ones(n))
    return degrees 


# In[2]:


#input: positive adjacency matrix, LP solver (in Gurobi)
#output: LP objective value, LP solution
def MinMaxLPonly(pos_adj_mx, method):
    
    n = np.shape(pos_adj_mx)[0]
    
    if not(np.array_equal(np.diagonal(pos_adj_mx), np.ones([n]))):
        raise Exception('Diagonal not all 1s')
    
    upper_bounds = np.ones(int(n*(n-1)/2)+1)
    upper_bounds[int(n*(n-1)/2)] = grb.GRB.INFINITY
    
    
    M = grb.Model('my_model')
    
    K = []
    
    for i in range(n):
        for j in range(n-i-1):
            K.append((i,i+j+1))
    K.append((0,0))       
    
    l = grb.tuplelist(K)
            
    x = M.addVars(l, name = 'x', ub = upper_bounds)
    
    for i in range(n-2):
        for j in range(n-2-i):
            for k in range(n-2-i-j):
                u = i
                v = i + j + 1
                w = i + j + k + 2
                M.addConstr(x[u,v] + x[v,w] >= x[u,w])
                M.addConstr(x[u,v] + x[u,w] >= x[v,w])
                M.addConstr(x[v,w] + x[u,w] >= x[u,v])
                
    for v in range(n):
        cons = x[0,0]
        neg_deg = n - np.dot(np.ones([n]), pos_adj_mx[v])
        for w in range(n):
            if w < v:
                if pos_adj_mx[w,v] == 0:
                    cons = cons + x[w,v]
                else:
                    cons = cons - x[w,v]
            if w > v:                
                if pos_adj_mx[v,w] == 0:
                    cons = cons + x[v,w]
                else:
                    cons = cons - x[v,w]
        M.addConstr(cons >= neg_deg)
        

    M.setObjective(x[0,0], grb.GRB.MINIMIZE)
        
    M.setParam('Method', method)

    M.optimize()
    
    distances = np.zeros([n,n])
    for u in range(n):
        for w in range(n-u-1):
            v = u + w + 1
            distances[u][v] = x[u, v].x
            distances[v][u] = x[u, v].x
    
    return M.objVal, distances


# In[20]:


#input: number of vertices, distances, radii r and r2 for rounding algorithm
#output: L_0 values, R and R2 neighborhoods
def MinMaxLPneighbors(n, distances, r, r2):
    #initialize a dictionary that stores the L_t values of each vertex
    L_t_vals = {}
    #initialize the dictionaries that store the r and r2 neighborhoods of each vertex
    neighborsR = {}
    neighborsR2 = {}
    for k in range(n):
        L_t_vals.update({k: 0})
        neighborsR.update({k: []})
        neighborsR2.update({k: []})
    
    for u in range(n):
        for w in range(n-u-1):
            v = u + w + 1
            #add to r2-neighborhood
            if distances[u][v] <= r2:
                neighborsR2[u].append(v)
                neighborsR2[v].append(u)
                #update L_t values and add to r-neighborhood
                if distances[u][v] <= r:
                    L_t_vals[u] = L_t_vals[u] + r - distances[u][v]
                    neighborsR[u].append(v)
                    L_t_vals[v] = L_t_vals[v] + r - distances[u][v]
                    neighborsR[v].append(u) 
    return L_t_vals, neighborsR, neighborsR2


# In[3]:


#Combines MinMaxLPonly and MinMaxLPneighbors
def MinMaxLP(pos_adj_mx, r, r2, method):
    
    t0 = time.time()
    
    n = np.shape(pos_adj_mx)[0]
    
    if not(np.array_equal(np.diagonal(pos_adj_mx), np.ones([n]))):
        raise Exception('Diagonal not all 1s')
    
    upper_bounds = np.ones(int(n*(n-1)/2)+1)
    upper_bounds[int(n*(n-1)/2)] = grb.GRB.INFINITY
    
    
    M = grb.Model('my_model')
    
    K = []
    
    for i in range(n):
        for j in range(n-i-1):
            K.append((i,i+j+1))
    K.append((0,0))       
    
    l = grb.tuplelist(K)
            
    x = M.addVars(l, name = 'x', ub = upper_bounds)
    
    for i in range(n-2):
        for j in range(n-2-i):
            for k in range(n-2-i-j):
                u = i
                v = i + j + 1
                w = i + j + k + 2
                M.addConstr(x[u,v] + x[v,w] >= x[u,w])
                M.addConstr(x[u,v] + x[u,w] >= x[v,w])
                M.addConstr(x[v,w] + x[u,w] >= x[u,v])
                
    for v in range(n):
        cons = x[0,0]
        neg_deg = n - np.dot(np.ones([n]), pos_adj_mx[v])
        for w in range(n):
            if w < v:
                if pos_adj_mx[w,v] == 0:
                    cons = cons + x[w,v]
                else:
                    cons = cons - x[w,v]
            if w > v:                
                if pos_adj_mx[v,w] == 0:
                    cons = cons + x[v,w]
                else:
                    cons = cons - x[v,w]
        M.addConstr(cons >= neg_deg)
        

    M.setObjective(x[0,0], grb.GRB.MINIMIZE)
        
    M.setParam('Method', method)

    M.optimize()
        
    
    L_t_vals = {}
    neighborsR = {}
    neighborsR2 = {}
    for k in range(n):
        L_t_vals.update({k: 0})
        neighborsR.update({k: []})
        neighborsR2.update({k: []})
    
    distances = np.zeros([n,n])
    for u in range(n):
        for w in range(n-u-1):
            v = u + w + 1
            distances[u][v] = x[u, v].x
            distances[v][u] = x[u, v].x
            if distances[u][v] <= r2:
                neighborsR2[u].append(v)
                neighborsR2[v].append(u)
                if distances[u][v] <= r:
                    L_t_vals[u] = L_t_vals[u] + r - distances[u][v]
                    neighborsR[u].append(v)
                    L_t_vals[v] = L_t_vals[v] + r - distances[u][v]
                    neighborsR[v].append(u)
    
    t1 = time.time()
    clock = t1 - t0
    
    return M.objVal, distances, L_t_vals, neighborsR, neighborsR2, clock


# In[22]:


#Pivot algorithm of Ailon, Charikar, and Newman
#input: positive adjacency matrix
#output: clustering, as a dictionary and as a list of lists 
def PivotAlg(pos_adj_mx):
    n = np.shape(pos_adj_mx)[0]
    #random perumtation of vertices
    perm = np.random.permutation(n)
    #keys will be pivots, values will be clusters
    pivot_clusters = {}
    pivot_clusters_list = []
    for i in range(n):
        #perm[i] clustered: 0 yes, 1 no
        clustered = 0 
        for key in pivot_clusters.keys():
            #check if perm[i] has a positive edge to an existing pivot (key)
            if pos_adj_mx[perm[i]][key] == 1:
                #add to perm[i] to the pivot's cluster
                pivot_clusters[key].append(perm[i])
                #mark perm[i] as clustered
                clustered = 1
                break
        #if perm[i] not clustered by existing pivot
        if clustered == 0:
            #make perm[i] a new pivot
            pivot_clusters.update({perm[i]: [perm[i]]})
    for key in pivot_clusters.keys():
        clus = pivot_clusters[key]
        pivot_clusters_list.append(clus)
    #output as dictionary and as list of lists
    return pivot_clusters, pivot_clusters_list


def dmn_python(edges, r1=0.7, r2=0.7):
    """
    This method takes in a list of edges as well as the two hyperparameters r1 and r2 of the DMN algorithm.
    It computes a clustering according to the DMN algorithm by calling the exact, cluster, and LocalObj method
    """
    # make node indices consecutive integers starting from 0
    graph = nx.Graph()
    for u, v in edges:
        graph.add_edge(u, v)
    graph = sort_nodes_by_degree(graph)

    # build adjacency matrix
    adj_mat = np.eye(graph.number_of_nodes())
    for u, v in graph.edges:
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1

    # execute DMN algorithm
    distances, l_t_vals, neighbors_r, neighbors_r2, clock, frac_val = exact(adj_mat, r1, r2)
    clustering, cluster_clock = cluster(distances, l_t_vals, neighbors_r, neighbors_r2, r1, r2)
    degrees = DegreeDist(adj_mat)
    _, alg_obj_val, _ = LocalObj(adj_mat, clustering, degrees, np.inf)
    return alg_obj_val, clock + cluster_clock
