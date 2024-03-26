import numpy as np
import networkx as nx
from cdt.metrics import SHD_CPDAG, SID, get_CPDAG, retrieve_adjacency_matrix
import torch

a = [
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
]
"""a  = np.array(a)
g = nx.from_numpy_array(a)
nx.is_chordal(g)
"""
a = torch.tensor(a)


def dfs(g: torch.Tensor, visited: torch.Tensor, index: int, part: torch.Tensor):
    part[index] = True
    visited[index] = True
    for i in range(0, g.shape[0]):
        if (not visited[i]) and g[index][i] == 1:
            dfs(g, visited, i, part)


def decomopose_graph_directed(g: torch.Tensor):
    # remove undirected edges
    g[g == g.T] = 0
    visited = torch.zeros(g.shape[0], dtype=torch.bool)
    directed_parts = []
    for i in range(0, g.shape[0]):
        if not visited[i]:
            part = torch.zeros(g.shape[0], dtype=torch.bool)
            dfs(g, visited, i, part)
            g_ = g.clone()
            g_[~part, :] = 0
            g_[:, ~part] = 0
            if g_.any():
                directed_parts.append(g)

    return directed_parts


def decomopose_graph_undirected(g: torch.Tensor):
    # remove directed edges
    g[g != g.T] = 0
    visited = torch.zeros(g.shape[0], dtype=torch.bool)
    undirected_parts = []
    for i in range(0, g.shape[0]):
        if not visited[i]:
            part = torch.zeros(g.shape[0], dtype=torch.bool)
            dfs(g, visited, i, part)
            g_ = g.clone()
            g_[~part, :] = 0
            g_[:, ~part] = 0
            if g_.any():
                undirected_parts.append(g_)

    return undirected_parts


def check_chordality(g: torch.Tensor):
    g = nx.from_numpy_array(g.numpy())
    return nx.is_chordal(g)


def check_acyclicity(g: torch.Tensor):
    g = nx.from_numpy_array(g.numpy())
    return nx.is_directed_acyclic_graph(g)

def get_edges_propabilities(p: torch.Tensor, cycles: list):
    max_p = -1
    indexes = (0, 0)
    for c in cycles:
        for i in range(0, len(c)):
            if p[c[i-1]][c[i]] > max_p:
                print(c[i - 1], c[i])
                max_p =  p[c[i-1]][c[i]] 
                indexes = (c[i-1], c[i])

            if p[c[i]][c[i-1]] > max_p:
                print(c[i - 1], c[i])
                max_p =  p[c[i]][c[i-1]] 
                indexes = (c[i], c[i-1])
    return indexes


def orient_unchordal_part(part: torch.Tensor, prob: torch.Tensor, directed_parts: list):
    cycles = []
    prob = prob * part
    prob = (1 - prob)
    #c = nx.chordless_cycles(nx.from_numpy_array(part.numpy()))
    for i in  nx.chordless_cycles(nx.from_numpy_array(part.numpy())):
        cycles.append(i)

    while not check_chordality(part):
        get_edges_propabilities(part, cycles)
        break

    print(prob)
    print(cycles)



def correct_graph(g: torch.Tensor, prob: torch.Tensor):
    undirected_parts = decomopose_graph_undirected(g)
    directed_parts = decomopose_graph_directed(g)
    for i in directed_parts:
        assert check_acyclicity(i)

    for i in undirected_parts:
        if not check_chordality(i):
            orient_unchordal_part(i, prob, directed_parts)
        
p = torch.zeros(a.shape).fill_(0.5)
correct_graph(a, p)
