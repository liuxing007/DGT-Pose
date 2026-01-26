import math

import torch
from torch import nn

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9,11,14], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None):
        self.nodes_ = """
        :param dim_int: Channel input dimension
        :param dim_out: Channel output dimension
        :param num_nodes: Number of nodes
        :param neighbour_num: Neighbor numbers. Used in temporal GCN to create edges
        :param mode: Either 'spatial' or 'temporal'
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param connections: Spatial connections for graph edges (Optional)
        """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"

        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections

        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

        self._init_gcn()

        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
        elif mode == 'temporal' and not self.use_temporal_similarity:
            self.adj = self._init_temporal_adj(temporal_connection_len)


