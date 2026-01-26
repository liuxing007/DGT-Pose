from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from timm.models.layers import DropPath
from model.modules.graph_frames import Graph
from model.modules.mlp_gcn import Mlp_gcn

from model.modules.attention import Attention
from model.modules.graph import GCN
from model.modules.mlp import MLP
from model.modules.tcn import MultiScaleTCN
from einops import rearrange


if __name__ == '__main__':
    _test()
