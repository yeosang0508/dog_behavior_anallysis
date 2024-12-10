import numpy as np
import torch
from torch_geometric.data import Data

# 관절 연결 정의 (수정된 버전)
edges = [
    (0, 1), (0, 3), (2, 3), (3, 4), (4, 5),
    (4, 6), (5, 7), (6, 8), (4, 13), (13, 9),
    (13, 10), (13, 14), (9, 11), (10, 12)
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


