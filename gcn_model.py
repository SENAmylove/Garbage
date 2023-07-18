import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import GraphConvolution


class GCN_FF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_FF, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(256, 6)
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropoutm, training=self.training)

        return self.linear(x)

if __name__ == '__main__':
    """
        N -> Batch size -> 64
        C -> Channel size -> 6
        T -> Time size -> 10
        V -> Node size -> 9
        
        ncv * vh nch 
    """