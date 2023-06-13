import argparse

import os
import datetime
from math import floor
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from SVD.svdDecom import SVDDecompose, logSVDToTensorboard, historyRecord
from torch.utils.tensorboard import SummaryWriter

class ChebNet(nn.Module):
    def __init__(self,in_size, hid_size, out_size, k, num_layers = 5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.5)

        self.layers.append(
            dglnn.ChebConv(in_size, hid_size, k, activation=F.relu, bias=True)
        )

        for _count_layers in range(num_layers):
            self.layers.append(
                dglnn.ChebConv(hid_size, hid_size, k, activation=F.relu, bias=True)
            )
        
        self.layers.append(
            dglnn.ChebConv(hid_size, out_size, k, activation=F.relu, bias=True)
        ) 
    def forward(self, g, features):
        h = features
        for _count, _count_layer in enumerate(self.layers):
            if _count != 0:
                h = self.dropout(h)
            h = _count_layer(g,h)
        return h


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        # multi-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )

        for _count_layers in range(num_layers-2):
            self.layers.append(
                dglnn.GraphConv(hid_size, hid_size, activation=F.relu)
            )

        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class GCN_FeedForward(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # multi-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )

        self.layers.append(dglnn.GraphConv(hid_size, hid_size))
        self.layers.append(nn.Linear(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                h = layer(h)
            else:
                if i != 0:
                    h = self.dropout(h)
                h = layer(g, h)
        return h

class NGSIMDataset(torch.utils.data.Dataset):
    def __init__(self, graph_path: str):
        assert os.path.exists(graph_path)
        self.graphs = dgl.load_graphs(graph_path)[0]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, ind):
        # Get a graph list for a vehicle ID
        return self.graphs[ind]

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model, recorder: historyRecord, dataset_name: str, writer: SummaryWriter):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # define the svd result dict
    #  svd_result = [{},{},{}]

    # define log interval
    log_interval = 1

    #define the save interval
    save_interval = 100

    # training loop
    for epoch in range(3000):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Here we calculate the SVD result
        # if epoch % log_interval == 0:
            #_decomposed_result = [SVDDecompose(model.layers[_layer_count].weight.cpu().detach().clone().numpy()) for _layer_count in range(len(model.layers))]
            # _decomposed_result = [SVDDecompose(model.layers[_layer_count].linear.weight.cpu().detach().clone().numpy()) for _layer_count in range(len(model.layers))]
            # recorder.add_epoch_data(epoch, _decomposed_result, dataset_name, writer)
            # for _layer_count in range(len(model.layers)):
            #     _U, _s, _Vh = SVDDecompose(model.layers[_layer_count].weight.cpu().detach().clone().numpy())
            #     logSVDToTensorboard(writer, [_U, _s, _Vh], epoch, _layer_count)

        # if epoch % save_interval == 0:
        #     _saved_path = f'time_{datetime.datetime.now()}_epoch_{epoch}_with_{len(model.layers)}'
        #     assert not os.path.exists(_saved_path)
        #     torch.save(model.state_dict(), _saved_path)

        #

        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )

def load_and_return(model: nn.Module, path: str):
    assert model is not None
    assert os.path.exists(path)

    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def train_gcn_feedforward():

    batch_frame_size = 10

    # Load the graphs
    graphs_base = './pickled_graphs'
    graphs_list = os.listdir(graphs_base)

    for _graph_file in graphs_list:
        _graph = dgl.add_self_loop(dgl.load_graphs(os.path.join(graphs_base, _graph_file))[0])
        _graph_length = len(_graph)

        _iteration_times = floor(_graph_length / batch_frame_size)




if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="cora",
    #     help="Dataset name ('cora', 'citeseer', 'pubmed').",
    # )
    # args = parser.parse_args()
    # print(f"Training with DGL built-in GraphConv module.")
    #
    # # load and preprocess dataset
    # transform = (
    #     AddSelfLoop()
    # )  # by default, it will first remove self-loops to prevent duplication
    # if args.dataset == "cora":
    #     data = CoraGraphDataset(transform=transform)
    # elif args.dataset == "citeseer":
    #     data = CiteseerGraphDataset(transform=transform)
    # elif args.dataset == "pubmed":
    #     data = PubmedGraphDataset(transform=transform)
    # else:
    #     raise ValueError("Unknown dataset: {}".format(args.dataset))
    #
    # # dataset has only one graph
    # g = data[0]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # g = g.int().to(device)
    # features = g.ndata["feat"]
    # labels = g.ndata["label"]
    # masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    #
    # # create GCN model+
    # in_size = features.shape[1]
    # out_size = data.num_classes
    # model = GCN(in_size, 16, out_size, 5).to(device)
    # # model = ChebNet(in_size, 16, out_size, 10, 2).to(device)
    #
    # # define the recorder
    # recorder = historyRecord(len(model.layers), [args.dataset],'cosine')
    #
    # # define the summary writer
    # writer = SummaryWriter()
    #
    # # model training
    # print("Training...")
    # train(g, features, labels, masks, model, recorder, args.dataset, writer)
    #
    # writer.close()
    #
    # # test the model
    # print("Testing...")
    # acc = evaluate(g, features, labels, masks[2], model)
    # print("Test accuracy {:.4f}".format(acc))
    base_dir = './pickled_graphs'
    graph_files = os.listdir(base_dir)

    # for _graph_file in graph_files:
    #     pass

    model = GCN_FeedForward(7, 256, 2)
    test_data = dgl.load_graphs('./pickled_graphs/1_smoothed_trajectories-0400-0415_deli.graph')
    g = dgl.add_self_loop(test_data[0][100])
    g1 = dgl.add_self_loop(test_data[0][101])
    feat = test_data[0][100].ndata['feat']
    feat1 = test_data[0][101].ndata['feat']



    print('done')