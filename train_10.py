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
import numpy as np
from SVD.svdDecom import SVDDecompose, logSVDToTensorboard, historyRecord
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


class ChebNet(nn.Module):
    def __init__(self, in_size, hid_size, out_size, k, num_layers=5):
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
            h = _count_layer(g, h)
        return h


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # multi-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )

        for _count_layers in range(num_layers - 2):
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

    # define the save interval
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
        # _decomposed_result = [SVDDecompose(model.layers[_layer_count].weight.cpu().detach().clone().numpy()) for _layer_count in range(len(model.layers))]
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

def train_gcn_ff_10(model: nn.Module):
    # Load the graphs
    graphs_base = './pickled_graphs/train'
    graphfiles_list = os.listdir(graphs_base)

    # loss function
    loss = nn.L1Loss()
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    #
    model.train()
    files_count = 100

    #
    for _graph_file in graphfiles_list:
        _graphs_list = dgl.load_graphs(os.path.join(graphs_base, _graph_file))[0]
        _loss_list = []

        #
        iteration = floor(len(_graphs_list) / 10)
        for _iter in range(iteration):


def train_gcn_feedforward(model: nn.Module):
    # Load the graphs
    graphs_base = './pickled_graphs/train'
    graphfiles_list = os.listdir(graphs_base)

    # loss function
    loss = nn.L1Loss()
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    #
    model.train()
    files_count = 100

    for _graph_file in graphfiles_list:
        _graphs_list = dgl.load_graphs(os.path.join(graphs_base, _graph_file))[0]
        _loss_list = []

        for _graph_index, _graph in enumerate(_graphs_list[:-1]):
            _current_vehicle_ID = list(np.array(_graph.ndata['feat'][:, 0]))
            _next_vehicle_ID = list(np.array(_graphs_list[_graph_index + 1].ndata['feat'][:, 0]))

            _predicted = model(dgl.add_self_loop(_graph), _graph.ndata['feat'][:, 1:])

            # ensure that the vehicles in this frame exists in next fram
            total_loss = torch.tensor(0.0)
            for _v_index, _v in enumerate(_next_vehicle_ID):
                if _v in _current_vehicle_ID:
                    _cur_v_index = _current_vehicle_ID.index(_v)
                    total_loss += loss(_graphs_list[_graph_index + 1].ndata['feat'][_v_index, 1:3],
                                       _predicted[_cur_v_index, :])
                    # _print_count -= 1
                    # if _print_count == 0:
                    # logger.info(f'predicted pos: {_predicted[_cur_v_index,:]}, ground truth: {_graphs_list[_graph_index+1].ndata["feat"][_v_index,1:3]}')
                    # _print_count = 500

            _loss_list.append(total_loss.detach().cpu().numpy())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        logger.info(f'Working on graph file {_graph_file} with mean loss {np.mean(_loss_list)}')

        files_count -= 1
        if files_count == 0:
            val_gcn_feedward(model)
            files_count = 100


def val_gcn_feedward(model: nn.Module):
    # load the val graphs
    graphs_base = './pickled_graphs/val'
    graphfiles_list = os.listdir(graphs_base)

    # loss
    loss = nn.L1Loss()

    model.eval()

    for _graph_file in graphfiles_list:
        _graphs_list = os.listdir(graphs_base)
        _loss_list = []

        for _graph_index, _graph in enumerate(_graphs_list[:-1]):
            _current_vehicle_ID = list(np.array(_graph.ndata['feat'][:, 0]))
            _next_vehicle_ID = list(np.array(_graphs_list[_graph_index + 1].ndata['feat'][:, 0]))

            _predicted = model(dgl.add_self_loop(_graph), _graph.ndata['feat'][:, 1:])

            total_loss = torch.tensor(0.0)
            for _v_index, _v in enumerate(_next_vehicle_ID):
                if _v in _current_vehicle_ID:
                    _cur_v_index = _current_vehicle_ID.index(_v)
                    total_loss += loss(_graphs_list[_graph_index + 1].ndata['feat'][_v_index, 1:3],
                                       _predicted[_cur_v_index, :])

            _loss_list.append(total_loss.detach().cpu().numpy())

        logger.info(f'Validating on graph file {_graph_file} with mean loss {np.mean(_loss_list)}')


if __name__ == "__main__":

    logger.add('gcn_feedforward.log')
    model = GCN_FeedForward(7, 256, 2)
    train_gcn_feedforward(model)

    print('done')