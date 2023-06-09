import argparse
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from gripmodel import GRIPModel,GRIPModel_with_replaced_graph_param
from ngsim_feeder import Feeder
from datetime import datetime
import random
import itertools
from torch.utils.tensorboard import SummaryWriter
from ngsim_utils import my_print, my_save_model, data_loader, display_result


# CUDA_VISIBLE_DEVICES='1'
# os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

# ---
max_x = 1.
max_y = 1.
history_frames = 6  # 3 second * 2 frame/second
future_frames = 6  # 3 second * 2 frame/second

batch_size_train = 64
batch_size_val = 32
batch_size_test = 1
total_epoch = 50
base_lr = 0.01
lr_decay_epoch = 5
dev = 'cuda:0'
work_dir = './trained_models'
log_file = os.path.join(work_dir, 'log.txt')
test_result_file = 'prediction_result.txt'
# ---

if not os.path.exists(work_dir):
    os.makedirs(work_dir)


def preprocess_data(pra_data, pra_rescale_xy):
    # pra_data: (N, C, T, V)
    # N -> Batch Size
    # C -> Feature Size
    # T -> Temporal
    # V -> maximum size of targets

    # C = 19: [Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,
    # v_Class,v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway] + [mask]

    # 取出 feature
    # Local_X, Local_Y, v_Length, v_Width ,Space_Headway + [mask]
    feature_id = [4, 5, 8, 9, 16, 18]
    ori_data = pra_data[:, feature_id].detach()
    # 复制一份
    data = ori_data.detach().clone()
    # 计算速度 \delta X 和 \delta Y
    new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0)
    data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
    data[:, :2, 0] = 0

    # 1 - motorcycle, 2 - auto, 3 - truck
    object_type = pra_data[:, 10:11]

    data = data.float().to(dev)
    ori_data = ori_data.float().to(dev)
    object_type = object_type.to(dev)  # type
    data[:, :2] = data[:, :2] / pra_rescale_xy

    return data, ori_data, object_type


def compute_RMSE(pra_pred, pra_GT, pra_mask, pra_error_order=2):
    pred = pra_pred * pra_mask  # (N, C, T, V)=(N, 2, 6, 120)
    GT = pra_GT * pra_mask  # (N, C, T, V)=(N, 2, 6, 120)

    x2y2 = torch.sum(torch.abs(pred - GT) ** pra_error_order, dim=1)  # x^2+y^2, (N, C, T, V)->(N, T, V)=(N, 6, 120)
    overall_sum_time = x2y2.sum(dim=-1)  # (N, T, V) -> (N, T)=(N, 6)
    overall_mask = pra_mask.sum(dim=1).sum(dim=-1)  # (N, C, T, V) -> (N, T)=(N, 6)
    overall_num = overall_mask

    return overall_sum_time, overall_num, x2y2


def train_model(pra_model, pra_data_loader, pra_optimizer, pra_epoch_log):
    # writer
    # writer = SummaryWriter()
    # pra_model.to(dev)
    pra_model.train()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y

    # train model using training data
    for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
        # print(iteration, ori_data.shape, A.shape)
        # ori_data: (N, C, T, V)
        # C = [Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,v_Class,
        # v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway] + [mask]
        data, no_norm_loc_data, object_type = preprocess_data(ori_data, rescale_xy)

        loss_per_iteration = []
        for now_history_frames in range(1, data.shape[-2]):
            input_data = data[:, :, :now_history_frames, :]  # (N, C, T, V) = (N, 11, 6, 120)
            output_loc_GT = data[:, :2, now_history_frames:, :]  # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:, -1:, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)

            predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                  pra_teacher_forcing_ratio=0,
                                  pra_teacher_location=output_loc_GT)  # (N, C, T, V)=(N, 2, 6, 120)

            ########################################################
            # Compute loss for training
            ########################################################
            # We use abs to compute loss to backward update weights
            # (N, T), (N, T)
            overall_sum_time, overall_num, _ = compute_RMSE(predicted, output_loc_GT, output_mask, pra_error_order=1)
            # overall_loss
            total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num),torch.ones(1, ).to(dev))  # (1,)

            now_lr = [param_group['lr'] for param_group in pra_optimizer.param_groups][0]
            my_print(
                '|{}|{:>20}|\tIteration:{:>5}|\tLoss:{:.8f}|lr: {}|'.format(datetime.now(), pra_epoch_log, iteration,
                                                                            total_loss.data.item(), now_lr), log_file)
            loss_per_iteration.append(total_loss.data.item())
            pra_optimizer.zero_grad()
            total_loss.backward()
            pra_optimizer.step()

        # writer.add_scalar(f'{pra_epoch_log}_Iteration/loss', np.mean(loss_per_iteration), iteration)

def train_model_save_weights(pra_model, pra_data_loader, pra_optimizer, pra_epoch_log, current_epoch):
    # writer
    # writer = SummaryWriter()
    # pra_model.to(dev)
    pra_model.train()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y



    # train model using training data
    for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
        # print(iteration, ori_data.shape, A.shape)
        # ori_data: (N, C, T, V)
        # C = [Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,v_Class,
        # v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway] + [mask]
        data, no_norm_loc_data, object_type = preprocess_data(ori_data, rescale_xy)

        loss_per_iteration = []
        for now_history_frames in range(1, data.shape[-2]):
            input_data = data[:, :, :now_history_frames, :]  # (N, C, T, V) = (N, 11, 6, 120)
            output_loc_GT = data[:, :2, now_history_frames:, :]  # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:, -1:, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)

            if current_epoch == 49 and iteration == (len(pra_data_loader) - 1) and now_history_frames == (data.shape[-2]-1):

                if not os.path.exists(f'./trainable_graph/{current_epoch}_epoch/'):
                    os.makedirs(f'./trainable_graph/{current_epoch}_epoch/')

                # Save input data
                torch.save(input_data,f'./trainable_graph/{current_epoch}_epoch/input.pt')
                torch.save(output_loc_GT, f'./trainable_graph/{current_epoch}_epoch/output_loc_GT.pt')
                torch.save(output_mask, f'./trainable_graph/{current_epoch}_epoch/output_mask.pt')
                torch.save(output_mask, f'./trainable_graph/{current_epoch}_epoch/Adjacency.pt')

                predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                      pra_teacher_forcing_ratio=0,
                                      pra_teacher_location=output_loc_GT,graph = None, replace_graph = [False, False, False],
                                      save_graph = True, save_grap_path = f'./trainable_graph/{current_epoch}_epoch/')  # (N, C, T, V)=(N, 2, 6, 120)
            else:
                predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                      pra_teacher_forcing_ratio=0,
                                      pra_teacher_location=output_loc_GT)  # (N, C, T, V)=(N, 2, 6, 120)


            ########################################################
            # Compute loss for training
            ########################################################
            # We use abs to compute loss to backward update weights
            # (N, T), (N, T)
            overall_sum_time, overall_num, _ = compute_RMSE(predicted, output_loc_GT, output_mask, pra_error_order=1)
            # overall_loss
            total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num),torch.ones(1, ).to(dev))  # (1,)

            now_lr = [param_group['lr'] for param_group in pra_optimizer.param_groups][0]
            my_print(
                '|{}|{:>20}|\tIteration:{:>5}|\tLoss:{:.8f}|lr: {}|'.format(datetime.now(), pra_epoch_log, iteration,
                                                                            total_loss.data.item(), now_lr), log_file)
            loss_per_iteration.append(total_loss.data.item())
            pra_optimizer.zero_grad()
            total_loss.backward()
            pra_optimizer.step()

    if current_epoch == 49:
        model.save_weights(49)

        # writer.add_scalar(f'{pra_epoch_log}_Iteration/loss', np.mean(loss_per_iteration), iteration)

def val_model_replace_param(pra_model, pra_data_loader, replace_graph: bool, replace_params: bool):
    # pra_model.to(dev)
    pra_model.eval()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []

    # train model using training data
    for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
        # data: (N, C, T, V)
        # C = [Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,v_Class,
        # v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway]  + [mask]
        data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

        for now_history_frames in range(6, 7):
            input_data = data[:, :, :now_history_frames, :]  # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:, :2, now_history_frames:, :]  # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:, -1:, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:, :2, now_history_frames:, :]
            ori_output_last_loc = no_norm_loc_data[:, :2, now_history_frames - 1:now_history_frames, :]

            # for category
            cat_mask = ori_data[:, 10:11, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)

            if replace_params:
                pra_model.load_weights(49)

            if replace_graph:
                conv1_graph = torch.load(f'./trainable_graph/49_epoch/1_conv_block_graph.graph')
                conv2_graph = torch.load(f'./trainable_graph/49_epoch/2_conv_block_graph.graph')
                conv3_graph = torch.load(f'./trainable_graph/49_epoch/3_conv_block_graph.graph')

                predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                      pra_teacher_forcing_ratio=0,
                                      pra_teacher_location=output_loc_GT, graph = [conv1_graph, conv2_graph, conv3_graph],
                                      replace_graph = [True,True,True],save_graph = False, save_graph_path = None)  # (N, C, T, V)=(N, 2, 6, 120)
            else:
                predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                      pra_teacher_forcing_ratio=0,
                                      pra_teacher_location=output_loc_GT)

                ########################################################
            # Compute details for training
            ########################################################
            predicted = predicted * rescale_xy
            # output_loc_GT = output_loc_GT*rescale_xy

            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind - 1:ind + 1], dim=-2)
            predicted += ori_output_last_loc

            ### overall dist
            # overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask)
            # all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            all_overall_sum_list.extend(now_x2y2)

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    return all_overall_sum_list, all_overall_num_list

def val_model(pra_model, pra_data_loader):
    # pra_model.to(dev)
    pra_model.eval()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []

    # train model using training data
    for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
        # data: (N, C, T, V)
        # C = [Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,v_Class,
        # v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway]  + [mask]
        data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

        for now_history_frames in range(6, 7):
            input_data = data[:, :, :now_history_frames, :]  # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:, :2, now_history_frames:, :]  # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:, -1:, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:, :2, now_history_frames:, :]
            ori_output_last_loc = no_norm_loc_data[:, :2, now_history_frames - 1:now_history_frames, :]

            # for category
            cat_mask = ori_data[:, 10:11, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)
            predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                  pra_teacher_forcing_ratio=0,
                                  pra_teacher_location=output_loc_GT)  # (N, C, T, V)=(N, 2, 6, 120)
            ########################################################
            # Compute details for training
            ########################################################
            predicted = predicted * rescale_xy
            # output_loc_GT = output_loc_GT*rescale_xy

            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind - 1:ind + 1], dim=-2)
            predicted += ori_output_last_loc

            ### overall dist
            # overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask)
            # all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            all_overall_sum_list.extend(now_x2y2)

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    return all_overall_sum_list, all_overall_num_list

def val_model_save_graph(pra_model, pra_data_loader):
    # pra_model.to(dev)

    # load pre-trained params
    params = torch.load('./trained_models/model_epoch_0049.pt')

    state_dicts_list = []
    for _index,_part in enumerate(pra_model.st_gcn_networks):
        _temp_dict = {}
        for _state_key in _part.state_dict().keys():
            _temp_dict[_state_key] = params['xin_graph_seq2seq_model'][f'st_gcn_networks.{_index}.' + _state_key]

        pra_model.st_gcn_networks[_index].load_state_dict(_temp_dict)


    pra_model.eval()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []

    # train model using training data
    for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
        # data: (N, C, T, V)
        # C = [Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,v_Class,
        # v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway]  + [mask]
        data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

        for now_history_frames in range(6, 7):
            input_data = data[:, :, :now_history_frames, :]  # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:, :2, now_history_frames:, :]  # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:, -1:, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:, :2, now_history_frames:, :]
            ori_output_last_loc = no_norm_loc_data[:, :2, now_history_frames - 1:now_history_frames, :]

            # for category
            cat_mask = ori_data[:, 10:11, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)
            predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                  pra_teacher_forcing_ratio=0,
                                  pra_teacher_location=output_loc_GT,save_graph=True,save_graph_path = f'./trainable_graph/49_epoch/{iteration}_val_graph')  # (N, C, T, V)=(N, 2, 6, 120)
            ########################################################
            # Compute details for training
            ########################################################
            predicted = predicted * rescale_xy
            # output_loc_GT = output_loc_GT*rescale_xy

            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind - 1:ind + 1], dim=-2)
            predicted += ori_output_last_loc

            ### overall dist
            # overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask)
            # all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            all_overall_sum_list.extend(now_x2y2)

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    return all_overall_sum_list, all_overall_num_list

def val_model_load_graph(pra_model, pra_data_loader):
    # pra_model.to(dev)

    # load pre-trained params
    params = torch.load('./trained_models/model_epoch_0049.pt')


    # state_dicts_list = []
    # _count = 0
    # for _index,_part in enumerate(pra_model.st_gcn_networks):
    #     _temp_dict = {}
    #     for _state_key in _part.state_dict().keys():
    #         _temp_dict[_state_key] = params['xin_graph_seq2seq_model'][f'st_gcn_networks.{_index}.' + _state_key]
    #         _count += 1
    #     pra_model.st_gcn_networks[_index].load_state_dict(_temp_dict)
    #
    # _temp_dict = {}
    # for _,_part in enumerate(pra_model.seq2seq_car.state_dict().keys()):
    #     _temp_dict[_part] = params['xin_graph_seq2seq_model'][f'seq2seq_car.{_part}']
    #     _count += 1
    #
    # pra_model.seq2seq_car.load_state_dict(_temp_dict)
    #
    # _temp_dict = {}
    # for _,_part in enumerate(pra_model.seq2seq_human.state_dict().keys()):
    #     _temp_dict[_part] = params['xin_graph_seq2seq_model'][f'seq2seq_car.{_part}']
    #     _count += 1
    #
    # pra_model.seq2seq_human.load_state_dict(_temp_dict)
    #
    # _temp_dict = {}
    # for _,_part in enumerate(pra_model.seq2seq_bike.state_dict().keys()):
    #     _temp_dict[_part] = params['xin_graph_seq2seq_model'][f'seq2seq_car.{_part}']
    #     _count += 1
    #
    # pra_model.seq2seq_bike.load_state_dict(_temp_dict)
    #
    # _edge_import = {}
    # _edge_import['0'] = params['xin_graph_seq2seq_model']['edge_importance.0']
    # _edge_import['1'] = params['xin_graph_seq2seq_model']['edge_importance.1']
    # _edge_import['2'] = params['xin_graph_seq2seq_model']['edge_importance.2']
    # _edge_import['3'] = params['xin_graph_seq2seq_model']['edge_importance.3']
    #
    # pra_model.edge_importance.load_state_dict(_edge_import)

    _temp_dict = {}
    for _part in pra_model.state_dict().keys():
        if 'st_gcn_networks' in _part:
            _temp_dict[_part.split('st_gcn_networks')[-1][1:]] = params['xin_graph_seq2seq_model'][_part]

    pra_model.st_gcn_networks.load_state_dict(_temp_dict)

    pra_model.eval()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []

    # train model using training data
    for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
        # data: (N, C, T, V)
        # C = [Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,v_Class,
        # v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway]  + [mask]
        data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

        for now_history_frames in range(6, 7):
            input_data = data[:, :, :now_history_frames, :]  # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:, :2, now_history_frames:, :]  # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:, -1:, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:, :2, now_history_frames:, :]
            ori_output_last_loc = no_norm_loc_data[:, :2, now_history_frames - 1:now_history_frames, :]

            # for category
            cat_mask = ori_data[:, 10:11, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)
            predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                  pra_teacher_forcing_ratio=0,
                                  pra_teacher_location=output_loc_GT,graph=[torch.load(f'./trainable_graph/49_epoch/{iteration}_val_graph0_conv_block_graph.graph'),
                                                                            torch.load(f'./trainable_graph/49_epoch/{iteration}_val_graph1_conv_block_graph.graph'),
                                                                            torch.load(f'./trainable_graph/49_epoch/{iteration}_val_graph2_conv_block_graph.graph')],replace_graph=[False,False,False])  # (N, C, T, V)=(N, 2, 6, 120)
            ########################################################
            # Compute details for training
            ########################################################
            predicted = predicted * rescale_xy
            # output_loc_GT = output_loc_GT*rescale_xy

            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind - 1:ind + 1], dim=-2)
            predicted += ori_output_last_loc

            ### overall dist
            # overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask)
            # all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            all_overall_sum_list.extend(now_x2y2)

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    return all_overall_sum_list, all_overall_num_list

def test_model(pra_model, pra_data_loader):
    # pra_model.to(dev)
    pra_model.eval()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []
    with open(test_result_file, 'w') as writer:
        # train model using training data
        for iteration, (ori_data, A, mean_xy) in enumerate(pra_data_loader):
            # data: (N, C, T, V)
            # C = 11: [Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,
            # v_Class,v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway] + [1]/[0]
            data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
            input_data = data[:, :, :history_frames, :]  # (N, C, T, V)=(N, 4, 6, 120)
            output_mask = data[:, -1, -1, :]  # (N, V)=(N, 120)
            # print(data.shape, A.shape, mean_xy.shape, input_data.shape)

            ori_output_last_loc = no_norm_loc_data[:, :2, history_frames - 1:history_frames, :]

            A = A.float().to(dev)
            predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=future_frames, pra_teacher_forcing_ratio=0,
                                  pra_teacher_location=None)  # (N, C, T, V)=(N, 2, 6, 120)
            predicted = predicted * rescale_xy

            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind - 1:ind + 1], dim=-2)
            predicted += ori_output_last_loc

            now_pred = predicted.detach().cpu().numpy()  # (N, C, T, V)=(N, 2, 6, 120)
            now_mean_xy = mean_xy.detach().cpu().numpy()  # (N, 2)
            now_ori_data = ori_data.detach().cpu().numpy()  # (N, C, T, V)=(N, 11, 6, 120)
            now_mask = now_ori_data[:, -1, -1, :]  # (N, V)

            now_pred = np.transpose(now_pred, (0, 2, 3, 1))  # (N, T, V, 2)
            now_ori_data = np.transpose(now_ori_data, (0, 2, 3, 1))  # (N, T, V, 11)

            # print(now_pred.shape, now_mean_xy.shape, now_ori_data.shape, now_mask.shape)

            for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, now_ori_data, now_mask):
                # (6, 120, 2), (2,), (6, 120, 11), (120, )
                num_object = np.sum(n_mask).astype(int)
                # only use the last time of original data for ids (frame_id, object_id, object_type)
                # (6, 120, 11) -> (num_object, 3)
                n_dat = n_data[-1, :num_object, :3].astype(int)
                for time_ind, n_pre in enumerate(n_pred[:, :num_object], start=1):
                    # (120, 2) -> (n, 2)
                    # print(n_dat.shape, n_pre.shape)
                    for info, pred in zip(n_dat, n_pre + n_mean_xy):
                        information = info.copy()
                        information[0] = information[0] + time_ind
                        result = ' '.join(information.astype(str)) + ' ' + ' '.join(pred.astype(str)) + '\n'
                        # print(result)
                        writer.write(result)


def run_trainval_replace_params(pra_model, pra_traindata_path, pra_testdata_path, graph_args):
    loader_train = data_loader(pra_traindata_path, graph_args, pra_batch_size=batch_size_train, pra_shuffle=True,
                               pra_drop_last=True, train_val_test='train')
    # loader_test = data_loader(pra_testdata_path, pra_batch_size=batch_size_train, pra_shuffle=True, pra_drop_last=True,
    #                           train_val_test='all')

    # evaluate on testing data (observe 5 frame and predict 1 frame)
    loader_val = data_loader(pra_traindata_path, graph_args, pra_batch_size=batch_size_val, pra_shuffle=False, pra_drop_last=False,
                             train_val_test='val')

    optimizer = optim.Adam([{'params': model.parameters()}, ], )  # lr = 0.0001)

    # for now_epoch in range(total_epoch):
        # all_loader_train = itertools.chain(loader_train, loader_test)

        # my_print('#     - Train -    #', log_file)

        # train_model(pra_model, loader_train, pra_optimizer=optimizer,
        #             pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch))

        # train_model_save_weights(pra_model, loader_train, pra_optimizer=optimizer,
        #             pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch), current_epoch=now_epoch)

        # my_save_model(pra_model, now_epoch, work_dir)

    my_print('#     - Test -    #', log_file)

    display_result(
            val_model_load_graph(pra_model, loader_val), log_file,
            pra_pref='{}_Epoch{}'.format('Val', 0)
        )
        # if now_epoch == 1:
        #     display_result(
        #         val_model_replace_param(pra_model, loader_val, True,False), log_file,
        #         pra_pref='{}_Epoch{}'.format('Val', now_epoch)
        #     )
        # else:
        #     display_result(
        #         val_model_replace_param(pra_model, loader_val,False, False), log_file,
        #         pra_pref='{}_Epoch{}'.format('Val', now_epoch)
        #     )

def run_trainval(pra_model, pra_traindata_path, pra_testdata_path, graph_args):
    loader_train = data_loader(pra_traindata_path, graph_args, pra_batch_size=batch_size_train, pra_shuffle=True,
                               pra_drop_last=True, train_val_test='train')
    # loader_test = data_loader(pra_testdata_path, pra_batch_size=batch_size_train, pra_shuffle=True, pra_drop_last=True,
    #                           train_val_test='all')

    # evaluate on testing data (observe 5 frame and predict 1 frame)
    loader_val = data_loader(pra_traindata_path, graph_args, pra_batch_size=batch_size_val, pra_shuffle=False, pra_drop_last=False,
                             train_val_test='val')

    optimizer = optim.Adam([{'params': model.parameters()}, ], )  # lr = 0.0001)

    for now_epoch in range(total_epoch):
        # all_loader_train = itertools.chain(loader_train, loader_test)

        my_print('#     - Train -    #', log_file)
        train_model(pra_model, loader_train, pra_optimizer=optimizer,
                    pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch))
        # train_model_save_weights(pra_model, loader_train, pra_optimizer=optimizer,
        #             pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch), current_epoch=now_epoch)

        my_save_model(pra_model, now_epoch, work_dir)

        my_print('#     - Test -    #', log_file)

        display_result(
            val_model(pra_model, loader_val), log_file,
            pra_pref='{}_Epoch{}'.format('Val', now_epoch)
        )


def run_test(pra_model, pra_data_path):
    loader_test = data_loader(pra_data_path, pra_batch_size=batch_size_test, pra_shuffle=False, pra_drop_last=False,
                              train_val_test='test')
    test_model(pra_model, loader_test)


if __name__ == '__main__':
    graph_args = {'max_hop': 2, 'num_node': 200}
    # model = GRIPModel(in_channels=6, graph_args=graph_args, edge_importance_weighting=True)
    model = GRIPModel_with_replaced_graph_param(in_channels=6, graph_args=graph_args, edge_importance_weighting=True)
    model.to(dev)

    # pkl_files = [
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_0-6mins_noised_local_x_local_y.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_0-1mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_1-2mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_2-3mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_3-4mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_4-5mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_5-6mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_6-7mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_7-8mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_8-9mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_9-10mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_10-11mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_11-12mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_12-13mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_13-14mins_no_overlap.pkl',
    #     './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_14-15mins_no_overlap.pkl'
    #              ]

    pkl_files = [
        './training_data/smoothed_trajectories-0400-0415_deli_downsampled_by_4_0-6mins.pkl'
    ]


    # train and evaluate model
    for pkl in pkl_files:
        print(f'Starting to train on {pkl} ...')
        run_trainval_replace_params(model, pkl, './test_data.pkl', graph_args)
        # run_trainval_replace_params(model, pkl, './test_data.pkl',graph_args)
        # renamed = 'log_' + os.path.basename(pkl).split('.')[0] + '.txt'
        # os.rename('./trained_models/log.txt',f'./training_log/{renamed}')





