import numpy as np 
import glob
import os 
from scipy import spatial 
import pickle
from datetime import datetime, timedelta
"""
A data process script on NGSIM dataset
"""

# Please change this to your location
data_root = './data/NGSIM'


history_frames = 6 # 3 second * 2 frame/second
future_frames = 6 # 3 second * 2 frame/second
total_frames = history_frames + future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 200 # maximum number of observed objects is 70
neighbor_distance = 10 # meter

# Baidu ApolloScape data format:
# frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
total_feature_dimension = 18 + 1 # we add mark "1" to the end of each row to indicate that this row exists

# after zero centralize data max(x)=127.1, max(y)=106.1, thus choose 130

def get_frame_instance_dict(pra_file_path):
	'''
	Read raw data from files and return a dictionary: 
		{frame_id: 
			{Vehicle_ID:
				# 18 features
				[Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_Length,v_Width,
				v_Class,v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway]
			}
		}
	'''
	with open(pra_file_path, 'r') as reader:
		# print(train_file_path)
		content = np.array([x.strip().split(',') for x in reader.readlines()]).astype(float)
		now_dict = {}
		for row in content:

			n_dict = now_dict.get(row[1], {})
			n_dict[row[0]] = row#[2:]
			now_dict[row[1]] = n_dict
	return now_dict

def process_data(pra_now_dict, iterated_id_set, pra_observed_last):
	visible_object_id_list = list(pra_now_dict[pra_observed_last].keys()) # object_id appears at the last observed frame
	num_visible_object = len(visible_object_id_list) # number of current observed objects

	# compute the mean values of x and y for zero-centralization. 
	visible_object_value = np.array(list(pra_now_dict[pra_observed_last].values()))
	xy = visible_object_value[:, 4:6].astype(float)
	mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
	m_xy = np.mean(xy, axis=0)
	mean_xy[4:6] = m_xy

	# compute distance between any pair of two objects
	dist_xy = spatial.distance.cdist(xy, xy)
	# if their distance is less than $neighbor_distance, we regard them are neighbors.
	neighbor_matrix = np.zeros((max_num_object, max_num_object))
	neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy<neighbor_distance).astype(int)

	now_all_object_id = set([val for x in iterated_id_set for val in pra_now_dict[x].keys()])
	non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
	num_non_visible_object = len(non_visible_object_id_list)

	# for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
	object_feature_list = []
	# non_visible_object_feature_list = []
	for frame_ind in iterated_id_set:
		# we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1] 
		# -mean_xy is used to zero_centralize data
		# now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
		now_frame_feature_dict = {obj_id : (list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] if obj_id in visible_object_id_list else list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[0]) for obj_id in pra_now_dict[frame_ind] }
		# if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
		now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in visible_object_id_list+non_visible_object_id_list])
		object_feature_list.append(now_frame_feature)

	# object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
	object_feature_list = np.array(object_feature_list)
	
	# object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
	object_frame_feature = np.zeros((max_num_object, len(iterated_id_set), total_feature_dimension))
	
	# np.transpose(object_feature_list, (1,0,2))
	object_frame_feature[:num_visible_object+num_non_visible_object] = np.transpose(object_feature_list, (1,0,2))

	return object_frame_feature, neighbor_matrix, m_xy
	

def generate_train_data(pra_file_path):
	'''
	Read data from $pra_file_path, and split data into clips with $total_frames length. 
	Return: feature and adjacency_matrix
		feture: (N, C, T, V) 
			N is the number of training data 
			C is the dimension of features, 10raw_feature + 1mark(valid data or not)
			T is the temporal length of the data. history_frames + future_frames
			V is the maximum number of objects. zero-padding for less objects. 
	'''
	now_dict = get_frame_instance_dict(pra_file_path)
	frame_id_set = sorted(set(now_dict.keys()))

	all_feature_list = []
	all_adjacency_list = []
	all_mean_list = []
	for _ind,start_ind in enumerate(frame_id_set[:-total_frames+1]):
		iterated_id_set = frame_id_set[_ind:_ind + total_frames]
		observed_last = frame_id_set[_ind + history_frames - 1]
		object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, iterated_id_set, observed_last)

		all_feature_list.append(object_frame_feature)
		all_adjacency_list.append(neighbor_matrix)	
		all_mean_list.append(mean_xy)	

	# (N, V, T, C) --> (N, C, T, V)
	all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
	all_adjacency_list = np.array(all_adjacency_list)
	all_mean_list = np.array(all_mean_list)
	# print(all_feature_list.shape, all_adjacency_list.shape)
	return all_feature_list, all_adjacency_list, all_mean_list


def generate_test_data(pra_file_path):
	now_dict = get_frame_instance_dict(pra_file_path)
	frame_id_set = sorted(set(now_dict.keys()))
	
	all_feature_list = []
	all_adjacency_list = []
	all_mean_list = []
	# get all start frame id
	start_frame_id_list = frame_id_set[::history_frames]
	for start_ind in start_frame_id_list:
		start_ind = int(start_ind)
		end_ind = int(start_ind + history_frames)
		observed_last = start_ind + history_frames - 1

		if observed_last > max(frame_id_set):
			continue

		# print(start_ind, end_ind)
		object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)

		all_feature_list.append(object_frame_feature)
		all_adjacency_list.append(neighbor_matrix)
		all_mean_list.append(mean_xy)

	# (N, V, T, C) --> (N, C, T, V)
	all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
	all_adjacency_list = np.array(all_adjacency_list)
	all_mean_list = np.array(all_mean_list)
	# print(all_feature_list.shape, all_adjacency_list.shape)
	return all_feature_list, all_adjacency_list, all_mean_list


def generate_data(pra_file_path_list, pra_is_train=True):
	all_data = []
	all_adjacency = []
	all_mean_xy = []
	for file_path in pra_file_path_list:
		if pra_is_train:
			now_data, now_adjacency, now_mean_xy = generate_train_data(file_path)
		else:
			now_data, now_adjacency, now_mean_xy = generate_test_data(file_path)
		all_data.extend(now_data)
		all_adjacency.extend(now_adjacency)
		all_mean_xy.extend(now_mean_xy)

	all_data = np.array(all_data) #(N, C, T, V)=(5010, 11, 12, 70) Train
	all_adjacency = np.array(all_adjacency) #(5010, 70, 70) Train
	all_mean_xy = np.array(all_mean_xy) #(5010, 2) Train

	# Train (N, C, T, V)=(5010, 11, 12, 70), (5010, 70, 70), (5010, 2)
	# Test (N, C, T, V)=(415, 11, 6, 70), (415, 70, 70), (415, 2)
	print(np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy))

	# save training_data and trainjing_adjacency into a file.
	if pra_is_train:
		save_path = 'train_data.pkl'
	else:
		save_path = 'test_data.pkl'
	with open(save_path, 'wb') as writer:
		pickle.dump([all_data, all_adjacency, all_mean_xy], writer)

def sample(str_list: list[str], interval: int):
	_count = 0
	_sampled_list = []

	for _, _count_str in enumerate(str_list):
		if _count < interval:
			_count += 1
			continue
		else:
			_count = 0
			_sampled_list.append(_count_str)

	_result_list = []
	_frame_id = None
	for _ind, _count_str in enumerate(_sampled_list):
		if _ind == 0:
			_result_list.append(_count_str)
			continue

		_temp_col_list = _count_str.split(',')
		_temp_col_list[1] = str(int(_result_list[-1].split(',')[1]) + 1)
		_result_list.append(','.join(_temp_col_list))


	return _result_list


def down_sample(file_path_list: list[str], interval: int, trunc_obj_id: int):

	for _count_file in file_path_list:

		dirname = os.path.dirname(_count_file)
		basename = os.path.basename(_count_file)
		tobesaved = basename.split('.')[0] + f'_downsampled_by_{interval}.' + basename.split('.')[-1]

		with open(_count_file, 'r') as r:
			_str_list = []
			w = open(os.path.join(dirname, tobesaved), 'w')
			_previous_obj_id = None
			while _count_line := r.readline():
				_temp_col_list = _count_line.split(',')
				if not _previous_obj_id:
					_previous_obj_id = _temp_col_list[0]
				else:

					if _previous_obj_id != _temp_col_list[0]:
						_str_list = sample(_str_list, interval)
						for _count_new_str in _str_list:
							w.write(_count_new_str)
						_str_list = []
						_previous_obj_id = _temp_col_list[0]

					else:
						_str_list.append(_count_line)
						_previous_obj_id = _temp_col_list[0]

			w.close()

		tobetrunc = basename.split('.')[0] + f'_trunc_by_{interval}.' + basename.split('.')[-1]
		with open(os.path.join(dirname, tobesaved), 'r') as r:
			w = open(os.path.join(dirname, tobetrunc), 'w')

			while _count_line := r.readline():
				if int(_count_line.split(',')[0]) > trunc_obj_id:
					break
				w.write(_count_line)

			w.close()

def split_dataset_with_overlap_by_1(file_path: list):

	for _count_file in file_path:
		with open(_count_file, 'r') as freader:
			content = freader.readlines()

			start_timestamp_1 = int(content[0].split(',')[3])
			end_timestamp_1 = start_timestamp_1 + 1000 * 60 * 6

			start_timestamp_2 = start_timestamp_1 + 1000 * 60 * 5
			end_timestamp_2 = start_timestamp_1 + 1000 * 60 * 8

			start_timestamp_3 = start_timestamp_1 + 1000 * 60 * 5
			end_timestamp_3 = start_timestamp_1 + 1000 * 60 * 11

			content_dict_with_object_id = {}
			for _count_line in content:
				cols = _count_line.split(',')
				_object_id = cols[0]
				object_content = content_dict_with_object_id.get(_object_id, [])
				object_content.append(_count_line)
				content_dict_with_object_id[_object_id] = object_content

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_0-6mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_1 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_1:
						wfile.write(_count_frame)
			wfile.close()

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_5-8mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_2<= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_2:
						wfile.write(_count_frame)
			wfile.close()

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_5-11mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_3 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_3:
						wfile.write(_count_frame)
			wfile.close()




def split_dataset_with_overlap_by_1_equal(file_path: list):

	for _count_file in file_path:
		with open(_count_file, 'r') as freader:
			content = freader.readlines()

			start_timestamp_1 = int(content[0].split(',')[3])
			end_timestamp_1 = start_timestamp_1 + 1000 * 60 * 3

			start_timestamp_2 = start_timestamp_1 + 1000 * 60 * 2
			end_timestamp_2 = start_timestamp_1 + 1000 * 60 * 5

			start_timestamp_3 = start_timestamp_1 + 1000 * 60 * 4
			end_timestamp_3 = start_timestamp_1 + 1000 * 60 * 7

			start_timestamp_4 = start_timestamp_1 + 1000 * 60 * 6
			end_timestamp_4 = start_timestamp_1 + 1000 * 60 * 9

			start_timestamp_5 = start_timestamp_1 + 1000 * 60 * 8
			end_timestamp_5 = start_timestamp_1 + 1000 * 60 * 11

			start_timestamp_6 = start_timestamp_1 + 1000 * 60 * 10
			end_timestamp_6 = start_timestamp_1 + 1000 * 60 * 13

			content_dict_with_object_id = {}
			for _count_line in content:
				cols = _count_line.split(',')
				_object_id = cols[0]
				object_content = content_dict_with_object_id.get(_object_id, [])
				object_content.append(_count_line)
				content_dict_with_object_id[_object_id] = object_content

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_0-3mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_1 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_1:
						wfile.write(_count_frame)
			wfile.close()

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_2-5mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_2<= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_2:
						wfile.write(_count_frame)
			wfile.close()

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_4-7mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_3 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_3:
						wfile.write(_count_frame)
			wfile.close()

			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_6-9mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_4 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_4:
						wfile.write(_count_frame)
			wfile.close()

			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_8-11mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_5 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_5:
						wfile.write(_count_frame)
			wfile.close()

			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_10-13mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_6 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_6:
						wfile.write(_count_frame)
			wfile.close()



def split_dataset_with_no_overlap(file_path: list):

	for _count_file in file_path:
		with open(_count_file, 'r') as freader:
			content = freader.readlines()

			start_timestamp_1 = int(content[0].split(',')[3])
			end_timestamp_1 = start_timestamp_1 + 1000 * 60 * 3

			start_timestamp_2 = start_timestamp_1 + 1000 * 60 * 3
			end_timestamp_2 = start_timestamp_1 + 1000 * 60 * 6

			start_timestamp_3 = start_timestamp_1 + 1000 * 60 * 6
			end_timestamp_3 = start_timestamp_1 + 1000 * 60 * 9

			start_timestamp_4 = start_timestamp_1 + 1000 * 60 * 9
			end_timestamp_4 = start_timestamp_1 + 1000 * 60 * 12

			start_timestamp_5 = start_timestamp_1 + 1000 * 60 * 12
			end_timestamp_5 = start_timestamp_1 + 1000 * 60 * 15

			content_dict_with_object_id = {}
			for _count_line in content:
				cols = _count_line.split(',')
				_object_id = cols[0]
				object_content = content_dict_with_object_id.get(_object_id, [])
				object_content.append(_count_line)
				content_dict_with_object_id[_object_id] = object_content

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_0-3mins_no_overlap.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_1 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_1:
						wfile.write(_count_frame)
			wfile.close()

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_3-6mins_no_overlap.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_2<= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_2:
						wfile.write(_count_frame)
			wfile.close()

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_6-9mins_no_overlap.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_3 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_3:
						wfile.write(_count_frame)
			wfile.close()

			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_9-12mins_no_overlap.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_4 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_4:
						wfile.write(_count_frame)
			wfile.close()

			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_12-15mins_no_overlap.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_5 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_5:
						wfile.write(_count_frame)
			wfile.close()

def split_dataset_with_short_time_length(file_path: list):

	for _count_file in file_path:
		with open(_count_file, 'r') as freader:
			content = freader.readlines()

			start_timestamp = int(content[0].split(',')[3])

			timestamps = []

			for _count in range(0,15):
				timestamps.append([start_timestamp + 1000 * 60 * 1 * _count , start_timestamp + 1000 * 60 * 1 * (_count + 1)])


			content_dict_with_object_id = {}
			for _count_line in content:
				cols = _count_line.split(',')
				_object_id = cols[0]
				object_content = content_dict_with_object_id.get(_object_id, [])
				object_content.append(_count_line)
				content_dict_with_object_id[_object_id] = object_content

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			for index,(start, end) in enumerate(timestamps):

				_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
											   ,os.path.basename(_count_file).split('.')[0] + f'_{index}-{index+1}mins_no_overlap.csv')
				wfile = open(_count_file_path_to_be_saved, 'w')
				for _count_object_id,_count_object_item in content_dict_with_object_id.items():
					for _count_frame in _count_object_item:
						if  start <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end:
							wfile.write(_count_frame)
				wfile.close()

def split_dataset_according_to_time(file_path: list):

	for _count_file in file_path:
		with open(_count_file, 'r') as freader:
			content = freader.readlines()

			# split into three datasets, each one with 6 mins, 3 mins of overlap
			start_timestamp_1 = int(content[0].split(',')[3])
			end_timestamp_1 = start_timestamp_1 + 1000 * 60 * 6

			start_timestamp_2 = end_timestamp_1 - 1000 * 60 * 3
			end_timestamp_2 = start_timestamp_2 + 1000 * 60 * 6

			start_timestamp_3 = end_timestamp_2 - 1000 * 60 * 3
			end_timestamp_3 = start_timestamp_3 + 1000 * 60 * 6

			content_dict_with_object_id = {}
			for _count_line in content:
				cols = _count_line.split(',')
				_object_id = cols[0]
				object_content = content_dict_with_object_id.get(_object_id, [])
				object_content.append(_count_line)
				content_dict_with_object_id[_object_id] = object_content

			# Here we get all content with
			# {object_id : { frame_id : content line }}
			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_first_3_mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if  start_timestamp_1<= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_1:
						wfile.write(_count_frame)
			wfile.close()

			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_second_3_mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if start_timestamp_2 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_2:
						wfile.write(_count_frame)
			wfile.close()

			_count_file_path_to_be_saved = os.path.join(os.path.dirname(_count_file)
										   ,os.path.basename(_count_file).split('.')[0] + '_third_3_mins.csv')
			wfile = open(_count_file_path_to_be_saved, 'w')
			for _count_object_id,_count_object_item in content_dict_with_object_id.items():
				for _count_frame in _count_object_item:
					if start_timestamp_3 <= int(_count_frame.split(',')[3]) and int(_count_frame.split(',')[3]) <= end_timestamp_3:
						wfile.write(_count_frame)
			wfile.close()


def insert_gaussian_noise(file_path: str):
	assert os.path.exists(file_path)
	poses = [4, 5]

	with open(file_path, 'r') as reader:
		# print(train_file_path)
		content = [turn_list_of_str_into_float(x.strip().split(',')) for x in reader.readlines()]
		content_dict_with_object = {}
		for row in content:
			n_dict = content_dict_with_object.get(row[0], {})
			n_dict[row[1]] = row#[2:]
			content_dict_with_object[row[0]] = n_dict


	for _object_id,_object_item in content_dict_with_object.items():
		sorted_frame_id_list = sorted(_object_item.keys())
		for _frame_index,_frame_id in enumerate(sorted_frame_id_list):

			_current_row = _object_item[_frame_id]
			if _frame_index + 1 >= len(sorted_frame_id_list):
				break
			_next_frame_id = sorted_frame_id_list[_frame_index + 1]
			_next_row = _object_item[_next_frame_id]

			_temp_averaged_list = [ ( _current_row[_col_index] + _next_row[_col_index] )/2   for _col_index in range(len(_object_item[_frame_id]))]

			for _count_pos in poses:
				noise = np.random.normal(_temp_averaged_list[_count_pos], 1)
				_temp_averaged_list[_count_pos] = noise

			_object_item[(_frame_id + _next_frame_id)/2] = _temp_averaged_list

	noised_file_name = file_path.split('.',-1)[0] + '_local_x_local_y.csv'
	with open(noised_file_name,'w') as writer:
		for _object_id, _object_item in content_dict_with_object.items():
			sorted_frame_id_list = sorted(_object_item.keys())
			for _frame_index, _frame_id in enumerate(sorted_frame_id_list):
				writer.write(','.join([str(_count_col) for _count_col in _object_item[_frame_id]]) + '\n')

def turn_list_of_str_into_float(target_list:list[str]):
	for _list_item_index,_list_item in enumerate(target_list):
		target_list[_list_item_index] = float(_list_item)

	return target_list

if __name__ == '__main__':
	# train_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_train/*.csv')))
	# test_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_test/*.csv')))

	# split_dataset_with_short_time_length(['./data/NGSIM/prediction_train/smoothed_trajectories-0400-0415_deli_downsampled_by_4.csv'])

	# down_sample(['./data/NGSIM/prediction_train/smoothed_trajectories-0515-0530_deli.csv'], 4, 1500)

	# print('Generating Training Data.')
	# generate_data(['./data/NGSIM/prediction_train/smoothed_trajectories-0400-0415_deli_downsampled_by_4_14-15mins_no_overlap.csv'], pra_is_train=True)

	# print('Generating Testing Data.')
	# generate_data(test_file_path_list, pra_is_train=False)

	insert_gaussian_noise('./data/NGSIM/prediction_train/smoothed_trajectories-0400-0415_deli_downsampled_by_4_0-1mins_no_overlap.csv')

