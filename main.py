import numpy as np
from math import sqrt

contents = []

with open('./smoothed_trajectories-0400-0415_deli_downsampled_by_4_0-1mins_no_overlap.csv', 'r') as f:
    contents = f.readlines()


frame_dict = {}
for _line in contents:
    _splitted = _line.split(',')
    frame = frame_dict.get(_splitted[1], [])
    frame.append([float(x) for x in _splitted])
    frame_dict[_splitted[1]] = frame

array_frame_dict = {}
for _key, _val in frame_dict.items():
    array_frame_dict[_key] = np.array(_val)

#
neighbor_dist = 1.0
for _key, _val in array_frame_dict.items():
    num_vehicles = _val.shape[0]
    num_features = _val.shape[1]
    feature_dict = {}
    adj_dict = {}
    for _v in range(num_vehicles):
        _temp_array = np.repeat(_val[_v:_v+1, :], num_vehicles, axis=0)
        _xy_diff = _val[:, 4:6] - _temp_array[:, 4:6]
        _xy_diff = _xy_diff[:, 0:1] ** 2 + _xy_diff[:, 1:2] ** 2
        _xy_diff = np.apply_along_axis(sqrt, 1, _xy_diff)

        #  A0 -> ego vehicle
        # A5 A1 A6
        # A3 A0 A4
        # A7 A2 A8
        _feature = np.zeros((9, num_features))
        _feature_settled = [False * 9]
        _sorted_indices = _xy_diff.argsort()
        _current_lane = _val[_v, 13]

        for _sorted in _sorted_indices:

            if not _feature[3]:
                if _val[_sorted, :][13] == _current_lane - 1 and _xy_diff[_sorted] > neighbor_dist:
                    _feature[3, :] = _val[_sorted, :]
                    _feature_settled[3] = True
                    if  

            if not _feature[4]:
                if _val[_sorted, :][13] == _current_lane + 1 and _xy_diff[_sorted] > neighbor_dist:
                    _feature[4, :] = _val[_sorted, :]
                    if

    # Add the
