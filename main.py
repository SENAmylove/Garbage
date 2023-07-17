import numpy as np
from math import sqrt

def get_frame_dict(file_path: str, coonfig: dict):
    """

    :param file_path:
    :return:
    """

    f = open('./smoothed_trajectories-0400-0415_deli_downsampled_by_4_0-1mins_no_overlap.csv', 'r')
    contents = f.readlines()
    f.close()

    frame_dict = {}
    for _line in contents:
        _splitted = _line.split(',')
        frame = frame_dict.get(_splitted[1], [])
        frame.append([float(x) for x in _splitted])
        frame_dict[_splitted[1]] = frame

    array_frame_dict = {}
    for _key, _val in frame_dict.items():
        array_frame_dict[_key] = np.array(_val)

    return array_frame_dict

def get_feature_dict(frame_dict: dict, config: dict):
    """

    :param frame_dict:
    :return:
    """
    frame_adj_dict = {}
    frame_feat_dict = {}
    neighbor_dist = config['NEIGHBOR_DISTANCE']

    for _key, _val in frame_dict.items():
        num_vehicles = _val.shape[0]
        num_features = _val.shape[1]
        _feat_dict = {}
        _adj_dict = {}
        # Iterate through all vehicles in this frame
        for _v in range(num_vehicles):
            _temp_array = np.repeat(_val[_v:_v+1, :], num_vehicles, axis=0)
            # Generate the distances for other vehicles to _v
            _xy_diff = _val[:, 4:6] - _temp_array[:, 4:6]
            _xy_diff = _xy_diff[:, 0:1] ** 2 + _xy_diff[:, 1:2] ** 2
            _xy_diff = np.apply_along_axis(sqrt, 1, _xy_diff)

            # Feature Matrix
            #  A0 -> ego vehicle
            # A5 A1 A6
            # A3 A0 A4
            # A7 A2 A8
            # Adjacency Matrix
            #   A0 A1 A2 A3 A4 A5 A6 A7 A8
            # A0
            # A1
            # A2
            # A3
            # A4
            # A5
            # A6
            # A7
            # A8
            _feature = np.zeros((9, num_features))
            _adj = np.eye(9)
            # Get the sorted val according to the distance
            _sorted_indices = _xy_diff.argsort()
            _sorted_val = _val[_sorted_indices]
            # Get the current lane
            _current_lane = _val[_v, 13]
            # Vehicle IDs
            _A1 = _val[_v, 14] if _val[_val[:, 0] == _val[:, 14]].shape[0] != 0 else 0
            _A2 = _val[_v, 15] if _val[_val[:, 0] == _val[:, 15]].shape[0] != 0 else 0
            _A3 = 0
            _A4 = 0
            _A5 = 0
            _A6 = 0
            _A7 = 0
            _A8 = 0
            #
            # Assign the A0 feature
            _feature[0, :] = _val[_v, :]
            #
            if _A1 != 0 and _val[_val[:, 0] == _val[:, 14]].shape[0] != 0:
                _feature[1, :] = _val[_val[:, 0] == _val[:, 14]]
            if _A2 != 0 and _val[_val[:, 0] == _val[:, 15]].shape[0] != 0:
                _feature[2, :] = _val[_val[:, 0] == _val[:, 15]]
            # Get the A3s
            _sorted_A3 = _sorted_val[_sorted_val[:, 13] == _current_lane - 1]
            if _sorted_A3.shape[0] == 0:
                # A3 does not exist
                # Set the vehicle ID to 0
                _A3 = 0
            else:
                # Get the vehicle ID
                _A3 = _sorted_A3[0, 0]
                _feature[3, :] = _sorted_A3[0, :]
                # Check the preceding vehicle
                if _sorted_A3[0, 14] == 0:
                    # A5 does not exist
                    _A5 = 0
                else:
                    if _val[_val[:, 0] == _sorted_A3[0, 14]].shape[0] != 0:
                        _A5 = _sorted_A3[0, 14]
                        _feature[5, :] = _val[_val[:, 0] == _A5]
                    else:
                        _A5 = 0

                # Check the following vehicle
                if _sorted_A3[0, 15] == 0:
                    # A7 does not exist
                    _A7 = 0
                else:
                    if _val[_val[:, 0] == _sorted_A3[0, 15]].shape[0] != 0:
                        _A7 = _sorted_A3[0, 15]
                        _feature[5, :] = _val[_val[:, 0] == _A7]
                    else:
                        _A7 = 0

            # Get the A4s
            _sorted_A4 = _sorted_val[_sorted_val[:, 13] == _current_lane - 1]
            if _sorted_A4.shape[0] == 0:
                # A4 does not exist
                # Set the vehicle ID to 0
                _A4 = 0
            else:
                # Get the vehicle ID
                _A4 = _sorted_A4[0, 0]
                _feature[4, :] = _sorted_A4[0, :]
                # Check the preceding vehicle
                if _sorted_A4[0, 14] == 0:
                    _A6 = 0
                else:
                    if _val[_val[:, 0] == _sorted_A3[0, 14]].shape[0] != 0:
                        _A6 = _sorted_A4[0, 14]
                        _feature[6, :] = _val[_val[:, 0] == _A6]
                    else:
                        _A6 = 0

                # Check the following vehicle
                if _sorted_A4[0, 15] == 0:
                    _A8 = 0
                else:
                    if _val[_val[:, 0] == _sorted_A3[0, 15]].shape[0] != 0:
                        _A8 = _sorted_A4[0, 15]
                        _feature[8, :] = _val[_val[:, 0] == _A8]
                    else:
                        _A8 = 0

            for _id, _neighbor in enumerate([_A1, _A2, _A3, _A4, _A5, _A6, _A7, _A8]):
                if _neighbor != 0:
                    _adj[0, _id + 1] = 1
            # Now we have constructed the feature matrix
            _feat_dict[str(_v)] = _feature
            _adj_dict[str(_v)] = _adj

        frame_feat_dict[_key + f'_feat'] = _feat_dict
        frame_adj_dict[_key + f'_adj'] = _adj_dict

    pass

if __name__ == '__main__':
    path = './smoothed_trajectories-0400-0415_deli_downsampled_by_4_0-1mins_no_overlap.csv'

    frames = get_frame_dict(path, {})
    get_feature_dict(frames, {'NEIGHBOR_DISTANCE':1.0})

