import pandas as pd

from aif360.datasets import AdultDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
optim_options = {
    "distortion_fun": get_distortion_adult,
    "epsilon": 0.05,
    "clist": [0.99, 1.99, 2.99],
    "dlist": [.1, 0.05, 0]
}

def load_data():
    dataset_orig = load_preproc_data_adult(['sex'])
    return split_data(dataset_orig)

def split_data(data):
    dataset_orig_train, dataset_orig_vt = data.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
    return dataset_orig_train, dataset_orig_valid, dataset_orig_test


def print_dataset_infos(data):
    print("#### Training Dataset shape")
    print(data.features.shape)
    print("#### Favorable and unfavorable labels")
    print(data.favorable_label, data.unfavorable_label)
    print("#### Protected attribute names")
    print(data.protected_attribute_names)
    print("#### Privileged and unprivileged protected attribute values")
    print(data.privileged_protected_attributes,
          data.unprivileged_protected_attributes)
    print("#### Dataset feature names")
    print(data.feature_names)

if __name__ == '__main__':
    x_train, x_valid, x_test = load_data()
    print(x_train)
    print_dataset_infos(x_train)
