import pandas as pd

from aif360.datasets import AdultDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult


PATH_DATA = "./data/adult.data"
DATASET_COLUMNS = ["age","workclass","fnlwgt","education","education_num",
                   "marital_status","occupation","relationship","race","sex",
                   "capital_gain","capital_loss","hours_per_week",
                   "native_country","income"]

def loadData():
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    dataset_orig = load_preproc_data_adult(['sex'])

    optim_options = {
        "distortion_fun": get_distortion_adult,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }
    return privileged_groups, unprivileged_groups, optim_options, dataset_orig

if __name__ == '__main__':
    privileged_groups, unprivileged_groups, optim_options, data = loadData()
    print(data)
