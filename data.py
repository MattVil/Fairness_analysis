import pandas as pd


PATH_DATA = "./data/adult.data"
DATASET_COLUMNS = ["age","workclass","fnlwgt","education","education_num",
                   "marital_status","occupation","relationship","race","sex",
                   "capital_gain","capital_loss","hours_per_week",
                   "native_country","income"]

def loadData():
    return pd.read_csv(PATH_DATA,names=DATASET_COLUMNS,sep=", ",engine='python')

if __name__ == '__main__':
    data = loadData()
    print(data)
