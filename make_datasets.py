import pandas as pd

train_url = 'http://azuremlsamples.azureml.net/templatedata/PM_train.txt'
test_url = 'http://azuremlsamples.azureml.net/templatedata/PM_test.txt'
truth_url = 'http://azuremlsamples.azureml.net/templatedata/PM_truth.txt'

feature_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']


def make_dataframe_from_source(url, truth_df=False):
    if truth_df:
        return pd.read_csv(url, header=None, names=['extra_cycles'])
    else:
        df = pd.read_csv(url, sep=" ", header=None)
        df.drop(df.columns[[26, 27]], axis=1, inplace=True)
        df.columns = feature_names
        return df

train_df = make_dataframe_from_source(train_url)
test_df = make_dataframe_from_source(test_url)
truth_df = make_dataframe_from_source(truth_url, True)

train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)
truth_df.to_csv('data/truth_data.csv', index=False)