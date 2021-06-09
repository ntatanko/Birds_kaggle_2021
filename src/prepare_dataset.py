# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
import numpy as np

def prepare_dataset(df, signal_lenght = 5):
    col = [
        "filename",
        "primary_label",
        "secondary_labels",
        "label_id",
        "secondary_labels_id",
        "start_sec",
        "end_sec",
        "row_id",
        "duration",
        "rating",
        "class_weights",
        "year",
        "file_path",
        "frames",
        "sin_month",
        "cos_month",
        "sin_longitude",
        "cos_longitude",
        "latitude",
        "norm_latitude",
        "date"
    ]
    import numpy as np
    import pandas as pd

    df = df[col]
    df[["start_sec", "end_sec", "rating"]] = df[["start_sec", "end_sec", "rating"]].astype("float16")
    df[["year", "label_id"]] = df[["year", "label_id"]].astype("int16")
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
    def my_floor(a, precision=2):
        dec = a - np.floor(a)
        dec = dec * 10 ** precision
        dec = np.floor(dec) / 10 ** precision
        b = np.floor(a) + dec
        return b
    df["end_sec"] = df["end_sec"].apply(my_floor)
    df['start_sec'] = df['end_sec']-signal_lenght
    return df

def make_dict_birds(df, secondary=True):
    dict_birds = {}
    list_primary = df["primary_label"].unique().tolist()
    if secondary:
        list_secondary = []
        for i in df["secondary_labels"]:
            if type(i) != float:
                i = i.split()
                list_secondary.extend(i)
        list_secondary = list(set(list_secondary))
        labels = sorted(list(set(list_primary + list_secondary)))
    else:
        labels = sorted(list(set(list_primary)))
    
    for i, bird in enumerate(labels):
        dict_birds[bird] = i
    df["label_id"] = df["primary_label"].replace(dict_birds)
    for i in df.index.tolist():
        if type(df.loc[i, "secondary_labels"]) != float:
            secondary_labels = df.loc[i, "secondary_labels"].split()
            list_ids = []
            for bird in secondary_labels:
                if bird in dict_birds.keys():
                    list_ids.append(str(dict_birds[bird]))
            df.loc[i, "secondary_labels_id"] = " ".join(list_ids)
    return  dict_birds, df


def choose_ids(distance_delta=600, months_delta=2, years_delta=5, start_year = 1980):
    import pandas as pd
    distances_df = pd.read_csv("/app/_data/distances.csv")
#     dates_sites = pd.read_csv('/app/_data/dates_sites.csv')
    df = distances_df.query('dist_COR <= @distance_delta or dist_SNE <= @distance_delta or dist_SSW <= @distance_delta or dist_COL <= @distance_delta')
    df = df.query('year >= @start_year').reset_index(drop=True)
    list_filenames = df['filename'].unique().tolist()
    return df, list_filenames


def make_intervals(array, sig_lenght=5, max_intervals=200, max_lenght=400):
    import pandas as pd
    def my_floor(a, precision=2):
        dec = a - np.floor(a)
        dec = dec * 10 ** precision
        dec = np.floor(dec) / 10 ** precision
        b = np.floor(a) + dec
        return b
    dict_intervals = {}

    for row in array:
        duration = row[1]
        filename = row[0]
        weight = row[2]
        if duration <= 10:
            step=0.3
        elif 10<duration<=20:
            step = 1
        elif 20<duration<=40:
            step = 1.5
        elif 40<duration<=max_lenght:
            step = 2
        if duration<=max_lenght:
            for i in np.arange(5, duration+0.1, step):
                end=my_floor(i)
                if end <= duration:
                    row_id = filename[:-4]+'_'+"_".join(str(end).split("."))
                    dict_intervals[row_id] = [end,weight, filename]
                else:
                    end = my_floor(duration - np.random.rand())
                    row_id = filename[:-4]+'_'+"_".join(str(end).split("."))
                dict_intervals[row_id] = [end, weight, filename]
        elif duration>max_lenght:
            for i in range(max_intervals):
                end = my_floor(np.random.randint(5,duration-2)+np.random.random())
                start = end-sig_lenght
                row_id = filename[:-4]+'_'+"_".join(str(end).split("."))
                dict_intervals[row_id] = [end, weight, filename]
    birds_intervals = pd.DataFrame(dict_intervals).T
    birds_intervals.columns = ['end_sec', 'class_weights', 'filename']
    birds_intervals['class_weights'] = birds_intervals['class_weights'].astype('float')
    birds_intervals['end_sec'] = birds_intervals['end_sec'].astype('float')
    return birds_intervals


def make_intervals_upsampling(array, sig_lenght=5, sum_intervals=500):
    import pandas as pd
    def my_floor(a, precision=2):
        dec = a - np.floor(a)
        dec = dec * 10 ** precision
        dec = np.floor(dec) / 10 ** precision
        b = np.floor(a) + dec
        return b
    dict_intervals = {}
    sum_lenght=array.sum(axis=0)[1]
    for row in array:
        duration = row[1]
        filename = row[0]
        weight = row[2]
        num_waves =np.ceil(duration*sum_intervals/sum_lenght)
        step = my_floor((duration-sig_lenght)/num_waves)
        for i in np.arange(sig_lenght, duration+step, step):
            end=my_floor(i)
            if end <= duration:
                row_id = filename[:-4]+'_'+"_".join(str(end).split("."))
                dict_intervals[row_id] = [end, weight, filename]
            else:
                end = my_floor(duration - np.random.rand())
                row_id = filename[:-4]+'_'+"_".join(str(end).split("."))
            dict_intervals[row_id] = [end, weight, filename]
    birds_intervals = pd.DataFrame(dict_intervals).T
    birds_intervals.columns = ['end_sec', 'class_weights', 'filename']
    birds_intervals['class_weights'] = birds_intervals['class_weights'].astype('float')
    birds_intervals['end_sec'] = birds_intervals['end_sec'].astype('float')
    return birds_intervals