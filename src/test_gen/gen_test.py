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

# +
import os

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import torch
import torchaudio
from PIL import Image
from tensorflow import keras


# -

class Functions:
    def make_df(path, sample_rate, df_coord_sites):
        list_files = []
        for filename in os.listdir(path):
            if filename.split(".")[-1] == "ogg":
                list_files.append(filename)
                wave, _ = librosa.load(path + filename, sr=sample_rate)
                duration = len(wave) / sample_rate
        df = pd.DataFrame()
        for filename in list_files:
            df.loc[filename, "filename"] = filename
            df.loc[filename, "audio_id"] = filename.split("_")[0]
            df.loc[filename, "site"] = filename.split("_")[1]
            df.loc[filename, "date"] = filename.split("_")[2].split(".")[0]
            df.loc[filename, "duration"] = duration
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df = df.merge(df_coord_sites, on="site", how="left")
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["sin_longitude"] = np.sin(2 * np.pi * (df["longitude"]) / 360)
        df["cos_longitude"] = np.cos(2 * np.pi * (df["longitude"]) / 360)
        df["norm_latitude"] = (df["latitude"] + 90) / 180
        df["audio_id"] = df["audio_id"]
        df = df.reset_index(drop=True)
        long_df = pd.DataFrame(columns=["row_id", "end_sec", "filename"])
        for i in df.index.tolist():
            audio_id = df.loc[i, "audio_id"]
            duration = df.loc[i, "duration"]
            site = df.loc[i, "site"]
            for end_sec in range(5, int(duration) + 1, 5):
                row_id = "_".join([str(audio_id), site, str(end_sec)])
                long_df.loc[row_id, "row_id"] = row_id
                long_df.loc[row_id, "end_sec"] = end_sec
                long_df.loc[row_id, "filename"] = df.loc[i, "filename"]
        df = long_df.merge(df, on="filename", how="left")
        df[['month', 'year']]
        return df

    def row_wise_f1_score_micro(y_true, y_pred):
        F1 = []
        for preds, trues in zip(y_pred, y_true):
            TP, FN, FP = 0, 0, 0
            preds = preds.split()
            trues = trues.split()
            for true in trues:
                if true in preds:
                    TP += 1
                else:
                    FN += 1
            for pred in preds:
                if pred not in trues:
                    FP += 1
            F1.append(2 * TP / (2 * TP + FN + FP))
        return np.mean(F1)

    def boost_multiple_occurences(
        df,
        labels,
        pred_col,
        out_col="y_pred",
        boost_coef=1.1,
        max_boost_coef=12,
        threshold=0.5,
    ):
        """
        Boost predictions in file:
            - if something occured once, multiply that class by boost_coef
            - if something occured more than once - keep multiplying until
                boost_coef reaches max_boost_coef
        """

        def _compute_boost_matrix(
            y_preds, labels, threshold, boost_coef, max_boost_coef
        ):
            nocall_ix = labels.index("nocall")
            boost_matrix = np.ones((len(labels)), dtype=np.float64)
            for p in y_preds:
                boost_matrix = boost_matrix * np.where(p > threshold, boost_coef, 1.0)
                boost_matrix = np.clip(boost_matrix, 1.0, max_boost_coef)
                boost_matrix[nocall_ix] = 1.0
            return boost_matrix

        dict_pred = {}
        for filename in set(df["filename"]):
            file_df = df[df.filename == filename]
            file_y_preds = file_df[pred_col].values
            list_row_id = file_df["row_id"].values
            bm = _compute_boost_matrix(
                file_y_preds,
                labels=labels,
                threshold=threshold,
                boost_coef=boost_coef,
                max_boost_coef=max_boost_coef,
            )

            file_y_preds = bm * file_y_preds
            for i in range(len(list_row_id)):
                dict_pred[list_row_id[i]] = file_y_preds[i]
        return dict_pred

    def pred_from_dict(df, cols, labels, thresh=0.5, as_is=False):
        submission = pd.DataFrame(columns=["row_id", "birds"])
        for ix in df.index.tolist():
            prediction = df.loc[ix, cols].values
            row_id = df.loc[ix, "row_id"]
            nocall_ix = labels.index("nocall")
            submission.loc[ix, "row_id"] = row_id
            if as_is:
                birds = " ".join(
                    [labels[i] for i in range(len(labels)) if prediction[i] > thresh]
                )
                submission.loc[ix, "birds"] = birds
            else:
                if np.argmax(prediction) == nocall_ix:
                    if np.sum(prediction > thresh) < 3:
                        submission.loc[ix, "birds"] = "nocall"
                    else:
                        prediction[nocall_ix] = 0
                        birds = " ".join(
                            [
                                labels[i]
                                for i in range(len(labels))
                                if prediction[i] > thresh
                            ]
                        )
                else:
                    birds = " ".join(
                        [
                            labels[i]
                            for i in range(len(labels))
                            if prediction[i] > thresh and i != nocall_ix
                        ]
                    )
                    submission.loc[ix, "birds"] = birds
        submission["birds"] = submission["birds"].replace("", "nocall")
        submission["birds"] = submission["birds"].fillna("nocall")
        return submission



class Test_Kaggle(keras.utils.Sequence):
    def __init__(
        self,
        df,
        mel_long,
        mel_image_size,
        n_mels,
        signal_lenght,
        batch_size=1,
        img_dtype="uint8",
        sin_cos_img=True,
        img_year = True
    ):
        self.df = df
        self.mel_long = mel_long
        self.signal_lenght = signal_lenght
        self.mel_image_size = mel_image_size
        self.img_dtype = "uint8"
        self.n_mels = n_mels
        self.sin_cos_img = sin_cos_img
        self.batch_size = batch_size
        self.img_year = img_year
    def __len__(self):
        return self.df.shape[0] // self.batch_size

    def sin_cos(self, mel_spec, ix):
        if self.img_dtype == "uint8":
            max_value = 255
        else:
            max_value = 1
        min_value = 0
        #         sin_month
        mel_spec[self.n_mels - 10 :, :20, 0] = max_value * self.df.loc[ix, "sin_month"]
        mel_spec[self.n_mels - 10 :, :20, 1] = max_value
        mel_spec[self.n_mels - 10 :, :20, 2] = max_value
        #         cos_month
        mel_spec[self.n_mels - 10 :, 20:40, 0] = max_value
        mel_spec[self.n_mels - 10 :, 20:40, 1] = (
            max_value * self.df.loc[ix, "cos_month"]
        )
        mel_spec[self.n_mels - 10 :, 20:40, 2] = max_value
        #         year
        if self.img_year:
            mel_spec[self.n_mels - 10 :, 40:60, 0] = max_value
            mel_spec[self.n_mels - 10 :, 40:60, 1] = max_value
            mel_spec[self.n_mels - 10 :, 40:60, 2] = (
                max_value * (2021 - self.df.loc[ix, "year"]) / 50
            )
        #         sin_longitude
        mel_spec[
            self.n_mels - 10 :,
            self.mel_image_size - 60 : self.mel_image_size - 40,
            0,
        ] = (
            max_value * self.df.loc[ix, "sin_longitude"]
        )
        mel_spec[
            self.n_mels - 10 :,
            self.mel_image_size - 60 : self.mel_image_size - 40,
            1,
        ] = max_value
        mel_spec[
            self.n_mels - 10 :,
            self.mel_image_size - 60 : self.mel_image_size - 40,
            2,
        ] = max_value
        #         cos_longitude
        mel_spec[
            self.n_mels - 10 :,
            self.mel_image_size - 40 : self.mel_image_size - 20,
            0,
        ] = max_value
        mel_spec[
            self.n_mels - 10 :,
            self.mel_image_size - 40 : self.mel_image_size - 20,
            1,
        ] = (
            max_value * self.df.loc[ix, "cos_longitude"]
        )
        mel_spec[
            self.n_mels - 10 :,
            self.mel_image_size - 40 : self.mel_image_size - 20,
            2,
        ] = max_value
        #         norm_latitude
        mel_spec[self.n_mels - 10 :, self.mel_image_size - 20 :, 0] = max_value
        mel_spec[self.n_mels - 10 :, self.mel_image_size - 20 :, 1] = max_value
        mel_spec[self.n_mels - 10 :, self.mel_image_size - 20 :, 2] = (
            max_value * self.df.loc[ix, "norm_latitude"]
        )
        return mel_spec

    def _get_one(self, ix):
        end_sec = self.df.loc[ix, "end_sec"]
        start = int((end_sec - 5) * (self.mel_image_size / self.signal_lenght))
        end = start + self.mel_image_size
        mel_short = self.mel_long[:, start:end]
        mel_short = (mel_short - np.min(mel_short)) / (
            np.max(mel_short) - np.min(mel_short)
        )

        if mel_short.shape != (self.n_mels, self.mel_image_size):
            mel_short = Image.fromarray(mel_short)
            mel_short = mel_short.resize(
                (self.mel_image_size, self.n_mels),
                Image.BICUBIC,
            )
            mel_short = np.array(mel_short)
        if self.img_dtype == "uint8":
            max_value = 255
            mel_short = np.round(mel_short * max_value)
            mel_short = np.repeat(np.expand_dims(mel_short.astype(np.uint8), 2), 3, 2)
        else:
            max_value = 1
            mel_short = np.repeat(np.expand_dims(mel_short.astype(np.float16), 2), 3, 2)
        if self.sin_cos_img:
            mel_short = self.sin_cos(mel_short, ix)
            x = mel_short

        else:
            features = np.array(
                [
                    self.df.loc[ix, "sin_longitude"],
                    self.df.loc[ix, "cos_longitude"],
                    self.df.loc[ix, "norm_latitude"],
                    self.df.loc[ix, "sin_month"],
                    self.df.loc[ix, "cos_month"],
                    (2022 - self.df.loc[ix, "year"]) / 50,
                ]
            )
            x = {"mel": mel_short, "data": features}
        y = 1
        return x, y

    def __getitem__(self, batch_ix):

        if self.sin_cos_img:
            if self.img_dtype == "uint8":
                x = np.zeros(
                    (self.batch_size, self.n_mels, self.mel_image_size, 3),
                    dtype=np.uint8,
                )
            else:
                x = np.zeros(
                    (self.batch_size, self.n_mels, self.mel_image_size, 3),
                    dtype=np.float16,
                )

            y = np.zeros(
                (self.batch_size, 1),
                dtype=np.float16,
            )

            for i in range(self.batch_size):
                x[i], y[i] = self._get_one(
                    i + self.batch_size * batch_ix,
                )

        else:
            b_x_dict, b_y = {}, []
            for i in range(self.batch_size):
                x_dict, y = self._get_one(i + self.batch_size * batch_ix)

                # single x is dictionary of <input>:<value>
                # but, batch needs to be a dictionaty of <input>:np.array(<values>)
                for k, v in x_dict.items():
                    if k not in b_x_dict:
                        b_x_dict[k] = []
                    b_x_dict[k].append(v)

                b_y.append(y)

            for k, v in b_x_dict.items():
                b_x_dict[k] = np.array(v)

            b_y = np.array(b_y)
            x = b_x_dict
            y = b_y

        return x, y


