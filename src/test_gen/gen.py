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


class Functions:
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
        for filename in set(df["filename"]):  # type: ignore
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


class Mel_Provider:
    def __init__(
        self,
        n_fft,
        win_length,
        n_mels,
        sample_rate,
        mel_image_size,
        min_frequency,
        max_frequency,
        signal_lenght,
        hop_length=None,
        norm_mel_long=False,
        device="cpu",
    ):
        self.norm_mel_long = norm_mel_long
        self._device = device
        self.signal_lenght = signal_lenght
        self.sample_rate = sample_rate
        self.mel_image_size = mel_image_size
        if hop_length is None:
            self.hop_length = int(
                self.signal_lenght * self.sample_rate / (self.mel_image_size - 1)
            )
        else:
            self.hop_length = hop_length
        self._melspectrogram = torchaudio.transforms.MelSpectrogram(
            power=2.0,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            hop_length=self.hop_length,
            f_min=min_frequency,
            f_max=max_frequency,
        ).to(self._device)

    def msg(self, wave):
        wave = torch.tensor(wave.reshape([1, -1]).astype(np.float32)).to(self._device)
        mel_spec = self._melspectrogram(wave)[0].cpu().numpy()
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        if self.norm_mel_long:
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        mel_spec.astype(np.float32)
        return mel_spec


class Test_Kaggle:
    def __init__(
        self,
        path,
        df_coord_sites,
        dict_birds,
        n_fft,
        sample_rate,
        mel_image_size,
        signal_lenght,
        mel_provider,
        norm_mel_short = True,
        hop_length=None,
        device="cpu",
        img_dtype='uint8'
    ):
        self.path = path
        self._device = device
        self.signal_lenght = signal_lenght
        self.sample_rate = sample_rate
        self.mel_image_size = mel_image_size
        self.hop_length = int(
            self.signal_lenght * self.sample_rate / (self.mel_image_size - 1)
        )
        self.norm_mel_short = norm_mel_short
        self.mel_provider = mel_provider
        self.n_fft = n_fft
        self.df_coord_sites = df_coord_sites
        self.dict_birds = dict_birds
        self.img_dtype = img_dtype

    def make_df(self):
        path = self.path
        list_files = []
        for filename in os.listdir(path):
            if filename.split(".")[-1] == "ogg":
                list_files.append(filename)
                call, srt = librosa.load(path + filename, sr=self.sample_rate)
                duration = librosa.get_duration(
                    call,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=int(
                        self.signal_lenght
                        * self.sample_rate
                        / (self.mel_image_size - 1)
                    ),
                )
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
        df = df.merge(self.df_coord_sites, on="site", how="left")
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["sin_longitude"] = np.sin(2 * np.pi * (df["longitude"]) / 360)
        df["cos_longitude"] = np.cos(2 * np.pi * (df["longitude"]) / 360)
        df["norm_latitude"] = (df["latitude"] + 90) / 180
        df["audio_id"] = df["audio_id"]
        return df

    def get_audio(self, file_path):
        wave, sr = librosa.load(file_path, sr=self.sample_rate)
        return wave


    def make_prediction(self, df, model, thresh=0.5, predict=True, return_mels=False):
        path = self.path
        dict_row_id = {}
        predictions = {}
        for ix in df.index.tolist():
            wave_name = df.loc[ix, "filename"]
            audio_id = df.loc[ix, "audio_id"]
            site = df.loc[ix, "site"]
            wave = self.get_audio(path + wave_name)
            mel_spec = self.mel_provider.msg(wave)
            list_mels = []
            for end_sec in range(
                5, int(df.loc[ix, "duration"]) + 1, self.signal_lenght
            ):
                row_id = "_".join([str(audio_id), site, str(end_sec)])
                start = int(
                    ((end_sec - self.signal_lenght) * self.mel_image_size)
                    / self.signal_lenght
                )
                mel_short = mel_spec[:, start : start + self.mel_image_size]

                if self.norm_mel_short:
                    mel_short = (mel_short - np.min(mel_short)) / (
                np.max(mel_short) - np.min(mel_short)
            )
                else:
                    mel_short = mel_short
                
                if mel_short.shape != (self.mel_image_size, self.mel_image_size):
                    mel_short = Image.fromarray(mel_short)
                    mel_short = mel_short.resize(
                        (self.mel_image_size, self.mel_image_size),
                        Image.BICUBIC,
                    )
                    mel_short = np.array(mel_short)
                if self.img_dtype=='uint8':
                    max_value = 255
                    mel_short = np.round(mel_short * max_value)
                    mel_short = np.repeat(
                    np.expand_dims(mel_short.astype(np.uint8), 2), 3, 2
                )
                else:
                    max_value=1
                    mel_short = np.repeat(
                    np.expand_dims(mel_short.astype(np.float32), 2), 3, 2
                )

                #         sin_month
                mel_short[self.mel_image_size - 10:, :20, 0] = (
                    max_value * df.loc[ix, "sin_month"]
                )
                mel_short[self.mel_image_size - 10:, :20, 1] = max_value
                mel_short[self.mel_image_size - 10:, :20, 2] = max_value
        #         cos_month
                mel_short[self.mel_image_size - 10:, 20:40, 0] = max_value
                mel_short[self.mel_image_size - 10:, 20:40, 1] = (
                    max_value * df.loc[ix, "cos_month"]
                )
                mel_short[self.mel_image_size - 10:, 20:40, 2] = max_value
        #         year
                mel_short[self.mel_image_size - 10:, 40:60, 0] = max_value
                mel_short[self.mel_image_size - 10:, 40:60, 1] = max_value
                mel_short[self.mel_image_size - 10:, 40:60, 2] = (
                    max_value * (2021 - df.loc[ix, "year"])/50
                )
        #         sin_longitude
                mel_short[
                    self.mel_image_size - 10:,
                    self.mel_image_size - 60: self.mel_image_size - 40,
                    0,
                ] = (
                    max_value * df.loc[ix, "sin_longitude"]
                )
                mel_short[
                    self.mel_image_size - 10:,
                    self.mel_image_size - 60: self.mel_image_size - 40,
                    1,
                ] = max_value
                mel_short[
                    self.mel_image_size - 10:,
                    self.mel_image_size - 60: self.mel_image_size - 40,
                    2,
                ] = max_value
        #         cos_longitude
                mel_short[
                    self.mel_image_size - 10:,
                    self.mel_image_size - 40: self.mel_image_size - 20,
                    0,
                ] = max_value
                mel_short[
                    self.mel_image_size - 10:,
                    self.mel_image_size - 40: self.mel_image_size - 20,
                    1,
                ] = (
                    max_value * df.loc[ix, "cos_longitude"]
                )
                mel_short[
                    self.mel_image_size - 10:,
                    self.mel_image_size - 40: self.mel_image_size - 20,
                    2,
                ] = max_value
        #         norm_latitude
                mel_short[
                    self.mel_image_size - 10:, self.mel_image_size - 20:, 0
                ] = max_value
                mel_short[
                    self.mel_image_size - 10:, self.mel_image_size - 20:, 1
                ] = max_value
                mel_short[self.mel_image_size - 10:, self.mel_image_size - 20:, 2] = (
                    max_value * df.loc[ix, "norm_latitude"]
                )
                list_mels.append([row_id, mel_short])

                if predict:
                    mel_short = tf.expand_dims(mel_short, axis=0)
                    pred = model.predict(mel_short)[0]
                    dict_row_id[row_id] = wave_name
                    predictions[row_id] = pred
        predictions = pd.DataFrame(predictions).T
        predictions['row_id'] = predictions.index
        dict_row_id = pd.DataFrame(dict_row_id, index=[0]).T
        dict_row_id.columns=['filename']
        dict_row_id['row_id'] = dict_row_id.index
        pred_df = predictions.merge(dict_row_id, on='row_id')
        if predict:
            return pred_df
        if return_mels:
            return mel_spec, list_mels