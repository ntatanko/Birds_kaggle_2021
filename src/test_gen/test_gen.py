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

import os
import numpy as np
import pandas as pd
import torchaudio
import torch
import librosa
from PIL import Image
from tensorflow import keras

TEST_AUDIO_PATH = '../input/birdclef-2021/test_soundscapes/'
TRAIN_AUDIO_PATH = '../input/birdclef-2021/train_soundscapes/'
SAMPLE_RATE = 32000
SEED = 42
IMG_SIZE = 260
N_FFT = 2048
SIGNAL_LENGTH = 5  # seconds
FREQ_MIN = 500
FREQ_MAX = 15000
WIN_LENGHT = 1024
BATCH_SIZE = 128

def make_df(path):
    list_files = []
    for filename in os.listdir(path):
        if filename.split(".")[-1] == "ogg":
            list_files.append(filename)
            call, srt = librosa.load(path + filename, sr=SAMPLE_RATE)
            duration = librosa.get_duration(
                call,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=int(SIGNAL_LENGTH * SAMPLE_RATE / (IMG_SIZE - 1),
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
    df = df.merge(coord_sites, on="site", how="left")
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_longitude"] = np.sin(2 * np.pi * (df["longitude"]) / 360)
    df["cos_longitude"] = np.cos(2 * np.pi * (df["longitude"]) / 360)
    df["norm_latitude"] = (df["latitude"] + 90) / 180
    df["audio_id"] = df["audio_id"].astype("int")
    return df


# +
def get_audio(
    file_path
):
    wave, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return wave


def _melspectrogram(wave):
    mel_spec_func = torchaudio.transforms.MelSpectrogram(
        power=2.0,
        win_length=WIN_LENGHT,
        n_fft=N_FFT,
        n_mels=IMG_SIZE,
        sample_rate=SAMPLE_RATE,
        hop_length=int(SIGNAL_LENGTH * SAMPLE_RATE / (IMG_SIZE - 1)),
        f_min=FREQ_MIN,
        f_max=FREQ_MAX,
    ).to("cpu")

    wave = torch.tensor(wave.reshape([1, -1]).astype(np.float32)).to("cpu")
    mel_spec = mel_spec_func(wave)[0].cpu().numpy()
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec))
    mel_spec.astype(np.float32)
    return mel_spec


# -

def make_prediction(df, path, thresh=0.5, predict=True, return_mels=False):
    model = keras.models.load_model("../input/eff01initial/eff0_1.h5")
    pred_df = pd.DataFrame(columns=["filename", "row_id", "y_pred", "birds"])
    pred_df["y_pred"] = pred_df["y_pred"].astype("object")
    predictions = {}
    for ix in df.index.tolist():
        wave_name = df.loc[ix, "filename"]
        audio_id = df.loc[ix, "audio_id"]
        site = df.loc[ix, "site"]
        wave = get_audio(path + wave_name)
        mel_spec = _melspectrogram(wave)
        list_mels = []
        for end_sec in range(5, int(df.loc[ix, "duration"]) + 1, SIGNAL_LENGTH):
            row_id = "_".join([str(audio_id), site, str(end_sec)])
            start = int(((end_sec - SIGNAL_LENGTH) * IMG_SIZE) / SIGNAL_LENGTH)
            mel_short = mel_spec[:, start : start + IMG_SIZE]

            mel_short = mel_short * 255
            if mel_short.shape != (IMG_SIZE, IMG_SIZE):
                mel_short = Image.fromarray(mel_short)
                mel_short = mel_short.resize(
                    (IMG_SIZE, IMG_SIZE),
                    Image.BICUBIC,
                )
                mel_short = np.array(mel_short)
            mel_short = np.repeat(np.expand_dims(mel_short.astype(np.uint8), 2), 3, 2)
            mel_short[IMG_SIZE - 15 :, :20, 0] = 255 * df.loc[ix, "sin_month"]
            mel_short[IMG_SIZE - 15 :, :20, 1] = 255
            mel_short[IMG_SIZE - 15 :, :20, 2] = 0
            mel_short[IMG_SIZE - 15 :, 20:40, 0] = 255
            mel_short[IMG_SIZE - 15 :, 20:40, 1] = 255 * df.loc[ix, "cos_month"]
            mel_short[IMG_SIZE - 15 :, 20:40, 2] = 0
            mel_short[IMG_SIZE - 15 :, IMG_SIZE - 60 : IMG_SIZE - 40, 0,] = (
                255 * df.loc[ix, "sin_longitude"]
            )
            mel_short[
                IMG_SIZE - 15 :,
                IMG_SIZE - 60 : IMG_SIZE - 40,
                1,
            ] = 255
            mel_short[
                IMG_SIZE - 15 :,
                IMG_SIZE - 60 : IMG_SIZE - 40,
                2,
            ] = 255
            mel_short[
                IMG_SIZE - 15 :,
                IMG_SIZE - 40 : IMG_SIZE - 20,
                0,
            ] = 255
            mel_short[IMG_SIZE - 15 :, IMG_SIZE - 40 : IMG_SIZE - 20, 1,] = (
                255 * df.loc[ix, "cos_longitude"]
            )
            mel_short[
                IMG_SIZE - 15 :,
                IMG_SIZE - 40 : IMG_SIZE - 20,
                2,
            ] = 255
            mel_short[IMG_SIZE - 15 :, IMG_SIZE - 20 :, 0] = 255
            mel_short[IMG_SIZE - 15 :, IMG_SIZE - 20 :, 1] = 255
            mel_short[IMG_SIZE - 15 :, IMG_SIZE - 20 :, 2] = (
                255 * df.loc[ix, "norm_latitude"]
            )
            list_mels.append(mel_short)

            if predict:
                mel_short = tf.expand_dims(mel_short, axis=0)
                pred = model.predict(mel_short)[0]
                list_birds = " ".join(
                    [
                        dict_birds_code[i]
                        for i in range(len(dict_birds_code))
                        if pred[i] > thresh
                    ]
                )
                pred_df.loc[row_id, "filename"] = wave_name
                pred_df.loc[row_id, "row_id"] = row_id
                pred_df.loc[row_id, "y_pred"] = np.array(pred)
                pred_df.loc[row_id, "birds"] = list_birds
                predictions[row_id] = pred
    if predict:
        return pred_df, predictions
    if return_mels:
        return mel_spec, list_mels, row_id


def boost_multiple_occurences(
    df,
    labels,
    pred_col="y_pred",
    out_col="y_pred",
    boost_coef=2.4,
    max_boost_coef=24,
    threshold=0.5,
):
    """
    Boost predictions in file:
        - if something occured once, multiply that class by boost_coef
        - if something occured more than once - keep multiplying until
            boost_coef reaches max_boost_coef
    """

    def _compute_boost_matrix(y_preds, labels, threshold, boost_coef, max_boost_coef):

        nocall_ix = labels.index("nocall")
        boost_matrix = np.ones((len(labels)), dtype=np.float64)
        for p in y_preds:
            boost_matrix = boost_matrix * np.where(p > threshold, boost_coef, 1.0)
            boost_matrix = np.clip(boost_matrix, 1.0, max_boost_coef)
            boost_matrix[nocall_ix] = 1.0

        return boost_matrix

    res_df = pd.DataFrame()

    for filename in set(df["filename"]):  # type: ignore

        file_df = df[df.filename == filename]
        file_y_preds = np.stack(file_df[pred_col].values, axis=0)
        bm = _compute_boost_matrix(
            file_y_preds,
            labels=labels,
            threshold=threshold,
            boost_coef=boost_coef,
            max_boost_coef=max_boost_coef,
        )

        file_y_preds = bm * file_y_preds

        file_df[out_col] = list(map(lambda x: x, file_y_preds))
        res_df = res_df.append(file_df)

    return res_df.reset_index(drop=True)


