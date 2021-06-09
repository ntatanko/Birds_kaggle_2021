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
import torch
import torchaudio
from PIL import Image
from tensorflow import keras


# -

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
        norm_mel=True,
        device="cpu",
        power=2,
    ):
        self.norm_mel = norm_mel
        self._device = device
        self.signal_lenght = signal_lenght
        self.sample_rate = sample_rate
        self.mel_image_size = mel_image_size
        self.power = power
        self.n_mels = n_mels
        if hop_length is None:
            self.hop_length = int(
                self.signal_lenght * self.sample_rate / (self.mel_image_size - 1)
            )
        else:
            self.hop_length = hop_length
        self._melspectrogram = torchaudio.transforms.MelSpectrogram(
            power=self.power,
            center=True,
            norm="slaney",
            onesided=True,
            win_length=win_length,
            pad_mode="reflect",
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
        if self.norm_mel:
            mel_spec = (mel_spec - np.min(mel_spec)) / (
                np.max(mel_spec) - np.min(mel_spec)
            )
        mel_spec = np.float16(mel_spec)
        return mel_spec


class MEL_Gen(keras.utils.Sequence):
    def __init__(
        self,
        df,
        n_mels,
        sample_rate,
        mel_image_size,
        signal_lenght,
        n_classes,
        seed,
        nocall_label_id,
        sin_cos_img=True,
        img_dtype="uint8",
        secondary_coeff=0.3,
        mel_provider=Mel_Provider,
        return_primary_labels=False,
        return_concat_labels=True,
        convert_to_rgb=True,
        wave_dir=None,
        short_mel_dir=None,
        batch_size=32,
        shuffle=True,
        augment=False,
        rand_aug=False
    ):
        self.mel_provider = mel_provider
        self.df = df.reset_index(drop=True)
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mel_image_size = mel_image_size
        self.signal_lenght = signal_lenght
        self.wave_dir = wave_dir
        self.short_mel_dir = short_mel_dir
        self.convert_to_rgb = convert_to_rgb
        self.batch_size = batch_size
        self._shuffle = shuffle
        self.nocall_label_id = nocall_label_id
        self.return_primary_labels = return_primary_labels
        self.return_concat_labels = return_concat_labels
        self.n_classes = n_classes
        self.seed = seed
        self.augment = augment
        self.img_dtype = img_dtype
        self.secondary_coeff = secondary_coeff
        self.sin_cos_img = sin_cos_img
        self.rand_aug = rand_aug
        if self._shuffle:
            self._shuffle_samples()
        if self.short_mel_dir is not None:
            if not os.path.exists(self.short_mel_dir):
                os.mkdir(self.short_mel_dir)
        if self.wave_dir is not None:
            if not os.path.exists(self.wave_dir):
                os.mkdir(self.wave_dir)

    def __len__(self):
        return self.df.shape[0] // self.batch_size

    def get_audio(
        self,
        file_path,
        end_sec=None,
    ):
        file_name = file_path.split("/")[-1][:-4]
        try:
            wave = np.load(self.wave_dir + file_name + ".npy")
        except:
            wave, sr = librosa.load(file_path, sr=self.sample_rate)
            if not os.path.isfile(self.wave_dir + file_name + ".npy"):
                np.save(self.wave_dir + file_name, wave)

        if end_sec is not None:
            if end_sec < self.signal_lenght:
                end_sec = self.signal_lenght
            end = int(end_sec * self.sample_rate)
            end = end if end < len(wave) else len(wave) - 100
            start = int(end - (self.signal_lenght * self.sample_rate))
            if start < 0:
                start = 0
                end = self.signal_lenght * self.sample_rate
            wave = wave[start:end]
        return wave

    def on_epoch_start(self):
        if self._shuffle:
            self._shuffle_samples()

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
        file_path = self.df.loc[ix, "file_path"]
        rating = self.df.loc[ix, "rating"]
        label_id = self.df.loc[ix, "label_id"]
        end_sec = self.df.loc[ix, "end_sec"]
        row_id = self.df.loc[ix, "row_id"]

        x = {}
        try:
            mel_spec = np.load(self.short_mel_dir + row_id + ".npy")
        except:
            wave = self.get_audio(file_path, end_sec)
            mel_spec = self.mel_provider.msg(wave)

            if mel_spec.shape != (self.n_mels, self.mel_image_size):
                mel_spec = Image.fromarray(mel_spec)
                mel_spec = mel_spec.resize(
                    (self.n_mels, self.mel_image_size),
                    Image.BICUBIC,
                )
                mel_spec = np.array(mel_spec)

            mel_spec = np.float16(mel_spec)
            if not self.augment and self.img_dtype == "uint8":
                mel_spec = np.round(mel_spec * 255).astype('uint8')
                if not os.path.isfile(self.short_mel_dir + row_id + ".npy"):
                    np.save(self.short_mel_dir + row_id, mel_spec)
            else:
                if not os.path.isfile(self.short_mel_dir + row_id + ".npy"):
                    np.save(self.short_mel_dir + row_id, mel_spec)

        if self.augment:
            if np.random.rand() > 0.9:
                mel_spec = self.mix_class(mel=mel_spec, class_idx=label_id)
            if self.rand_aug:
                if np.random.rand() < 0.05:
                    mel_spec = self.noise(mel_spec)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())

        if self.convert_to_rgb:
            if self.img_dtype == "uint8":
                mel_spec = np.round(mel_spec * 255).astype('uint8')
                mel_spec = np.repeat(np.expand_dims(mel_spec.astype(np.uint8), 2), 3, 2)
            else:
                mel_spec = np.repeat(
                    np.expand_dims(mel_spec.astype(np.float16), 2), 3, 2
                )
            if self.sin_cos_img:
                mel_spec = self.sin_cos(mel_spec, ix)

        primary_y = np.zeros(self.n_classes)
        secondary_y = np.zeros(self.n_classes)
        assert (
            self.return_primary_labels + self.return_concat_labels == 1
        ), "only one of return_primary_labels or return_concat_labels can be True"

        primary_y[label_id] = 1
        if self.return_primary_labels:
            y = primary_y
        # !!!!!!!!! float
        if type(self.df.loc[ix, "secondary_labels_id"]) != float:
            if self.df.loc[ix, "secondary_labels_id"] != '':
                for i in self.df.loc[ix, "secondary_labels_id"].split(" "):
                    i = int(i)
                    secondary_y[i] = self.secondary_coeff

        if self.return_concat_labels:
            y = primary_y + secondary_y
            y = np.where(y > 1, 1, y)

        assert mel_spec.shape == (self.n_mels, self.mel_image_size, 3) or (
            self.n_mels,
            self.mel_image_size,
        )
        x["mel"] = mel_spec
        x["data"] = np.array(
            [
                self.df.loc[ix, "sin_longitude"],
                self.df.loc[ix, "cos_longitude"],
                self.df.loc[ix, "norm_latitude"],
                self.df.loc[ix, "sin_month"],
                self.df.loc[ix, "cos_month"],
                (2022 - self.df.loc[ix, "year"]) / 50,
            ]
        )
        return x, y

    def _shuffle_samples(self):
        self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def noise(self, mel):
        level_noise = 0.05

        def random_power(mel, power=1.5, c=0.6):
            mel = mel ** (np.random.random() * power + c)
            return mel

        if np.random.random() < 0.4:
            mel = random_power(mel)
        if np.random.random() < 0.2:
            mel = mel + (
                np.random.sample((self.n_mels, self.mel_image_size)).astype(np.float16)
                + 8
            ) * mel.mean() * level_noise * (np.random.sample() + 0.2)
        if np.random.random() < 0.2:
            r = np.random.randint(self.n_mels // 2, self.n_mels)
            x = np.random.random() / 2
            pink_noise = np.array(
                [
                    np.concatenate(
                        (1 - np.arange(r) * x / r, np.zeros(self.n_mels - r) - x + 1)
                    )
                ]
            ).T
            mel = mel * pink_noise
        if np.random.random() < 0.2:
            a = np.random.randint(0, self.mel_image_size - 12)
            b = np.random.randint(a + 11, self.mel_image_size)
            mel[:, a:b] = mel[:, a:b] + (
                np.random.sample((self.n_mels, b - a)).astype(np.float16) + 9
            ) * 0.2 * mel.mean() * level_noise * (np.random.sample() + 0.9)
        return mel

    def mix_class(self, mel, class_idx):
        if np.random.randint(0, 2) == 1:
            class_idx = class_idx
        else:
            class_idx = self.nocall_label_id
        data = self.df[self.df["label_id"] == class_idx].sample(1)
        file_path = data["file_path"].values[0]
        label_id = data["label_id"].values[0]
        end_sec = data["end_sec"].values[0]
        row_id = data["row_id"].values[0]
        try:
            mel_mix = np.load(self.short_mel_dir + row_id + ".npy")
        except:
            wave = self.get_audio(file_path, end_sec)
            mel_mix = self.mel_provider.msg(wave)

            if mel_mix.shape != (self.n_mels, self.mel_image_size):
                mel_mix = Image.fromarray(mel_mix)
                mel_mix = mel_mix.resize(
                    (self.n_mels, self.mel_image_size),
                    Image.BICUBIC,
                )
                mel_mix = np.array(mel_mix)
            if not os.path.isfile(self.short_mel_dir + row_id + ".npy"):
                np.save(self.short_mel_dir + row_id, mel_mix)
        mel = mel + mel_mix * np.random.uniform(0.6, 0.9)
        return mel

    def __getitem__(self, batch_ix):

        if self.sin_cos_img:
            x, y = [], []
            for i in range(self.batch_size):
                x_, y_ = self._get_one(
                    i + self.batch_size * batch_ix,
                )
                x.append(x_)
                y.append(y_)
            x = np.array(x)
            y = np.array(y)

        else:
            x, y = {}, []
            b_x_mel = []
            b_x_data = []
            for i in range(self.batch_size):
                x_dict, y_ = self._get_one(i + self.batch_size * batch_ix)
                b_x_mel.append(x_dict["mel"])
                b_x_data.append(x_dict["data"])
                y.append(y_)
            x["mel"] = np.array(b_x_mel)
            x["data"] = np.array(b_x_data)
            y = np.array(y)

        return x, y
