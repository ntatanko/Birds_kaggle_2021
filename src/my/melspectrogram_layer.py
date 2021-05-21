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
import tensorflow as tf


class MelSpectrogram(tf.keras.layers.Layer):
    """
    Compute log-magnitude mel-scaled spectrograms.
    Based on:
    https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
    Should be followed by PowerToDb layer.
    Example:
    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow import keras
    from lib.float2d_to_rgb_layer import Float2DToRGB
    from lib.melspectrogram_layer import MelSpectrogram
    from lib.power_to_db_layer import PowerToDB
    N_FFT = 2048
    N_MELS = 256
    N_TIMESTEPS = 256
    POWER = 2
    AUDIO_SR = 32000
    WAVE_LEN_SAMPLES = AUDIO_SR * 5
    WAVE_DTYPE = np.float16
    wave = np.random.randn((WAVE_LEN_SAMPLES)).astype(WAVE_DTYPE)
    waves = np.repeat(wave[np.newaxis, ...], 16, axis=0)
    i_wave = x = keras.layers.Input(shape=WAVE_LEN_SAMPLES, dtype=WAVE_DTYPE)
    x = MelSpectrogram(
        sample_rate=AUDIO_SR,
        fft_size=N_FFT,
        n_mels=N_MELS,
        hop_length=WAVE_LEN_SAMPLES // (N_TIMESTEPS - 1),
        power=POWER,
    )(x)
    o_float = x = PowerToDB()(x)
    o_rgb = x = Float2DToRGB()(x)
    m = keras.models.Model(inputs=[i_wave], outputs=[o_float, o_rgb])
    msgs_f, msgs_rgb = m.predict(waves)
    plt.imshow(msgs_f[0])
    plt.figure()
    plt.imshow(msgs_rgb[0])
    ```
    """

    def __init__(
        self,
        sample_rate,
        fft_size,
        hop_length,
        n_mels,
        f_min=0.0,
        f_max=None,
        pad_end=True,
        power=2,
        **kwargs,
    ):
        super(MelSpectrogram, self).__init__(**kwargs)

        self._f_min = f_min
        self._power = power
        self._n_mels = n_mels
        self._pad_end = pad_end
        self._fft_size = fft_size
        self._hop_length = hop_length
        self._sample_rate = sample_rate
        self._f_max = f_max if f_max is not None else sample_rate // 2

        self._mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self._n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self._sample_rate,
            lower_edge_hertz=self._f_min,
            upper_edge_hertz=self._f_max,
        )

    def build(self, input_shape):
        self.non_trainable_weights.append(self._mel_filterbank)
        super(MelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples).
        A batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq).
        The corresponding batch of log-mel-spectrograms.
        """

        spectrograms = tf.signal.stft(
            tf.cast(waveforms, tf.float32),
            frame_length=self._fft_size,
            frame_step=self._hop_length,
            pad_end=self._pad_end,
        )

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(
            tf.pow(magnitude_spectrograms, self._power),
            self._mel_filterbank,
        )

        return mel_spectrograms

    def get_config(self):

        config = {
            "f_min": self._f_min,
            "f_max": self._f_max,
            "power": self._power,
            "n_mels": self._n_mels,
            "pad_end": self._pad_end,
            "fft_size": self._fft_size,
            "hop_size": self._hop_length,
            "sample_rate": self._sample_rate,
        }

        config.update(super(MelSpectrogram, self).get_config())

        return config
