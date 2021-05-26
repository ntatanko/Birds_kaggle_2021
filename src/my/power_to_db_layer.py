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
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class PowerToDb(keras.layers.Layer):
    """
    Does roughly the same as librosa.power_to_db() but faster.
    Takes 2D float32 input, not RGB uint8 image.
    Should be followed by Flot2dToRGB layer for input into EfficientNet.
    Use:
    ```python
    i = keras.layers.Input(shape=msg.shape, dtype='float32')
    x = PowerToDb()(i)
    m = keras.models.Model(inputs=[i], outputs=[x])
    ```
    """

    def __init__(self, ref=1.0, amin=1e-10, top_db=80, **kwargs):
        super(PowerToDb, self).__init__(**kwargs)

        if amin <= 0:
            raise Exception("amin must be strictly positive")

        if top_db is not None:
            if top_db < 0:
                raise Exception("top_db must be non-negative")

        self._ref = ref
        self._amin = amin
        self._top_db = top_db

    def call(self, inputs):
        # see https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py#L1447

        LOG10 = 2.302585092994046

        log_spec = K.cast(inputs, "float32")
        log_spec = K.clip(log_spec, min_value=self._amin, max_value=None)
        log_spec = 10.0 * K.log(log_spec) / LOG10
        ref = K.max([self._amin, K.abs(self._ref)])
        log_spec -= 10.0 * K.log(ref) / LOG10

        if self._top_db is not None:

            log_spec = K.clip(
                log_spec,
                min_value=K.max(log_spec, axis=(1, 2), keepdims=True) - self._top_db,
                max_value=None,
            )

        return log_spec

    def get_config(self):
        return {
            "ref": self._ref,
            "amin": self._amin,
            "top_db": self._top_db,
        }