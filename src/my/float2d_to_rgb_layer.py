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
import tensorflow.keras.backend as K
from tensorflow import keras


class Float2DToRGB(keras.layers.Layer):
    """
    Converts 2D float input to uint8 RGB
    Use:
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    a = np.random.randn(256, 256)
    a[:, 10:30] = 0
    a[10:30, :] = 1
    i = keras.layers.Input(shape=a.shape)
    x = Float2DToRGB()(i)
    m = keras.models.Model(inputs=[i], outputs=[x])
    r = m.predict(a[np.newaxis, ...])
    plt.imshow(r[0])
    plt.figure()
    plt.imshow(a, cmap='gray')
    ```
    """

    def __init__(self, E=1e-7, **kwargs):
        super(Float2DToRGB, self).__init__(**kwargs)
        # minimum value for division to avoid overflows
        self._e = E

    def call(self, inputs):
        res = inputs - K.min(inputs, axis=(1, 2), keepdims=True)
        res /= K.max(res, axis=(1, 2), keepdims=True) + self._e
        res *= 255.0
        res = K.cast(res, "uint8")
        res = K.expand_dims(res, axis=3)
        res = K.repeat_elements(res, 3, 3)

        return res

    def get_config(self):
        return {
            "E": self._e,
        }

