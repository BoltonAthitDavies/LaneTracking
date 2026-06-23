"""
Author: Sippawit Thammawiset
Date: September 29, 2024.
File: debug.py
"""

from typing import Union, Optional
import matplotlib.pyplot as plt
import numpy as np


def plotimg(image: np.ndarray,
            cmap: Optional[str] = None,
            stop: Union[bool, int] = True) -> None:
    plt.imshow(image, cmap=cmap)
    plt.show()

    if stop:
        assert False


def plotscatter(x: np.ndarray,
                y: np.ndarray,
                stop: Union[bool, int] = True) -> None:
    plt.scatter(x, y)
    plt.show()

    if stop:
        assert False


def stop() -> None:
    raise AssertionError(
        'Stop here'
    )
