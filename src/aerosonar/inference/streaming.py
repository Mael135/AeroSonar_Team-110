"""Rolling-window sample buffer for the live detection loop.

Kept independent of ``sounddevice`` so the buffering logic can be exercised on hosts
without an audio input device or that optional dependency installed.
"""
import numpy as np


def push_frame(buffer: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Append a block of samples to the tail of a rolling window, in place.

    The buffer holds the most recent ``len(buffer)`` samples in chronological order.
    Existing contents shift left by ``len(block)`` samples to make room.

    Args:
        buffer: The rolling window, modified in place.
        block: New samples. Flattened before use. A block at least as long as the
            window replaces its entire contents with the block's final samples.

    Returns:
        np.ndarray: The same ``buffer`` object, to allow chaining.
    """
    data = np.asarray(block).reshape(-1)
    n = data.shape[0]
    window = buffer.shape[0]
    if n == 0:
        return buffer
    if n >= window:
        buffer[:] = data[-window:]
        return buffer
    buffer[:-n] = buffer[n:]
    buffer[-n:] = data
    return buffer
