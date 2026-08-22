"""Convolutional network for spectrogram-based drone presence detection.

Implements a VGG-style binary classifier over decibel mel spectrograms produced by
:class:`aerosonar.features.transforms.SpectrogramTransform`.
"""

import torch.nn as nn


class SpectrogramCNN(nn.Module):
    """Binary drone-presence classifier over mel spectrograms.

    Four convolutional blocks (Conv2d, BatchNorm2d, ReLU) widen the channel dimension
    from 1 to 256, with max pooling after the first three. Global adaptive max pooling
    then reduces each channel to a single value, making the network independent of the
    input's frequency and time extent, and a two-layer head produces the class scores.

    Input shape: ``(batch, 1, freq_bins, time_steps)``.
    Output shape: ``(batch, 2)`` of raw logits, ordered ``[no drone, drone]``.

    The network emits logits, not probabilities. Callers apply softmax themselves and
    take index 1 as the drone probability.
    """

    def __init__(self):
        """Build the feature extractor and classification head."""
        super(SpectrogramCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (64 freq, T/2)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (32 freq, T/4)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # (16 freq, T/8)

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """Classify a batch of spectrograms.

        Args:
            x: Spectrogram batch of shape ``(batch, 1, freq_bins, time_steps)``.

        Returns:
            torch.Tensor: Logits of shape ``(batch, 2)``, ordered
            ``[no drone, drone]``.
        """
        x = self.features(x)
        return self.classifier(x)
