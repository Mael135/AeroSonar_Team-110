"""AeroSonar: acoustic drone presence detection.

Package layout:

``config``
    YAML configuration loading.
``data``
    Raw-audio preprocessing, the spectrogram dataset, augmentation and data splitting.
``features``
    Waveform-to-spectrogram feature extraction.
``models``
    Model definitions and trained weights.
``training``
    Training loops for the CNN and the SVM baseline.
``inference``
    Live microphone detection and its rolling-window buffer.
``evaluation``
    The verification and evaluation suite and its report generator.
``utils``
    Seeding, figure styling and the dataset audit tool.

Entry points are module-level and expect the repository root as the working directory,
since configuration paths are resolved relative to it.
"""
