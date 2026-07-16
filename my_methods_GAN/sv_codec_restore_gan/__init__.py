"""SV-CodecRestoreGAN package.

A speaker-verification-oriented codec restoration pipeline:
- 48 kHz internal TF processing
- CWS-style split/merge with GridNet-style blocks
- optional GAN fine-tuning
- WavLM + CAM++ consistency losses
"""

__all__ = [
    "data",
    "models",
    "train",
    "utils",
]
