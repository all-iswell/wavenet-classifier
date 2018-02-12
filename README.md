# Binary Audio Classifier adapted from WaveNet

---

## Goal:
Build a classifier to distinguish sounds of laughing from those of crying.

---

## Status:
Work in progress.
* Basic model structure and working training process implemented.
  * Uses temporary dummy data (non-humanlabeled)

---

## TODO:
* Streamline preprocessing pipeline (mp3 -> TFRecord)
* Process viable dataset for classification (manual editing)
  * Data augmentation (possibly)
* Refactor code to match Google Cloud Platform ML Engine specifications

---

by John Choi (최정혁), 2018  
based on [this TensorFlow implementation of WaveNet](https://github.com/ibab/tensorflow-wavenet)   
sound sources obtained from [Freesound](https://www.freesound.org)
