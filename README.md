# Binary Audio Classifier adapted from WaveNet

## Goal
Build a classifier to distinguish sounds, such as laughing and crying.

## Motivation
Audio data analysis has typically relied on preprocessed features such as MFCCs (Mel-frequency cepstral coefficients) for meaningful information retrieval. However, with the advent of the WaveNet generative model for raw audio, I speculated that it may be possible to construct an end-to-end audio classifier that uses only raw audio and dilated causal convolution structures to directly extract enough information for meaningful classification. This project is an attempted implementation of such a classifier.

## Status
Work in progress.
* Implemented:
  * Data preprocessing pipeline (wav -> TFRecord)
  * Basic model structure (dilated causal convolution)
    * Designed with cross-entropy loss on sigmoid activation, for binary classification
  * Training and evaluation script
    * Testing with TensorFlow single-word speech data (`onoff_ver` branch)
  * Packaging for deployment on Google Cloud Platform (`onoff_ver` branch)

## TODO
* Process viable datasets for classification
  * Laughing-crying
    * Manual edits for consistent sample data
    * Data augmentation (possibly)
* Adjust structure for better fitting and prediction
  * Regularization/dropout
  * Hyperparameter tuning (for optimizers)
* Modify for multi-class classification
  * Network structure
  * Loss function

---

by John Choi (최정혁), 2018  
* [PDF of original WaveNet paper by Google DeepMind](https://arxiv.org/pdf/1609.03499.pdf)
* based on [this TensorFlow implementation of WaveNet](https://github.com/ibab/tensorflow-wavenet)
* sound sources obtained from [Freesound](https://www.freesound.org)
* additional speech data obtained from:
  * Warden P. Speech Commands: A public dataset for single-word speech recognition, 2017. Available from [TensorFlow (direct download)](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
