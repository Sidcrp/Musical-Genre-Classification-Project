# Musical-Genre-Classification-Project
This project explores the classification of music genres using Deep Neural Networks (DNNs), specifically focusing on Convolutional Neural Networks (CNNs). By leveraging the GTZAN dataset of audio files across 10 genres, we aim to compare the performance of a baseline linear model, a simple CNN, and a complex CNN to understand the effectiveness of CNNs in music genre classification.

# Introduction
Music genre classification is a challenging task due to the overlapping boundaries and similar features across genres. This project uses spectrograms derived from audio files as input to a series of machine learning models, comparing their accuracy and generalization capabilities. The ultimate goal is to demonstrate how CNNs perform better in capturing features relevant to audio classification.

# Dataset
We use the GTZAN dataset sourced from HuggingFace. The dataset contains:

1,000 audio files (100 files per genre)
Genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
Format: 30-second .wav files
Due to a processing error, one jazz file was excluded, resulting in:

799 files for training (79 from jazz, 80 from others)
200 files for testing (20 from each genre)

# Methods
Feature Extraction and Spectrogram Creation
Using the librosa library, spectrograms were created with:

FFT window length: 2048
Hop length: 512
Mel bands: 128
Spectrograms were converted to a dB scale and resized to 64x100 for computational efficiency.

Linear Model
A baseline model with:

64,010 parameters
Optimized with Adam and CrossEntropyLoss
Batch size: 128 (training), 64 (testing)
Learning rate: 0.0001
Simple CNN
A CNN with:

15,610 parameters
3 convolutional layers with batch normalization, ReLU activation, max pooling, and dropout
2 fully connected layers
Complex CNN
A more parameter-heavy CNN with:

59,114 parameters
Doubling the channels and nodes of the simple CNN

# Results
Linear Model: 50.50% test accuracy (significant overfitting)
Simple CNN: 54.50% test accuracy (minimal overfitting)
Complex CNN: 62.50% test accuracy (overfitting observed)
Key observations:

Complex CNN showed the highest accuracy but was prone to overfitting.
Simple CNN provided a balance between performance and generalization.
Pop genre proved challenging due to overlapping features with other genres.

# Limitations and Future Work
Dataset Size: Limited to 1,000 files, leading to potential overfitting.
Cross-validation: Results are based on a single run; future iterations should include cross-validation.
Architectural Exploration: Investigate pre-trained models (e.g., ResNet) and optimize SVM architectures.
Hyperparameter Tuning: Further tuning could improve model performance. 

# Dependencies
Python 3.9+
Libraries: 
librosa
Pytorch
Scikit-learn
NumPy
Matplotlib
