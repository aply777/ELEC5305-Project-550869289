# SpeakerID-Project
Speaker Identification and Characterisation using Audio ML
## Project title (from Canvas ELEC5305)
Speaker identification and characterization:
Speech recognition has focussed on the lexical content of speech (i.e. the words) and worked quite hard to exclude other aspects of the signal. Yet when we listen to speech, we infer considerable information about the speaker, such as gender, age, country of origin etc. All this information should be present in the signal, it is simply a matter of finding the right features and training the right recognizer.
## Objectives
Extract relevant audio features for speaker characterisation<br>
Implement a baseline machine learning classifier (SVM/KNN)<br>
Evaluate classifier performance on a small speech dataset<br>
Optionally explore more advanced models (e.g., CNN) if time permits
## Methods
**Pre-processing**<br>
Normalise audio signals using functions from MATLAB<br>
Remove leading/trailing silence using energy thresholding<br>
**Feature Extraction**<br>
Extract MFCC features using mfcc()<br>
Estimate pitch using the pitch function<br>
Generate spectrograms<br>
**Classification**<br>
Train a Support Vector Machine classifier<br>
Alternatively, use K-Nearest Neighbours <br>
**Evaluation**<br>
Compute accuracy and confusion matrix
## Data
Public speech datasets such as VoxCeleb, TIMIT
## Reference
Speaker Identification Using Custom SincNet Layer and Deep Learning:<br>
https://au.mathworks.com/help/deeplearning/ug/speaker-identification-using-custom-sincnet-layer-and-deep-learning.html<br>
VoxCeleb:<br>
https://github.com/a-nagrani/VGGVox<br>
TIMIT<br>
https://github.com/datasets-mila/datasets--timit/tree/master
