# Speaker Recognition and Audio Classification Using SVM, CNN (AlexNet), and ECAPA-TDNN
## Project title
Speaker identification:
This project explores a complete pipeline of speaker‐related audio classification tasks using the TIMIT corpus, including Speaker Identification (SI), Gender Classification, and Dialect Region Recognition.
Three modelling paradigms were implemented and compared.

## Objectives
Extract relevant audio features for speaker characterisation<br>
Implement a baseline machine learning classifier (SVM)<br>
Evaluate classifier performance on a small speech dataset<br>
Optionally explore more advanced models (e.g., CNN) if time permits

## Method 1 — Traditional Machine Learning (MATLAB SVM)
The project begins with classical feature-based speaker recognition.<br>
	MFCC and Log-Mel features were extracted in MATLAB.
  SVM was used for gender, dialect, and speaker classification.
  Results showed that traditional MFCC + SVM retains strong phonetic cues, giving medium performance on dialect classification, but very limited performance on speaker identification.
Code for SVM is named "SVM.mlx"

## Method 2 - CNN-Based Feature Learning (AlexNet)
Mel-spectrograms were generated from all TIMIT audio and used to train a CNN model:
A multi-task AlexNet was trained for three tasks (SI, Gender, Dialect).
The network shares convolutional layers and uses three separate classification heads.
Demonstrated improved gender classification, but limited performance for dialect and speaker ID, mainly due to spectrogram downsampling and data scarcity.
Code for converting audio into spectrogram is named "Sepc.mlx"
Code for AlexNet classification is named "alexnet.py"

## Method 3 — Modern Embedding-Based Model (ECAPA-TDNN with SpeechBrain)
The final and most advanced approach uses pre-trained ECAPA-TDNN embeddings extracted with the SpeechBrain toolkit:
Trained on VoxCeleb1 & VoxCeleb2, providing strong generalisation.
ECAPA-TDNN achieved high accuracy in speaker identification (98.33%) and gender (99.58%), but dialect classification remained difficult due to the small phonetic differences in TIMIT.
Code for fixing the WAV file is named "convertWav.py"
Code for Method 3 is named "planB.py"

## Data
Public speech datasets TIMIT

## Conclusion
The best performance belongs to ECAPA-TDNN, but regarding the dialect region classification, the performance could be further enhanced.

## Notes
The "alexnet.py" and "planB.py" code need to run with CUDA.

## Reference
TIMIT:<br>
https://github.com/datasets-mila/datasets--timit/tree/master
