# Comment-Toxicity-Model-with-Deep-Learning

This repository implements a Comment Toxicity Detection Model to classify online comments into multiple categories of toxicity. The project utilizes TensorFlow, LSTM, and TextVectorization to process and classify textual data effectively.

About the Project
The objective of this project is to detect and categorize toxic comments into six categories:
 - toxic
 - severe toxic
 - obscene
 - threat
 - insult
 - identity hate
The model is trained on the Jigsaw Toxic Comment Classification dataset and predicts the likelihood of each comment belonging to one or more categories.



Dataset
The dataset is sourced from the Jigsaw Toxic Comment Classification Challenge on Kaggle.

Preprocessing
 - Text Cleaning: Removed punctuation, special characters, and redundant spaces.
 - Vectorization: Used TensorFlow's TextVectorization layer to tokenize and convert text to     sequences of integers.
 - Max Vocabulary Size: Limited to 200,000 unique tokens.
 - Sequence Length: Fixed at 1,800 tokens per comment for uniformity.

Model Workflow
 - Text Data Loading: Text comments are extracted from the dataset and labels are encoded as a one-hot array for multi-class classification.
 - Vectorization: TextVectorization is applied to preprocess text into a numerical format suitable for neural networks.
 - Model Training: A Sequential model with LSTM and dense layers trains on the vectorized data.
 - Evaluation: The model's performance is evaluated using metrics like precision, recall, and accuracy.
 - Deployment: Integrated with Gradio for interactive toxicity detection.

Model Architecture
 - Embedding Layer: Converts tokens to dense vectors of fixed size.
 - Bidirectional LSTM: Processes text in both forward and backward directions.
 - Fully Connected Layers: Three dense layers with ReLU activation for feature extraction.
 - Output Layer: Sigmoid activation for multi-label classification.

Future Scope
 - Extend the model to detect toxicity in multiple languages.
 - Implement attention mechanisms for better context understanding.
 - Improve the interface by deploying it on cloud platforms like AWS or Azure.
