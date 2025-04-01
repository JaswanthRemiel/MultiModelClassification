# MultiModelClassification
![Image](https://github.com/user-attachments/assets/960508ca-9df5-4d94-9e6b-80773ea47c4b)

A multimodal classification project combining text and image processing. Includes pre-trained models and a streamlined interface for inference.

## Folder Structure
```MultiModelClassification/
├── NotebooksPY/ # Jupyter/Python notebooks for experimentation
├── app.py # Main application interface
├── image_classification_model.pt # Pretrained ResNet50 image classification model
├── requirements.txt # Python dependencies
├── text_classification_model.h5 # Pretrained BiLSTM+Attention text classification model
├── tokenizer.joblib # Text tokenizer for preprocessing
└── README.md # Project documentation
```
```Pre-trained Models
text_classification_model.h5: BiLSTM + Attention model for text classification
image_classification_model.pt: ResNet50 model for image classification
tokenizer.joblib: Pre-fitted tokenizer for text preprocessing
```
