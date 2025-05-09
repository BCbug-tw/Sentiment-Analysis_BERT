# Sentiment Analysis (BERT)
This project uses the BERT model to perform sentiment classification on the sentiment140 dataset. The task is a binary classification (positive and negative), serving as a practical example of sentiment analysis in the NLP field.
---

## Dataset
The dataset is based on the open-source[Sentiment140 dataset with 1.6 million tweets](<https://www.kaggle.com/datasets/kazanova/sentiment140> "Title")from Kaggle. It contains 1,600,000 tweets, including sentiment labels, user IDs, usernames, timestamps, and tweet texts. The sentiment labels are divided into two classes: positive (4) and negative (0), with 800,000 tweets in each class. This project only uses the sentiment labels and tweet texts for model training.

---

## Technologies Used
- Model: ```bert-base-chinese```
- Frameworks: HuggingFace Transformers、Datasets
- Evaluation: Precision、Recall、F1-Score、混淆矩陣

---

## How to Run
1. Install dependencies (preferably in a virtual environment):
```python
pip install -r requirement.txt
```

2. Run ```Sentiment_analysis_BERT_FT.ipynb```
You can open and execute the notebook using Jupyter Notebook or VS Code. The key steps include:

- Loading and preprocessing the data
- Tokenization
- Training the model using the HuggingFace ```Trainer``` API
- Model evaluation and visualization of the confusion matrix
- Simple prediction (You can try it on your own sentence!)

---

## Evaluation
### Precision, Recall & F1-score
<div align="center">

 Categories|Precision| Recall | F1-score
 :-------|:------: | :------: | :------: 
 Negitive|0.87   |   0.88  | 0.87
 Pocitive|0.88   |   0.87  | 0.87

</div>

### Confusion Matrix
![image1](<results/confusion matrix.jpg> "confusion matrix")

---

## Future Work
- Deploy the model as an API or web application (Flask / FastAPI / Streamlit)
- Add Early Stopping or a Learning Rate Scheduler
- Experiment with alternative pretrained language models

--- 

## Reference
- Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.
- https://huggingface.co/docs/transformers/en/training



