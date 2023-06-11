# Classify Male/Female image
[Github link](https://github.com/dangquan1402/udacity-datascientist-nanodegree/tree/main/project4)
## Installations
 - NumPy
 - Pandas
 - Seaborn
 - Matplotlib
 - torch
 - torchvision
 - Pillow
 - Streamlit
 - Flask
 - FlaskRestful 
 

## Project Motivation
For this project I was interested in classifying male/female image. Dataset from [kaggle](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)

The project involved:
  - Loading and preprocessing dataset (download from kaggle)
  - Training using pretrained model from torchvision. The model used is Alexnet.
  - Customizing model for 2 classes (male and female)
  - Flask API for model inference
  - Streamlit application for UI


## File Descriptions
- The detail can be seen in `male_female_classification_readme.md` file.
- It can also run step by step in the notebook as well to see how it works.


## Result
- The model has trained by 20 epochs.
- ![image](images/accuracy.png)
- ![image](images/losses.png)]
- To run the app:
  - 1st step: `python app.py` for starting flask API for model inference
  - 2st step: `streamlit run streamlit_app.py` for starting streamlit application.

When the app runs successfully, it can be accessed by this link
`http://localhost:8501`
- ![image](images/sample_app.jpg)

