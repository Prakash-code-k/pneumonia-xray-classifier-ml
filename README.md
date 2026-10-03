# Pneumonia Detection from Chest X-Rays

A web app that looks at a chest X-ray and tells whether it is **Normal**, **Bacterial pneumonia** or **Viral pneumonia**.

It is built with classical machine learning, not deep learning. Every X-ray is turned into 1,845 hand-crafted features (shape, texture and brightness), and a trained classifier picks the most likely class. It runs on an ordinary CPU.

**Live demo:** [add your Render link here]

> This is a student project for learning purposes. It is not a medical device and its result is not a diagnosis. A real X-ray must always be read by a doctor.

## How it works

1. The X-ray is converted to grayscale, resized to 128 x 128 and enhanced with CLAHE.
2. Four groups of features are extracted:
   - HOG (1,764): edge directions that describe the shape of the lungs, ribs and heart
   - LBP (28): local texture patterns
   - GLCM (12): texture measures such as contrast, homogeneity and energy
   - Intensity (41): brightness statistics and a histogram
3. Five models were trained and compared: Logistic Regression, KNN, SVM, Random Forest and XGBoost.
4. The best model was picked using 5-fold cross-validation with macro F1 score, and saved for the website.

## Dataset

[Chest X-rays: bacterial / viral pneumonia / normal](https://www.kaggle.com/datasets/kostasdiamantaras/chest-xrays-bacterial-viral-pneumonia-normal) by Kostas Diamantaras on Kaggle. It has 4,672 labeled pediatric X-rays: 1,227 normal, 2,238 bacterial and 1,207 viral. The dataset is not included in this repository.

## Results

| Model | Accuracy | Macro F1 | ROC AUC | Pneumonia detection recall |
|---|---|---|---|---|
| Logistic Regression | | | | |
| KNN | | | | |
| SVM | | | | |
| Random Forest | | | | |
| XGBoost | | | | |

## Project structure

```
app.py                              Flask web app
templates/index.html                Upload page
pneumonia_classical_ml.joblib       Trained model
notebook/                           Training notebook
docs/                               Project report
requirements.txt                    Python packages
render.yaml                         Deployment settings for Render
```

## Run it on your computer

```
git clone https://github.com/YOUR-USERNAME/pneumonia-xray-classifier.git
cd pneumonia-xray-classifier
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 and upload an X-ray.

## Train the model again

Open `notebook/Pneumonia_Detection_Classical_ML.ipynb`, set `DATA_PATH` to the dataset folder and run all cells. It saves a new `pneumonia_classical_ml.joblib`.

## Author

[Your Name], B.Tech Computer Science and Engineering, [College Name]
