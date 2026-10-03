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
4. Each model was tuned with 5-fold cross-validation using macro F1 score. SVM came out on top and is the model behind the website.

## Dataset

[Chest X-rays: bacterial / viral pneumonia / normal](https://www.kaggle.com/datasets/kostasdiamantaras/chest-xrays-bacterial-viral-pneumonia-normal) by Kostas Diamantaras on Kaggle. It has 4,672 labeled pediatric X-rays: 1,227 normal, 2,238 bacterial and 1,207 viral. The dataset is not included in this repository.

## Results

All models were tested on a held-out 20% of the labeled images that were never used during training.

| Model | Accuracy | Macro F1 | ROC AUC | Pneumonia detection recall | Pneumonia detection specificity |
|---|---|---|---|---|---|
| Logistic Regression | 77.6% | 0.772 | 0.907 | 95.9% | 94.6% |
| KNN | 80.8% | 0.794 | 0.912 | 93.3% | **95.0%** |
| **SVM** | **82.8%** | **0.819** | **0.931** | **96.4%** | 94.6% |
| Random Forest | 78.7% | 0.765 | 0.913 | 94.8% | 92.6% |
| XGBoost | 80.9% | 0.794 | 0.928 | 96.1% | 93.0% |

Pneumonia detection recall and specificity treat bacterial and viral as one "pneumonia" group, which answers the simpler question: is there pneumonia at all?

What the numbers show:

- **SVM did best overall**, with 82.8% accuracy on the three classes and the highest macro F1 and ROC AUC. It is the model used in the web app.
- **Spotting pneumonia is the easy part.** SVM caught 96.4% of pneumonia cases and correctly cleared 94.6% of normal X-rays.
- **Telling bacterial from viral is the hard part.** That is why three-class accuracy sits around 83% while detection is above 96%. Both types show up as white patches on an X-ray, and in hospitals they are usually confirmed with lab tests.

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

Prakash Kumar <br>
B.Tech Computer Science and Engineering
