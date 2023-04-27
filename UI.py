# Importing libraries
import pandas as pd
import numpy as np 
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from PIL import Image

image = Image.open('wine_image.jpeg')
st.image(image)

#app heading
st.write(""" # Wine Quality Prediction App
This app predicts the ***Wine Quality*** type!
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 , 0.52)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.0)
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'chlorides': chlorides,
              'total_sulfur_dioxide':total_sulfur_dioxide,
              'alcohol':alcohol,
                'sulphates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)
#reading csv file
wine = pd.read_csv("winequalityN.csv")
for col in wine.columns:
  if wine[col].isnull().sum() > 0:
    wine[col] = wine[col].fillna(wine[col].mean())
 
wine.isnull().sum().sum()
X =np.array(wine[['fixed acidity', 'volatile acidity' , 'citric acid' , 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(wine['quality'])
wine['best quality'] = [1 if x > 5 else 0 for x in wine.quality]
wine.replace({'white': 1, 'red': 0}, inplace=True)
features = wine.drop(['quality', 'best quality'], axis=1)
target = wine['best quality']
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)
xtrain.shape, xtest.shape
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# Evaluating the model
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
        models[i].fit(xtrain, ytrain)
        st.text(f'{models[i]} : ')
        st.text(metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
        st.text(metrics.roc_auc_score(
		ytest, models[i].predict(xtest)))
        
st.subheader('Confusion Matrix')       
st.write((metrics.confusion_matrix(ytest,models[i].predict(xtest))))

st.subheader('Prediction Probability')
st.text((metrics.classification_report(ytest,models[1].predict(xtest))))




