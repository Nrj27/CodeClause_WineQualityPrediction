# CodeClause_WineQualityPrediction
Wine Quality Prediction
This project is focused on predicting the quality of wine using machine learning algorithms.

## Dataset
https://www.kaggle.com/datasets/rajyellow46/wine-quality

## Dependencies
pandas
numpy
matplotlib
seaborn
sklearn
xgboost


##Implementation
The following machine learning algorithms are implemented in this project:

Support Vector Machine (SVM)
XGBoost Classifier
Logistic Regression
The dataset is preprocessed using the MinMaxScaler from the sklearn.preprocessing module. This ensures that all the input features are in the same range. The data is then split into training and testing sets using train_test_split function from the sklearn.model_selection module.

The models are trained using the training set and the performance is evaluated on the testing set using various metrics such as accuracy, precision, recall, and F1 score. The metrics module from the sklearn library is used for evaluating the model performance.

Finally, the results are visualized using matplotlib and seaborn libraries.

##Usage
Clone the repository
Install the dependencies using pip install -r requirements.txt
Run the wine_quality_prediction.ipynb notebook to see the implementation and results.

##Results
The best performing model was XGBoost Classifier with an accuracy of 0.97 and F1-score of 0.80. SVM and Logistic Regression models had an accuracy of 0.70 and 0.70 respectively.

The performance of the models can be further improved by tuning the hyperparameters and by using more advanced feature selection techniques.

Contributors
Neeraj Rikhari
