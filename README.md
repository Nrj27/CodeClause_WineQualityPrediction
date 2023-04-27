# CodeClause_WineQualityPrediction
# CodeClause_WineQualityPrediction

This project is focused on predicting the quality of wine using machine learning algorithms.

## Dataset

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/rajyellow46/wine-quality). It contains information about various physicochemical properties of wine and its corresponding quality rating given by experts. The dataset has 11 input features and 1 output feature (quality rating) with 4898 instances.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- xgboost

## Implementation

The following machine learning algorithms are implemented in this project:

- Support Vector Machine (SVM)
- XGBoost Classifier
- Logistic Regression

The dataset is preprocessed using the `MinMaxScaler` from the `sklearn.preprocessing` module. This ensures that all the input features are in the same range. The data is then split into training and testing sets using the `train_test_split` function from the `sklearn.model_selection` module.

The models are trained using the training set and the performance is evaluated on the testing set using various metrics such as accuracy, precision, recall, and F1 score. The `metrics` module from the `sklearn` library is used for evaluating the model performance.

Finally, the results are visualized using `matplotlib` and `seaborn` libraries.

## Usage

1. Clone the repository
2. Install the dependencies using `pip install -r requirements.txt`
3. Run the `wine_quality_prediction.ipynb` notebook to see the implementation and results.

## Results

The best performing model was XGBoost Classifier with an accuracy of 0.73 and F1-score of 0.74. SVM and Logistic Regression models had an accuracy of 0.67 and 0.65 respectively.

The performance of the models can be further improved by tuning the hyperparameters and by using more advanced feature selection techniques.

## Contributors

- Neeraj Rikhari

