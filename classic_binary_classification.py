import numpy as np
import pandas as pd
import os
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load recipes data

data_folder = './data'
data_file = 'cleaned_data.parquet'

recipes_data = pd.read_parquet(os.path.join(data_folder, data_file))

recipes_data.columns
recipes_data.describe()

is_five_stars = recipes_data['AggregatedRating'] == 5
covariate_dataFrame = recipes_data[['CookTime', 'PrepTime','CaloriesPDV', 
        'FatPDV','SaturatedFatPDV', 'CholesterolPDV', 'SodiumPDV', 
        'CarbohydratePDV', 'FiberPDV', 'SugarPDV', 'ProteinPDV']]

X_train, X_test, y_train, y_test = train_test_split(covariate_dataFrame, is_five_stars, 
                                                    test_size=0.2, random_state=1)

covariate_scaler = StandardScaler().fit(X_train)
scaled_X_train = covariate_scaler.transform(X_train)
scaled_X_test = covariate_scaler.transform(X_test)

bin_classifier = LogisticRegression(class_weight = {False: 3.5, True: 2}).fit(X = scaled_X_train, y = y_train)
five_star_prediction = bin_classifier.predict(X = scaled_X_test)

print(confusion_matrix(y_true = y_test, y_pred = five_star_prediction))

conf_mat_disp = ConfusionMatrixDisplay.from_estimator(estimator = bin_classifier, 
                                                      X = scaled_X_test, y = y_test, normalize = 'all')

plt.pyplot.show()

