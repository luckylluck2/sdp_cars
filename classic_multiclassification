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


covariate_dataFrame = recipes_data[['CookTime', 'PrepTime','CaloriesPDV', 
        'FatPDV','SaturatedFatPDV', 'CholesterolPDV', 'SodiumPDV', 
        'CarbohydratePDV', 'FiberPDV', 'SugarPDV', 'ProteinPDV']]

X_train, X_test, y_train, y_test = train_test_split(covariate_dataFrame, recipes_data['AggregatedRating'], 
                                                    test_size=0.2, random_state=1)

#convert y_* to string type, to allow for classification (not possible on float type)
y_train, y_test = np.array(y_train).astype(dtype=str), np.array(y_test).astype(dtype=str)


covariate_scaler = StandardScaler().fit(X_train)
scaled_X_train = covariate_scaler.transform(X_train)
scaled_X_test = covariate_scaler.transform(X_test)

multi_classifier = LogisticRegression(multi_class = 'ovr').fit(X = scaled_X_train, y = y_train)
star_prediction = multi_classifier.predict(X = scaled_X_test)

print(confusion_matrix(y_true = y_test, y_pred = star_prediction))

conf_mat_disp = ConfusionMatrixDisplay.from_estimator(estimator = multi_classifier, 
                                                      X = scaled_X_test, y = y_test, normalize = 'all')

plt.pyplot.show()

