import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error



# Load recipes data

data_folder = './data'
recipes_file = 'cleaned_data.parquet'

cars_data = pd.read_parquet(os.path.join(data_folder, recipes_file))

cars_prices = cars_data['price']
cars_data.drop(columns='price', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(cars_data, cars_prices, test_size=0.15, random_state=1)

scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed  = scaler.transform(X_test)


lin_mod = LinearRegression().fit(X_train_transformed, y_train)

y_pred = lin_mod.predict(X_test_transformed)

#for negative predictions, set to 1 USD
y_pred[y_pred <= 0] = 1


rmse = np.sqrt(mean_squared_error(y_pred, y_test))
msle = mean_squared_log_error(y_pred, y_test)
print(f'rmse: {rmse}, msle: {msle}')
np.array(np.round(y_pred[:10]))
np.array(y_test.iloc[:10])

c = [i for i in range(len(y_pred))]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=0.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
plt.show()

errors = y_test - y_pred

(mu, sigma) = norm.fit(errors[np.abs(errors) <= 10000])

n, bins, patches = plt.hist(errors, range = (-10000, 10000), bins = 100, density = True, color='green')
y = norm.pdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)
plt.show()