import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# Load recipes data

data_folder = './data'
recipes_file = 'cleaned_data.parquet'

cars_data = pd.read_parquet(os.path.join(data_folder, recipes_file))
cars_data_cat = pd.read_parquet(os.path.join(data_folder, 'cleaned_data_categorical.parquet'))


cars_prices = cars_data['price']
cars_data.drop(columns='price', inplace=True)

indicator_column_names = cars_data.columns[3:]

X_train, X_test, y_train, y_test = train_test_split(cars_data, cars_prices, 
                                                    test_size=0.10, random_state=37)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=0.10/0.90, random_state=42)

price_scaler = StandardScaler().fit(np.array(y_train).reshape(-1,1))
y_train_transformed = price_scaler.transform(np.array(y_train).reshape(-1,1))
y_test_transformed = price_scaler.transform(np.array(y_test).reshape(-1,1))
y_val_transformed = price_scaler.transform(np.array(y_val).reshape(-1,1))

scaler = StandardScaler().fit(X_train)
X_train_transformed = pd.DataFrame(scaler.transform(X_train), columns= X_train.columns)
X_train_transformed.index = X_train.index
X_train_transformed[indicator_column_names] = X_train[indicator_column_names]
X_test_transformed  = pd.DataFrame(scaler.transform(X_test), columns= X_test.columns)
X_test_transformed.index = X_test.index
X_test_transformed[indicator_column_names] = X_test[indicator_column_names]
X_val_transformed  = pd.DataFrame(scaler.transform(X_val), columns= X_val.columns)
X_val_transformed.index = X_val.index
X_val_transformed[indicator_column_names] = X_val[indicator_column_names]

lin_mod = LinearRegression().fit(X_train_transformed, y_train_transformed)

y_pred_lm = lin_mod.predict(X_test_transformed)

print(lin_mod.coef_)

rmse_lm = np.sqrt(mean_squared_error(y_pred_lm, y_test_transformed))
print(f'rmse: {rmse_lm}')
np.array(np.round(y_pred_lm[:10]))
np.array(y_test_transformed.iloc[:10])

c = [i for i in range(len(y_pred_lm))]
fig = plt.figure()
plt.plot(c,y_test_transformed-y_pred_lm, color="blue", linewidth=0.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
plt.show()

errors = y_test_transformed - y_pred_lm

(mu, sigma) = norm.fit(errors)

n, bins, patches = plt.hist(errors, bins = 100, density = True, color='green')
y = norm.pdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)
plt.show()

print(f'rmse: {rmse_lm}')


#what about SVM/SVR?

from sklearn.svm import SVR

#C a bit decreased, since we do have some outliers/noisy data
svr = SVR(C = 0.8, epsilon=0.1, kernel= 'linear').fit(X_train_transformed[:10000], 
                                                      np.ravel(y_train_transformed[:10000]))
print(svr.coef_.round(2))
# print('Support vectors: \n' + ''.join(str(svr.support_vectors_)))
y_pred_svr = svr.predict(X_test_transformed)
rmse_svr = np.sqrt(mean_squared_error(y_pred_svr, y_test_transformed))

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
# het gemiddelde van neighbors is de voorspelling

knr = KNeighborsRegressor(n_neighbors=11).fit(X_train_transformed, y_train_transformed)
y_pred_knr = knr.predict(X_test_transformed)
rmse_knr = np.sqrt(mean_squared_error(y_pred_knr, y_test_transformed))

#rnr ruineert memory, dus nee

from sklearn.tree import DecisionTreeRegressor
# maak een boom, en fit daarmee dan een step-function op de data.

dtr = DecisionTreeRegressor(random_state=0).fit(X_train_transformed, y_train_transformed)
y_pred_dtr = dtr.predict(X_test_transformed)
rmse_dtr = np.sqrt(mean_squared_error(y_pred_dtr, y_test_transformed))

#requires y_test and y_pred as known variables
import seaborn as sns
import matplotlib.pyplot as plt

sns.jointplot(x = y_test_transformed.flatten(), y = y_pred_dtr.flatten(), kind = 'hist')
# plt.plot(y_test_transformed.flatten(), y_test_transformed.flatten(), 'r--')
plt.xlabel('Real prices')
plt.ylabel('Predicted prices')
plt.show()

# Lasso en Ridge regression
from sklearn.linear_model import Lasso, Ridge
lasso = Lasso(alpha = 0.4).fit(X_train_transformed, y_train_transformed)
y_pred_las = lasso.predict(X_test_transformed)
rmse_las = np.sqrt(mean_squared_error(y_pred_las, y_test_transformed))

ridge = Ridge(alpha = 1.0).fit(X_train_transformed, y_train_transformed)
y_pred_rid = ridge.predict(X_test_transformed)
rmse_rid = np.sqrt(mean_squared_error(y_pred_rid, y_test_transformed))


print(f'Linear model RMSE: {rmse_lm}\n' + 
      f'Support vector model RMSE: {rmse_svr}\n' + 
      f'K-Neighbors model RMSE: {rmse_knr}\n' +  
      f'Desision tree model RMSE: {rmse_dtr}\n' + 
      f'Lasso model RMSE: {rmse_las}\n' + 
      f'Ridge model RMSE: {rmse_rid}\n')

ridge.coef_.round(2)
X_train.columns

output_string = [(f'{(ridge.coef_[0][i]*price_scaler.scale_)[0]: .0f}'+ ' ' + X_train.columns[i]) for i in range(len(X_train.columns))]
print('\n'.join(output_string))
price_scaler.scale_
price_scaler.mean_
price_scaler.inverse_transform(np.array([0,1]).reshape(-1,1))

scaler.mean_[2]
scaler.scale_[2]
ridge.coef_[0]
foo = scaler.inverse_transform(np.array([int(i == 0) for i in range(210)]).reshape(2, 105))
foo[0] - foo[1]
foo[1] - scaler.mean_

def interpret_numeric_coefs(coefs, names, price_scalar, feature_scalar):
      numeric_scales   = (lambda numeric_coef : (np.array(numeric_coef)/feature_scalar.scale_[:3])*
                          price_scalar.scale_) # length-3 input
      
      return(['Numeric', names[:3], numeric_scales(coefs[:3])])

def interpret_coefs(coefs, names, price_scalar, feature_scalar, substring):
      #assumes coefs and prices are scaled.
      indicator_transform = (lambda indicator : price_scalar.scale_ * np.array(indicator))
      
      indicator_cols = [names[i].__contains__(substring) for i in range(len(names))]

      interpret_indicators = [substring.capitalize(), 
                              np.char.replace(names.to_numpy(dtype = 'str')[indicator_cols], 
                                              old = substring + '_', new = ''), 
                              indicator_transform(coefs[indicator_cols])]
      
      return (interpret_indicators)

def print_interpretation(interpretation):
      print(interpretation[0] + ':')
      for col in range(len(interpretation[1])):
            print(f'Value: {interpretation[2][col]: .2f} for {interpretation[1][col].capitalize()}')
      return(0)

def plot_interpretation(interpretation):
      paint_color_dict = {cars_data_cat['paint_color'].value_counts().index[i]: cars_data_cat['paint_color'].value_counts().index[i] for i in range(len(cars_data_cat['paint_color'].value_counts().index))}
      paint_color_dict['custom'] = 'white'
      paint_color_dict['None'] = 'white'
      
      if interpretation[0] != 'Paint_color':
            paint_color_dict = ['r' if interpr < 0 else 'g' for interpr in interpretation[2]]
      df = pd.DataFrame()
      df['value'] = interpretation[2]
      df['category'] = interpretation[1]
      sns.set_context(rc = {'patch.linewidth': 0.6})
      ax = sns.barplot(df, x = 'category', y = 'value', 
                       palette = paint_color_dict,
                       edgecolor = 'black')
      ax.set(ylabel = 'Value (USD)', xlabel = interpretation[0])
      plt.xticks(rotation=90)
      plt.show()

print_interpretation(interpret_coefs(ridge.coef_[0], X_train.columns, price_scaler, scaler, 'manufacturer'))
plot_interpretation(interpret_coefs(ridge.coef_[0], X_train.columns, price_scaler, scaler, 'manufacturer'))

for category in ['manufacturer', 'condition', 'fuel', 
                                   'title_status', 'transmission', 'drive', 
                                   'size', 'type', 'paint_color']:
      print_interpretation(interpret_coefs(ridge.coef_[0], X_train.columns, price_scaler, scaler, category))
      plot_interpretation(interpret_coefs(ridge.coef_[0], X_train.columns, price_scaler, scaler, category))
