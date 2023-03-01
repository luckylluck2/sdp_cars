import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load recipes data

data_folder = '../data'
cars_data_train = 'cleaned_data_train.parquet'
cars_data_test = 'cleaned_data_test.parquet'
cars_data_val = 'cleaned_data_val.parquet'
cars_price_train = 'cleaned_price_train.parquet'
cars_price_test = 'cleaned_price_test.parquet'
cars_price_val = 'cleaned_price_val.parquet'

cars_data_cat = pd.read_parquet(os.path.join(data_folder, 'cleaned_data_categorical.parquet'))

y_train = pd.read_parquet(os.path.join(data_folder, cars_price_train))
y_train.index = range(len(y_train))
y_test = pd.read_parquet(os.path.join(data_folder, cars_price_test))
y_test.index = range(len(y_test))
y_val = pd.read_parquet(os.path.join(data_folder, cars_price_val))
y_val.index = range(len(y_val))
X_train = pd.read_parquet(os.path.join(data_folder, cars_data_train))
X_train.index = range(len(y_train))
X_test = pd.read_parquet(os.path.join(data_folder, cars_data_test))
X_test.index = range(len(y_test))
X_val = pd.read_parquet(os.path.join(data_folder, cars_data_val))
X_val.index = range(len(y_val))

indicator_column_names = X_train.columns[5:]

price_scaler = MinMaxScaler().fit(np.array(y_train).reshape(-1,1))
y_train = pd.DataFrame(price_scaler.transform(np.array(y_train).reshape(-1,1)), columns= ['price'])
y_test = pd.DataFrame(price_scaler.transform(np.array(y_test).reshape(-1,1)), columns= ['price'])
y_val = pd.DataFrame(price_scaler.transform(np.array(y_val).reshape(-1,1)), columns= ['price'])

cov_scaler = StandardScaler().fit(X_train)
X_train_t = pd.DataFrame(cov_scaler.transform(X_train), columns= X_train.columns)
X_train_t.index = X_train.index
X_train_t[indicator_column_names] = X_train[indicator_column_names]
X_test_t = pd.DataFrame(cov_scaler.transform(X_test), columns= X_test.columns)
X_test_t.index = X_test.index
X_test_t[indicator_column_names] = X_test[indicator_column_names]
X_val_t = pd.DataFrame(cov_scaler.transform(X_val), columns= X_val.columns)
X_val_t.index = X_val.index
X_val_t[indicator_column_names] = X_val[indicator_column_names]


X_train, X_test, X_val = X_train_t, X_test_t, X_val_t

lin_mod = LinearRegression().fit(X_train, y_train)

y_pred_lm = lin_mod.predict(X_test)

rmse_lm = np.sqrt(mean_squared_error(y_pred_lm, y_test))

np.array(np.round(y_pred_lm[:10]))
np.array(y_test.iloc[:10])

c = [i for i in range(len(y_pred_lm))]
fig = plt.figure()
plt.plot(c,y_test-y_pred_lm, color="blue", linewidth=0.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
plt.show()

errors = y_test - y_pred_lm

(mu, sigma) = norm.fit(errors)

n, bins, patches = plt.hist(errors, bins = 100, density = True, color='green')
y = norm.pdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)
plt.show()


#what about SVM/SVR?

from sklearn.svm import SVR

#C a bit decreased, since we do have some outliers/noisy data
svr = SVR(C = 0.8, epsilon=0.1, kernel= 'linear').fit(X_train[:10000], 
                                                      np.ravel(y_train[:10000]))
print(svr.coef_.round(2))
# print('Support vectors: \n' + ''.join(str(svr.support_vectors_)))
y_pred_svr = svr.predict(X_test)
rmse_svr = np.sqrt(mean_squared_error(y_pred_svr, y_test))

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
# het gemiddelde van neighbors is de voorspelling

knr = KNeighborsRegressor(n_neighbors=11).fit(X_train, y_train)
y_pred_knr = knr.predict(X_test)
rmse_knr = np.sqrt(mean_squared_error(y_pred_knr, y_test))

#rnr ruineert memory, dus nee

from sklearn.tree import DecisionTreeRegressor
# maak een boom, en fit daarmee dan een step-function op de data.

dtr = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)
rmse_dtr = np.sqrt(mean_squared_error(y_pred_dtr, y_test))

import sklearn.inspection as ski

perm_imp = ski.permutation_importance(dtr, X_val, y_val)

plt.scatter(x = range(len(perm_imp['importances_mean'])), y = perm_imp['importances_mean'])
plt.show()

#requires y_test and y_pred as known variables
import seaborn as sns
import matplotlib.pyplot as plt

sns.jointplot(x = y_test.T.to_numpy()[0], y = y_pred_dtr, 
              kind = 'scatter', alpha = 0.2, s = 2)
# plt.plot(y_test.flatten(), y_test.flatten(), 'r--')
plt.xlabel('Real prices')
plt.ylabel('Predicted prices')
plt.show()

# Lasso en Ridge regression
from sklearn.linear_model import Lasso, Ridge
lasso = Lasso(alpha = 0.4).fit(X_train, y_train)
y_pred_las = lasso.predict(X_test)
rmse_las = np.sqrt(mean_squared_error(y_pred_las, y_test))

ridge = Ridge(alpha = 1.0).fit(X_train, y_train)
y_pred_rid = ridge.predict(X_test)
rmse_rid = np.sqrt(mean_squared_error(y_pred_rid, y_test))


print(f'Linear model RMSE: {rmse_lm}\n' + 
      f'Support vector model RMSE: {rmse_svr}\n' + 
      f'K-Neighbors model RMSE: {rmse_knr}\n' +  
      f'Desision tree model RMSE: {rmse_dtr}\n' + 
      f'Lasso model RMSE: {rmse_las}\n' + 
      f'Ridge model RMSE: {rmse_rid}\n')


output_string = [(f'{(ridge.coef_[0][i]*price_scaler.scale_)[0]: .2f}'+ ' ' + X_train.columns[i]) for i in range(len(X_train.columns))]
print('\n'.join(output_string))


def interpret_numeric_coefs(coefs, names, price_scalar, feature_scalar, train_data):
      numeric_scales   = (lambda numeric_coef : (np.array(numeric_coef)/feature_scalar.scale_[:3])/
                          price_scalar.scale_) # length-3 input
      
      return(['Numeric', names[:3], numeric_scales(coefs[:3]), len(train_data)])

def interpret_coefs(coefs, names, price_scalar, feature_scalar, substring, train_data):
      #assumes coefs and prices are scaled.
      indicator_transform = (lambda indicator : np.array(indicator)/price_scalar.scale_)
      
      indicator_cols = [names[i].__contains__(substring) for i in range(len(names))]

      sample_counts = [sum(train_data[name] == 1) for name in names[indicator_cols]]
      
      interpret_indicators = [substring.capitalize(), 
                              np.char.replace(names.to_numpy(dtype = 'str')[indicator_cols], 
                                              old = substring + '_', new = ''), 
                              indicator_transform(coefs[indicator_cols]),
                              sample_counts]
      
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
      
      sns.set(rc = {'figure.figsize': (8,8)})
      sns.set_context(rc = {'patch.linewidth': 0.6})
      ax = sns.barplot(df, x = 'category', y = 'value', 
                       palette = paint_color_dict,
                       edgecolor = 'black', 
                       kwargs = {'width': np.array(interpretation[3])/max(interpretation[3])*0.8})
      ax.set(ylabel = 'Value (USD)', xlabel = interpretation[0])
      for i in ax.containers:
            print(i)
            ax.bar_label(i, labels = interpretation[3], label_type = 'center')
      plt.xticks(rotation=90)
      plt.show()

price_scaler.inverse_transform(pd.DataFrame([0,1]))
# price_scaler.

print_interpretation(interpret_coefs(ridge.coef_[0], X_train.columns, price_scaler, cov_scaler, 'manufacturer', X_train))
plot_interpretation(interpret_coefs(ridge.coef_[0], X_train.columns, price_scaler, cov_scaler, 'type', X_train))
# print_interpretation(['Decision tree', X_train.columns, perm_imp['importances_mean']*100])



categories = ['manufacturer', 'condition', 'fuel', 
              'title_status', 'transmission', 'drive', 
              'size', 'type', 'paint_color',
              'numeric']

#sums to total of 259556.6 USD
#foo sums to total of 3.1

for category in categories:
      print(category)
      cat_interpretation = interpret_numeric_coefs(
            ridge.coef_[0], X_train.columns, price_scaler, cov_scaler, X_train) if category == 'numeric' else interpret_coefs(
                  ridge.coef_[0], X_train.columns, price_scaler, cov_scaler, category, X_train)
      # print(cat_interpretation)
      print(f'Total weight category (Ridge): {np.sum(np.abs(cat_interpretation[2]))/259556.6*100}')
      foo = np.sum(perm_imp['importances_mean'] * [X_train.columns[i].__contains__(category) for i in range(len(X_train.columns))])
      print(f'Total weight category (DTR): {foo/3.1*100: .2f}')
      #foo for numerics: {'year': 0.815, 'odometer': 0.524, 'cylinders': 0.35}
      # print_interpretation(cat_interpretation)
      plot_interpretation(cat_interpretation)

