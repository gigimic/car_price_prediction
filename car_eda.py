import numpy as np
import pandas as pd 

df = pd.read_csv('car_data.csv') 
print(df.shape)
print(df.describe())
# print(df.head())
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df.isnull().sum())
final_data=df[['Year',  'Selling_Price',  'Present_Price',  'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission',  'Owner']]
# print(final_data.head())
final_data['current year'] = 2020
final_data['no year'] = final_data['current year'] - final_data['Year']
final_data.drop(['Year'], axis=1, inplace = True)
final_data.drop(['current year'], axis=1, inplace = True)
# print(final_data.head())
final_data=pd.get_dummies(final_data, drop_first = True)
print(final_data.head())
# print(final_data.corr())


col_names = final_data.columns
# print(col_names)
# print("number of col ", len(final_data.columns))

# for index, col_n in enumerate(df.columns):
# 	print(index, col_n,  len(df[col_n].unique()))

# print(df.nunique())

counts = df.nunique()
col_to_del = [index for index, v in enumerate(counts) if v==2]
# print(col_to_del)

for index, col_n in enumerate(df.columns):
    num_uniq = len(df[col_n].unique())
    percentage = num_uniq/df.shape[0]*100
    print('{0:3d} {1:4d} {2:8.3f}'.format(index, num_uniq, percentage))

from sklearn.feature_selection import VarianceThreshold
import seaborn as sns 
import matplotlib.pyplot as plt
# %matplotlib inline
# matplotlib.use('Agg')
print(sns.__version__)
sns.pairplot(df)
plt.show()

#get correlations of each features in dataset
# corrmat = df.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(10,10))
#plot heat map
# g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plt.show()

X = final_data.iloc[:,1:]
y = final_data.iloc[:,0]
# print(X['Owner'].unique())
print(X.head())
print(y.head())
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(5).plot(kind='barh')
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state =0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV
# number of features to consider at every split 
max_features = ['auto', 'sqrt']
# maximum number of levels in tree 
max_depth = [int(x) for x in np.linspace(5, 30, num =6)]
# Minimum number of samples required to split a node 
min_samples_split = [2, 5, 10, 15, 100]
# Minumum number of samples required at each lead node 
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf }

print(random_grid)
# rf = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator =rf, param_distributions = random_grid,
# scoring = 'neg_mean_squared_error', n_iter = 10, cv = 5, verbose = 2, random_state = 42, n_jobs =1)
# rf_random.fit(X_train, y_train)