
Price Prediction of used cars
-----------------------------
used car price is predicted here. 
Car data is a kaggle data set.
The features are 'Year',  'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'. 
The output is “selling price”.
The “year of production” can be converted to “years used”, as latter is the more suitable feature for the prediction.
The features ‘Fuel_Type', 'Seller_Type', 'Transmission', 'Owner' are categorical features and can be converted to one hot encoding.

Seaborn pairplot is made to understand the correlation between the features and the output. 
Random forest regressor was used for predictions. 
