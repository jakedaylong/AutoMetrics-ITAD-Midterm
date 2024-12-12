import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

cardata = pd.read_csv('.venv/cars.csv')

cardata.describe(include='all')

data = cardata.drop(['Model'],axis=1)
data.describe(include='all')

data_no_rv = data.dropna(axis=0)
data_no_rv.describe(include='all')

#datasns = sns.load_dataset("data_no_rv")

sns.displot(data_no_rv['Price'])

plt.show()

#Deal with outliers

q = data_no_rv['Price'].quantile(0.99)
data_price_in = data_no_rv[data_no_rv['Price']<q]
data_price_in.describe(include='all')

sns.displot(data_no_rv['Price'])

plt.show()

#Add mileage, enginev, year

q = data_no_rv['Mileage'].quantile(0.99)
data_mileage_in = data_no_rv[data_no_rv['Mileage']<q]
data_mileage_in.describe(include='all')

sns.displot(data_no_rv['Mileage'])
plt.show()

q = data_no_rv['EngineV'].quantile(0.99)
data_engine_in = data_no_rv[data_no_rv['EngineV']<q]
data_engine_in.describe(include='all')

sns.displot(data_no_rv['EngineV'])
plt.show()

q = data_no_rv['Year'].quantile(0.99)
data_year_in = data_no_rv[data_no_rv['Year']<q]
data_year_in.describe(include='all')

sns.displot(data_no_rv['Year'])
plt.show()

#This is setting data_all to the whole set that is created after Year
#If you use a different name then update accordingly

data_all = data_engine_in[data_engine_in['Year'] < q]
data_all.describe(include='all')

data_cleaned = data_all.reset_index(drop=True)
data_cleaned.describe(include='all')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')
plt.show()

#print(data_cleaned.columns.values)

# Relax the assumptions (Prevent Overfitting)
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns

data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()

# Once the above is completed, let's train the model...
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)
print(inputs)


# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

reg = LinearRegression()
reg.fit(x_train,y_train)

y_hat = reg.predict(x_train)

plt.scatter(y_train, y_hat)
m, b = np.polyfit(y_train, y_hat, 1)  # Calculate the slope and intercept
plt.plot(y_train, m*y_train + b, color='blue')  # Add the regression line
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

exit(0)