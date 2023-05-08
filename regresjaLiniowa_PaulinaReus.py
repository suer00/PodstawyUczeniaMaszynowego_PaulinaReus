# Code source: Jaques Grobler
# License: BSD 3 clause
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression


#------CZYSZCZENIE TABELI
df= pd.read_csv('Salary Data.csv')
df=df.replace('Male',0)
df=df.replace('Female',1)
df=df.replace("Bachelor's",1)
df=df.replace("Master's",2)
df=df.replace("PhD",3)
#zbyt dużo róznych nazw tytułów, dlatego też należało usunąć "Job Titile"
df=df.drop(columns=['Job Title'])
df=df.dropna(how='all')
print(df.head())


x = df[['Age', 'Gender', 'Education Level', 'Years of Experience']]

y =df['Salary']

diabetes_X_train,diabetes_X_test, diabetes_y_train, diabetes_y_test=train_test_split(x, y,test_size=0.2, random_state=42)



##------------------------Regresja Liniowa--------------------------------

model = LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)

# predykcja wyników
y_pred_train_regl = model.predict(diabetes_X_train)
y_pred_test_regl = model.predict(diabetes_X_test)


train_mse_regl = mean_squared_error(diabetes_y_train, y_pred_train_regl, squared=True)
test_mse_regl = mean_squared_error(diabetes_y_test, y_pred_test_regl, squared=True)
train_r2_regl = r2_score(diabetes_y_train, y_pred_train_regl)
test_r2_regl = r2_score(diabetes_y_test, y_pred_test_regl)


####-----------------------LASSO---------------------------------------------------------

model = Lasso(alpha=0.1)
model.fit(diabetes_X_train, diabetes_y_train)


y_pred_train_lasso = model.predict(diabetes_X_train)
y_pred_test_lasso = model.predict(diabetes_X_test)



train_mse_lasso = mean_squared_error(diabetes_y_train, y_pred_train_lasso , squared=True)
test_mse_lasso = mean_squared_error(diabetes_y_test, y_pred_test_lasso, squared=True)
train_r2_lasso = r2_score(diabetes_y_train, y_pred_train_lasso )
test_r2_lasso = r2_score(diabetes_y_test, y_pred_test_lasso)

print(f"Mean Squared Error dla zbioru treningowego: {train_mse_lasso}")
print(f"Mean Squared Error dla zbioru testowego: {test_mse_lasso}")
print(f"R^2 dla zbioru treningowego: {train_r2_lasso}")
print(f"R^2 dla zbioru testowego: {test_r2_lasso}")


##------------------------Regresja wielomianowa----------------------\

# utworzenie nowych cech
poly = PolynomialFeatures(degree=3)
X_polyF_train = poly.fit_transform(diabetes_X_train)
X_polyF_test = poly.transform(diabetes_X_test)

# tworzenie i trenowanie modelu
model = LinearRegression()
model.fit(X_polyF_train, diabetes_y_train)


y_pred_train_m = model.predict(X_polyF_train)
y_pred_test_m = model.predict(X_polyF_test)


train_mse_m = mean_squared_error(diabetes_y_train, y_pred_train_m, squared=True)
test_mse_m = mean_squared_error(diabetes_y_test, y_pred_test_m, squared=True)
train_r2_m = r2_score(diabetes_y_train, y_pred_train_m)
test_r2_m = r2_score(diabetes_y_test, y_pred_test_m)

print(f"Mean Squared Error dla zbioru treningowego: {train_mse_m}")
print(f"Mean Squared Error dla zbioru testowego: {test_mse_m}")
print(f"R^2 dla zbioru treningowego: {train_r2_m}")
print(f"R^2 dla zbioru testowego: {test_r2_m}")

##------------------Drzewo decyzyjne----------------

model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(diabetes_X_train, diabetes_y_train)

# predykcja wyników
y_pred_train_t = model.predict(diabetes_X_train)
y_pred_test_t = model.predict(diabetes_X_test)


train_mse_t = mean_squared_error(diabetes_y_train, y_pred_train_t, squared=True)
test_mse_t = mean_squared_error(diabetes_y_test, y_pred_test_t, squared=True)
train_r2_t = r2_score(diabetes_y_train, y_pred_train_t)
test_r2_t = r2_score(diabetes_y_test, y_pred_test_t)


##-----------------------PODSUMOWANIE----------------------------

data={'Funkcja':['Regresja liniowa','Regresja LASSO','Regresja Wielomianowa','Dryewo deczyzjne'],
      'Błąd Średniokwadratowy zestawy testowego':[test_mse_regl, test_mse_lasso, test_mse_m, test_mse_t],
      'R^2 zestawu testowego':[test_r2_regl,test_r2_lasso,test_r2_m,test_r2_t]}

df_podsumowanie= pd.DataFrame(data)
print(tabulate(df_podsumowanie, headers='keys', tablefmt='psql'))