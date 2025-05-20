# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for predicting Employee Salary.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets
2. Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters
3. Train your model -Fit model to training data -Calculate mean salary value for each subset
4. Evaluate your model -Use a model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance
5. Tune hyperparameters -Experiment with different hyperparameters to improve performance
6. Deploy your model. Use a model to make predictions on new data in a real-world application.




## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: B R SWETHA NIVASINI

RegisterNumber: 212224040345

```
import pandas as pd
data=pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\Folders\\ML\\DATASET-20250226\\Salary.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/fb7e5837-fa27-4dd2-9fc8-87b5072ac334)

```
data.info()
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/e898c62f-8a56-4b06-9149-7b960db1ea02)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
![image](https://github.com/user-attachments/assets/b7b88868-579e-498d-9114-e41159ff1733)

```
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
```
![image](https://github.com/user-attachments/assets/af7af4d2-82a1-4f1f-88e7-6f2347954aeb)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/06fd2d05-c429-4c48-83e1-6038dbe82145)

```
from sklearn import metrics
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2
```
![image](https://github.com/user-attachments/assets/3f40d399-0f10-4ab7-8117-86ab06f5c8b1)
```
dt.predict([[5,6]])
```

![image](https://github.com/user-attachments/assets/5bddfaf6-ef95-4654-8526-11fea921f807)

















## Result:
Thus, the program to implement the Decision Tree Regression Model for Predicting the Salary of the Employee is written and verified using Python programming.
