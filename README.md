# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for predicting Employee Salary.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting Employee Salary.
Developed by: B R SWETHA NIVASINI
Register Number:  
*/
```
```
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("drive/MyDrive/ML/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```


## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

![Screenshot 2025-04-19 165922](https://github.com/user-attachments/assets/501126a8-d6fd-48e0-b4ed-fd6fe7b3cfe0)

![Screenshot 2025-04-19 165932](https://github.com/user-attachments/assets/43c48203-9b4a-47af-a45c-84c42ea1cad4)

![Screenshot 2025-04-19 165939](https://github.com/user-attachments/assets/647e6b86-08c6-4601-8d49-5845afa10493)

![Screenshot 2025-04-19 165948](https://github.com/user-attachments/assets/d1ca7960-26f2-479e-915c-b1949489a097)

![Screenshot 2025-04-19 165956](https://github.com/user-attachments/assets/02c11ae2-cfb4-4437-9add-e609618b12da)

![Screenshot 2025-04-19 170004](https://github.com/user-attachments/assets/03c5cd34-03e8-489d-beb2-d2b46cf3ec7b)

![Screenshot 2025-04-19 170010](https://github.com/user-attachments/assets/8e145bb2-4858-47d8-abfa-7012aaa92e3a)

![Screenshot 2025-04-19 170020](https://github.com/user-attachments/assets/8f7ac2c8-3d5f-4b08-a536-54ee5b5272c5)










## Result:
Thus, the program to implement the Decision Tree Regression Model for Predicting the Salary of the Employee is written and verified using Python programming.
