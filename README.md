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
```

Program to implement the Decision Tree Regressor Model for Predicting Employee Salary.
Developed by: B R SWETHA NIVASINI
Register Number:  

```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```


## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

![Screenshot 2025-04-20 080430](https://github.com/user-attachments/assets/1fdce190-99fe-48d6-973e-e37b2cbe3065)

![Screenshot 2025-04-20 080443](https://github.com/user-attachments/assets/b15df943-2254-4943-a447-5d4e5e582e01)

![Screenshot 2025-04-20 080453](https://github.com/user-attachments/assets/5fe64c27-29bb-4bd1-9fd9-aaed57f99cf9)

![Screenshot 2025-04-20 080505](https://github.com/user-attachments/assets/a37e75ae-d078-4dfd-a9a2-e66f467a0378)

![Screenshot 2025-04-20 080517](https://github.com/user-attachments/assets/3da973c1-662c-4ef6-b4e1-ad1ebc4fab3c)


![Screenshot 2025-04-20 080527](https://github.com/user-attachments/assets/f459a553-67be-44ea-80da-acfb14319674)


![Screenshot 2025-04-20 080533](https://github.com/user-attachments/assets/58241620-1353-4df7-b9d7-66f5d171dde4)


![Screenshot 2025-04-20 080542](https://github.com/user-attachments/assets/3b4dbfcd-6411-4847-95e4-78613928e92d)


![Screenshot 2025-04-20 080550](https://github.com/user-attachments/assets/8a01751f-704e-4d56-bc5d-7c81c6cf1f09)



















## Result:
Thus, the program to implement the Decision Tree Regression Model for Predicting the Salary of the Employee is written and verified using Python programming.
