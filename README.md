# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries for data handling and model building.
2. Load the dataset and view basic information and structure.
3. Encode categorical variables (e.g., salary).
4. Select features (x) and the target variable (y).
5. Split the data into training and test sets (80% training, 20% testing).
6. Train the Decision Tree classifier using the training data.
7. Predict the outcomes using the test data.
8. Calculate the accuracy of the model.
9. Make a prediction using new sample input data.
10. End program

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:   SURESH S
RegisterNumber: 212223040215
*/
```

```
import pandas as pd
data = pd.read_csv("/content/Employee .csv")
data.head()
```
<img width="950" alt="image" src="https://github.com/user-attachments/assets/499a4039-a824-45de-835b-a99337564c58">

```
data.info()
```
<img width="650" alt="image" src="https://github.com/user-attachments/assets/362a2428-4975-4e2a-bb6d-75d0b881cd35">

```
data.isnull().sum()
data["left"].value_counts() 
```
<img width="450" alt="image" src="https://github.com/user-attachments/assets/9bd38626-b84e-46e9-b533-03464956c47e">

```
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"]) 
data.head() 
```
<img width="950" alt="image" src="https://github.com/user-attachments/assets/d951a005-337f-47e2-b782-441d83be78bd">

```
x=data[["satisfaction_level", "last_evaluation", "number_project","average_montly_hours", "time_spend_company",
        "Work_accident", "promotion_last_5years", "salary"]]
x.head() 
y=data["left"]
x.head()
```
<img width="950" alt="image" src="https://github.com/user-attachments/assets/9af09064-2513-437b-9388-ee424b14156c">

```
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state= 2)
from sklearn.tree import DecisionTreeClassifier 
dt=DecisionTreeClassifier(criterion="entropy") 
dt.fit(x_train,y_train) 
y_pred=dt.predict(x_test) 
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred) 
print(accuracy) 
```
<img width="350" alt="image" src="https://github.com/user-attachments/assets/64a0580b-03f3-4c23-9a48-c35a85290c3e">

```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
<img width="950" alt="image" src="https://github.com/user-attachments/assets/95c35f14-89cf-4ea5-b533-0c5dc7df4625">

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
