import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('heart(1).csv')
print(df.head())
print(df.shape)
##to check null value in data
print(df.isnull().values.any())
#####To check dataset is balanced or not
# count_o_values=len(df(df['Prediction(Heart disease)']==0))
# print(count_o_values)

print(df.corr)

#Slicing
X_features_input=df.iloc[:,:-1]
y_label_output=df.iloc[:,-1]

#Splitting into training & Splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_features_input,y_label_output,test_size=0.20,random_state=1)

#Training on SVM
from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

##Save the model
import joblib
filename='supportmodel.sav'
joblib.dump(classifier,'supportmodel.sav')
##load the model
loaded_model=joblib.load('supportmodel.sav')
result=loaded_model.predict(X_test)
result2=loaded_model.score(X_test,y_test)
print(result2)

#############################################
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score
print('Accuracy Score:',accuracy_score(y_test,y_pred))

f1_metric=f1_score(y_test,y_pred,average='macro')
print(f1_metric)


#taking input from user
input_heartpatient=int(input("Enter your value"))
input_heartpatient2=int(input("Enter your value"))
#####output
output=loaded_model.predict([[input_heartpatient,input_heartpatient2]])
print(output)