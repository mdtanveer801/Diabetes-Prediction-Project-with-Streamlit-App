import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import warnings
warnings.filterwarnings("ignore")



df=pd.read_csv(r"E:\Data Science Notes and projects\PDF and Datasets\diabetes_prediction_dataset.csv")
df
df.head()

df.isna().any()
df.corr(numeric_only=True) 
df.shape

for column in df.columns: # itreating each column in df.columns 
    unique_values = df[column].unique() #finding unique values of each column

    #printing unique values
    print('Column "{}" has unique values: {}'.format(column, unique_values)) 
    
df["smoking_history"].value_counts()
    
df["smoking_history"].value_counts()/len(df)

df['smoking_history'] = df['smoking_history'].replace('No Info', pd.NA)

# Replace missing values with the mode it is string so we are using mode
mode_value = df['smoking_history'].mode()[0]
df['smoking_history'] = df['smoking_history'].fillna(mode_value) #filling no info values 

# Printing the updated value counts
print(df['smoking_history'].value_counts())
df.info()

df.gender.value_counts()

df.describe()

df["bmi"] = [float(str(i).replace(",", "")) for i in df["bmi"]] 

df['diabetes'].value_counts().plot(kind='barh') 

#Xlabel name
plt.xlabel('count')

#ylabel name
plt.ylabel('diabetes')

#title of the plot
plt.title('count of diabetes and Non diabetes')

#invert ylabes to no diabetes on top
plt.gca().invert_yaxis() 

#printing the plot
plt.show()

df['diabetes'].value_counts()/len(df)

df.info()

le=LabelEncoder() #activating label encoder function

le

Label_encod_columns=['gender','smoking_history']  #selecting columns to apply labelencoder in next step

df[Label_encod_columns]=df[Label_encod_columns].apply(le.fit_transform)

df.head(3)

sns.boxplot(data=df[['age','blood_glucose_level','bmi']])

sns.boxplot(data=df['HbA1c_level'])
sns.lmplot(data=df, x='blood_glucose_level', y='diabetes', fit_reg=False)#implot plot

sns.pairplot(df) #using pairplot to check relation between parameters

#print the pairplot
plt.show()

df.corr()

plt.figure(figsize=(20,8)) #figsize

#printing graphical representations of
df.corr()['diabetes'].sort_values(ascending=False).plot(kind='bar')

#selecting X variables
X = df.loc[:, 'age':'heart_disease'].join(df.loc[:, 'bmi':'blood_glucose_level']) 
X

y=df.loc[:,'diabetes'] 
y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

X_train.head() 
print('Shape of Train data')

print(X_train.shape)

print(y_train.shape)

print('Shape of Testing data')

print(X_test.shape)

print(y_test.shape)
    
ss=StandardScaler() #activating StandardScaler()

ss   

X_train_scaled=ss.fit_transform(X_train)
if len(X_test.shape) == 1:   #if x is 1d array
    X_test = X_test.values.reshape(-1, 1) #converting to 2d array

X_test_scaled = ss.fit_transform(X_test)

model_lr=LogisticRegression() 
model_lr.fit(X_train_scaled,y_train)

y_pred=model_lr.predict(X_test_scaled) #predecting y_test data
y_pred[:10]

y_test[:10]

accuracy_score(y_pred,y_test)

print(classification_report(y_pred,y_test))


confusion_matrix(y_pred,y_test)
y_train.value_counts()
value_counts=y_train.value_counts()

plt.figure(figsize=(16, 8))

plt.pie(value_counts, labels=value_counts.index, autopct='%1.2f%%', startangle=140)

plt.title('Distribution of y_train')

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

from imblearn.over_sampling import SMOTE # using smote function to balance our set

smote=SMOTE()

X_ovs,y_ovs=smote.fit_resample(X,y) #passing X and y variables to it to balance out data to 50 50

fig, oversp = plt.subplots() 

oversp.pie( y_ovs.value_counts(), autopct='%.2f')

oversp.set_title("Over-sampling")

plt.show()
Xr_train,Xr_test,yr_train,yr_test=train_test_split(X_ovs,y_ovs,train_size=0.7,random_state=42)

print('train data shape')

print(Xr_train.shape)

print(yr_train.shape)

print('test data shape')

print(Xr_test.shape)

print(yr_test.shape)
print('y_train and y_test value_count')
print(yr_train.value_counts())
print(yr_test.value_counts())

ss=StandardScaler()

ss
data=Xr_train,Xr_test



xr_train_sc=ss.fit_transform(Xr_train) # scaling our resampling data xr train


Xr_test_sc=ss.fit_transform(Xr_test)

Xr_train_scaled = pd.DataFrame(xr_train_sc) #Xr_train_scaled converting into the data frame

print(Xr_train_scaled.shape)
Xr_train_scaled.head()
print(yr_train.shape)
Xr_test_scaled=pd.DataFrame(Xr_test_sc) #Xr_test converting into the dataframe

print(Xr_test_scaled.shape)
Xr_test_scaled.head() 

model_lk=LogisticRegression()  


model_lk.fit(Xr_train_scaled,yr_train)
y_pred_lr=model_lk.predict(Xr_test_scaled) #predecting yr_test data
y_pred_lr[:10]

yr_test[:10]

print(classification_report(y_pred_lr,yr_test))

confusion_matrix(y_pred_lr,yr_test) 
model_dtc=DecisionTreeClassifier() 

# passing xr_train_scaled, yr_train to trining the model
model_dtc.fit(Xr_train_scaled,yr_train)

model_dtc

y_pred_dtc=model_dtc.predict(Xr_test_scaled)

print(classification_report(y_pred_dtc,yr_test))

confusion_matrix(y_pred_dtc,yr_test)
model_rfc=RandomForestClassifier() #activating the fuction

model_rfc.fit(Xr_train_scaled,yr_train)
y_pred_rfc=model_rfc.predict(Xr_test_scaled)

print(classification_report(y_pred_rfc,yr_test))

confusion_matrix(y_pred_rfc,yr_test)

model_xgb=XGBClassifier()

model_xgb.fit(Xr_train_scaled,yr_train)

y_pred_xgb=model_xgb.predict(Xr_test_scaled)

print(classification_report(y_pred_xgb,yr_test))

confusion_matrix(y_pred_xgb,yr_test)

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression

# Define the parameter grid to search over
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2']                # Penalty type
}

# Create a Logistic Regression model
logistic = LogisticRegression()

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=logistic, param_grid=param_grid, cv=10)

# Initialize an empty list to store the accuracy scores
accuracy_scores = []

# Perform cross-validation 10 times
for _ in range(10):
    # Fit the GridSearchCV object to the training data
    grid_search.fit(Xr_train_scaled, yr_train)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    
    # Perform cross-validation with the best model
    cv_scores = cross_val_score(grid_search.best_estimator_, Xr_train_scaled, yr_train, cv=10)
    
    # Store the mean accuracy score
    accuracy_scores.append(cv_scores.mean())

# Print the accuracy scores obtained over 10 iterations
#print("Accuracy scores over 10 iterations:", accuracy_scores)
print("Accuracy scores over 10 iterations:", ["{:.2f}".format(score) for score in accuracy_scores])


# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters found:", best_params)
print("Best cross-validation score:", best_score)

from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model with the best parameters
final_model = LogisticRegression(C=0.001, penalty='l2')

# Fit the final model to the entire training dataset
final_model.fit(Xr_train_scaled, yr_train)

import pickle

# Save the final model to a pickle file
with open('final_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)
    
    import pickle
import numpy as np

# Load the model from the pickle file
with open('final_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define the mean and standard deviation of the training data
mean_values = [41.885856, 0.07485, 0.03942, 27.320767, 5.527507, 138.058060]
std_values = [22.516840, 0.26315, 0.194593, 6.636783, 1.070672, 40.708136]

# Define the input features for prediction
age = 30
hypertension = 0
heart_disease = 0
bmi = 100.0
HbA1c_level = 5.0
blood_glucose_level = 90

# Scale the input features manually
scaled_features = [(x - mean) / std for x, mean, std in zip(
    [age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level],
    mean_values, std_values
)]

# Make predictions on the scaled data
prediction = loaded_model.predict([scaled_features])

# Print the prediction
if prediction[0] == 1:
    print("Diabetic")
else:
    print("Not Diabetic")

