import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler #preprocessing
#Train and Test split
from sklearn.model_selection import train_test_split # splitiing data into training and test.

#Classification
from sklearn.neighbors import KNeighborsClassifier # KNN model
from sklearn.svm import SVC # Support Vector Classifier

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV # hyperparameter tuning
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#Dataset Analysis 



dd=pd.read_csv('C:/Users/ishaa/.spyder-py3/heart.csv')
#We are using a numerical plus categorical dataset for Heart Patients and is a classification problem.
print("Number of rows is = ", dd.shape[0], " \nNumber of columns is = " , dd.shape[1]) 
print(dd.duplicated().sum())
print(dd.drop_duplicates())# dropping duplicated rows. We could add more details regarding what to do with the duplicated row.
print(dd.isnull().sum()) 
print(dd.describe())
print(dd['output'].value_counts())
#0 means high chance of heart attack for output
#1 means low chance of heart attack for output.

# Lets deal with missing values first.
dd.fillna(dd.mean(numeric_only=True).round(0), inplace=True) # filling missing values with mean.
print(dd.isnull().sum())#check for missing values again.
print(dd.columns)
plt.figure()
dd.hist(figsize=(12, 10))
plt.tight_layout()# Ensures proper spacing between subplots
plt.savefig('histogram.png')  
plt.show()







# Lets give the dataset description as mentioned on Kaggle.
"""

About this dataset
Age : Age of the patient

Sex : Sex of the patient

exang: exercise induced angina (1 = yes; 0 = no)

ca: number of major vessels (0-3)

cp : Chest Pain type chest pain type

Value 1: typical angina
Value 2: atypical angina
Value 3: non-anginal pain
Value 4: asymptomatic
trtbps : resting blood pressure (in mm Hg)

chol : cholestoral in mg/dl fetched via BMI sensor

fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

rest_ecg : resting electrocardiographic results

Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach : maximum heart rate achieved

target : 0= less chance of heart attack 1= more chance of heart attack

"""
#Finding correlation between output and other factors.
 # With this heatmap we can find the correlation to output to find the right features and their importance for making the prediction.
 
 
 
#Feature Engineering

correlation= pd.DataFrame(dd.corr().output)
plt.figure(figsize=(12,10))
sns.heatmap(correlation,annot=True)
# This is a heatmap explaining correlation of the feature to output.
data = dd.copy()
Xa = data.iloc[:,0:13]  #independent columns
ya = data.iloc[:,-1]    #target column 
#apply SelectKBest class to extract top best features
best = SelectKBest(score_func=chi2, k=10) #Ties between features with equal scores will be broken in an unspecified way.
fit = best.fit(Xa,ya) # best is used to refer to the best features related to the perforamnce variable.
ddscores = pd.DataFrame(fit.scores_)
ddcolumns = pd.DataFrame(Xa.columns)
featureScores = pd.concat([ddcolumns,ddscores],axis=1)
featureScores.columns = ['Columns','Score']  #naming the dataframe columns
print(featureScores.nlargest(12,'Score'))  #print best features
plt.figure(figsize=(14,7))
plt.bar(featureScores['Columns'],featureScores['Score'])
plt.xlabel('Columns')
plt.ylabel('Score')
plt.title('Best Features')
plt.show() # We plot the scores with highest score being most related to the target.

for s in dd.columns:
    print(s,len(dd[s].unique()))
#Printing unique values of each column in the dataset.




# Data Visualization
ddvisual=dd.copy()
def change(sex):
    if sex == 0:
        return 'female'
    else :
        return 'male'
    
def out(prob):
     if prob ==0:
         return 'Low chance of heart attack'
     else :
         return 'High chance of heart attack'
     
     
ddvisual['sex']= ddvisual['sex'].apply(change)
ddvisual['output']= ddvisual['output'].apply(out) 
sns.set_palette('tab10')
sns.set_style('ticks')
plt.figure()
sns.countplot(data=ddvisual,x='sex',hue='output')
plt.savefig('Gender on output ')


#We have 4 Categorical columns as seen in Data Description:
#cp — chest_pain_type
#restecg — rest_ecg_type
#slope — st_slope_type
#thal — thalassemia_type
plt.figure()
sns.countplot(data= ddvisual, x='slp',hue='output')
plt.title('Slope v/s Target\n')
plt.savefig('Slope reference to target ')

ddvisual2=dd[dd['output']==1][['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']]
plt.figure()
sns.jointplot(data=ddvisual2,
              x='age',y='trtbps')  # plotting feature correlation among each other.
plt.figure()
sns.jointplot(data=ddvisual2,
              x='age',y='chol')


dd.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg_type', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target']
#.loc - Access a group of rows and columns by label(s) or a boolean array.
dd.loc[dd['chest_pain_type'] == 0, 'chest_pain_type'] = 'asymptomatic'
dd.loc[dd['chest_pain_type'] == 1, 'chest_pain_type'] = 'atypical angina'
dd.loc[dd['chest_pain_type'] == 2, 'chest_pain_type'] = 'non-anginal pain'
dd.loc[dd['chest_pain_type'] == 3, 'chest_pain_type'] = 'typical angina'
#restecg - rest_ecg_type
dd.loc[dd['rest_ecg_type'] == 0, 'rest_ecg_type'] = 'left ventricular hypertrophy'
dd.loc[dd['rest_ecg_type'] == 1, 'rest_ecg_type'] = 'normal'
dd.loc[dd['rest_ecg_type'] == 2, 'rest_ecg_type'] = 'ST-T wave abnormality'
#slope - st_slope_type
dd.loc[dd['st_slope_type'] == 0, 'st_slope_type'] = 'downsloping'
dd.loc[dd['st_slope_type'] == 1, 'st_slope_type'] = 'flat'
dd.loc[dd['st_slope_type'] == 2, 'st_slope_type'] = 'upsloping'
#thal - thalassemia_type
dd.loc[dd['thalassemia_type'] == 0, 'thalassemia_type'] = 'nothing'
dd.loc[dd['thalassemia_type'] == 1, 'thalassemia_type'] = 'fixed defect'
dd.loc[dd['thalassemia_type'] == 2, 'thalassemia_type'] = 'normal'
dd.loc[dd['thalassemia_type'] == 3, 'thalassemia_type'] = 'reversable defect'
#One hot encoding

data = pd.get_dummies(dd, drop_first=False)
print(data.columns)
df_temp = data['thalassemia_type_fixed defect']
data = pd.get_dummies(dd, drop_first=True)
print(data.columns)
# One hot encoding dropped column thalessmia type fixed defect which is a useful column compared to thalessmia type nothing.


frames = [data, df_temp]
result = pd.concat(frames,axis=1)#Concatenate pandas objects along a particular axis.
result.drop('thalassemia_type_nothing',axis=1,inplace=True)
resultc = result.copy()

sc = StandardScaler()#Standardize features by removing the mean and scaling to unit variance.
col_scale = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
result[col_scale] = sc.fit_transform(result[col_scale])

X = result.drop('target', axis = 1)
y = result['target'] # Only scaling necessary columns.

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)#program should use a specific seed of randomness (42)
cols = X_train.columns

error_rate = []

for error in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=error)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='X',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# got the least error rate against k value at 9.
classifierKNN = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)# Classifier implementing the k-nearest neighbors vote.
y_pred = classifierKNN.predict(X_test)
cm1 = confusion_matrix(y_test,y_pred)
print(cm1)
print("Accuracy of K-Nearest Neighbors:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure()
indices = np.arange(len(y_test))
plt.plot(indices, y_test, marker='o', color='blue', label='Real')
plt.plot(indices, y_pred, marker='o', color='red', label='Predicted')
plt.xlabel('Smpl')
plt.ylabel('Class')
plt.title('Test vs. Real ')
plt.legend()
plt.show()



#We find out error rate is lowest among 6-8 ish range


classifierLin = SVC(kernel = 'linear')
classifierLin.fit(X_train, y_train)
y_pred_svc = classifierLin.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred_svc)# C-Support Vector Classification.
print(cm2)
print("Accuracy of Support Vector Classification:",accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))


plt.figure()
x_values = np.arange(len(y_test))

# Plot the predicted values
plt.plot(x_values, y_pred, label='y_pred', marker='o')
plt.plot(x_values, y_pred_svc, label='y_pred_svc', marker='o')
plt.title('Prediction Comparison for Support Vector Classification ')
plt.show()
#param_grid={'C':[0.01,0.1,1,10,100,1000,10000,100000],
#             'gamma':[1,0.1,0.01,0.001,0.0001,0.00001,0.000001]}
grid=GridSearchCV(SVC(),param_grid={'C': [0.01,0.1,1, 10,100,1000,10000,100000], 'gamma': [1,0.1,0.01,0.001,0.0001,0.00001,0.000001]})
grid.fit(X_train,y_train) # Exhaustive search over specified parameter values for an estimator.
grid_pred=grid.predict(X_test)
print(confusion_matrix(y_test,grid_pred))
print("Accuracy after GridSearch tuning :",accuracy_score(y_test, grid_pred))
print (classification_report(y_test, grid_pred))

pca=PCA(n_components=0.95) # chosen according to variance ratio
pca_data=pca.fit_transform(X_train)
classifierKNN.fit(pca_data, y_train)
print('Accuracy after Principal Component Analysis',classifierKNN.score(pca.transform(X_test), y_test))

# applying a keras sequential model on non preproccesed data to see results.
Xa_train,Xa_test,ya_train,ya_test=train_test_split(Xa,ya,test_size=.20,random_state=5) 

scaler = MinMaxScaler() #create an instance of the function
Xa_train_scaled = scaler.fit_transform(Xa_train) #fit and tranform training data
Xa_train = pd.DataFrame(Xa_train_scaled)
Xa_test_scaled = scaler.transform(Xa_test) #only tranform test data
Xa_test = pd.DataFrame(Xa_test_scaled)


model = Sequential()

model.add(Dense(13,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(13,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(13,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(13,activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(1,activation='sigmoid')) #output is binary values so using sigmoid function
model.compile(optimizer='adam',loss='binary_crossentropy') #use binary_crossentropy as output is binary values

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=30)
#model.save('Sequential.h5')
history=model.fit(x=Xa_train,y=ya_train,epochs=150,validation_data=(Xa_test,ya_test),callbacks=[early_stop])
predictions=  model.predict(Xa_test[:1])
#Saving and loading model.
#from keras.models import load_model
#loaded_model = load_model('Sequential.h5')
print(predictions)
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


