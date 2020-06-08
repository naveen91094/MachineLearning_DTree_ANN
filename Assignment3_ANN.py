import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import  roc_auc_score
import scikitplot as skplt
from sklearn import metrics
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

#download tensorflow as well --pip install tensorflow
df = pd.read_csv('C:/Users/DELL/Desktop/Second Sem/ML/Assignment 3/assignment3.csv')
#df = df.drop(['song_name','song_popula'],axis=1)
features = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','ACCR','LEVI']
X = df[features]
y = df.c_manipulator
acc_score=[0]*4
pre_score=[0]*4
rec_score=[0]*4
fone_score=[0]*4
roc_score=[0]*4
cm_arr=[0]*4
y_pred_prob =[0]*4
model_name = ["Initial ANN","ANN Post OverSampling","ANN Post UnderSampling","ANN Post SMOTE Sampling"]

#data split into test & train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

def ann(X_train,y_train,model_num):
    print (model_num)
    model = Sequential()
    #model.add(Dense(15,input_dim=13,activation='relu'))
    model.add(Dense(7,activation='relu',input_shape=(8,)))
    #model.add(Dense(10,activation='relu'))
    model.add(Dense(4,activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=100,batch_size=20,validation_data=(X_test,y_test))
    #model.fit(X_train,Y_train,epochs=2,batch_size=1,verbose=1)
    scores = model.evaluate(X_test,y_test)
    print("Model Metric: " +str(model.metrics_names[1])+"Score: "+str(scores[1]))
    y_pred = model.predict_classes(X_test)
    calculate_score(y_test,y_pred,y_pred_prob,model_num,model)

def calculate_score(y_test,y_pred,y_pred_prob,model_num,model):
    y_pred_prob[model_num] = model.predict_proba(X_test)
    acc_score[model_num]= round(100 * accuracy_score(y_test, y_pred), 2)
    pre_score[model_num]= round(100 * precision_score(y_test, y_pred), 2)
    rec_score[model_num]= round(100 * recall_score(y_test, y_pred), 2)
    fone_score[model_num] = round(100 * f1_score(y_test, y_pred), 2)
    roc_score[model_num] = metrics.roc_auc_score(y_test, y_pred_prob[model_num])
    confusion_matrix(y_test,y_pred,y_pred_prob,model_num)


def confusion_matrix(y_test,y_pred,y_pred_prob,model_num):
    cm_arr[model_num] = metrics.confusion_matrix(y_test,y_pred)
    #skplt.metrics.plot_roc(y_test, y_pred_prob)
    # fpr, tpr, threshold = metrics.plot_roc_curve(y_test,y_pred_prob)
    # metrics.plot_roc_curve(clf,X_test,y_test)
    plt.show()
    if(model_num == 3):
        display_score()

def display_score():
    for i in range(4):
        print("---Model:", model_name[i],"---")
        print(" Accuracy:", acc_score[i],"%")
        print(" Precision:",pre_score[i],"%")
        print(" Recall: ",rec_score[i],"%")
        print(" F1 Score",fone_score[i],"%")
        print("Confusion Matrix", cm_arr[i])
        print("ROC AUC Score",roc_score[i])
        cm_heatmap(cm_arr[i])
        plotROC(y_pred_prob[i])

def cm_heatmap(cm):
    #plt.figure(figsize=(10,7))
    sn.heatmap(cm,annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.ylim(0,5)
    plt.show()

def plotROC(pred_prob_y):
    #print(pred_prob_y)
    #print(y_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred_prob_y)
    roc_auc2 = metrics.auc(fpr, tpr)
    #print("Roc Curve _ scikiplot", skplt.metrics.plot_roc(y_test, pred_prob_y))
    plt.plot(fpr, tpr, label='MLP AUC = %0.2f' % roc_auc2)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#initial ann model
ann(X_train,y_train,0)

#oversampling - increasing 1s
x = pd.concat([X_train,y_train],axis=1)
not_manipulated = x[x.c_manipulator == 0]
manipulated = x[x.c_manipulator == 1]
man_upsampled = resample(manipulated, replace=True,
                         n_samples=len(not_manipulated),random_state=1)
upsampled = pd.concat([not_manipulated,man_upsampled])
print(upsampled.c_manipulator.value_counts())
y_uptrain = upsampled.c_manipulator
X_uptrain = upsampled.drop('c_manipulator',axis = 1)
ann(X_uptrain,y_uptrain,1)

#under sampling - decreasing 0s
x = pd.concat([X_train,y_train],axis=1)
not_manipulated = x[x.c_manipulator == 0]
manipulated = x[x.c_manipulator == 1]
noman_downsampled = resample(not_manipulated, replace=True,
                         n_samples=len(manipulated),random_state=1)
downsampled = pd.concat([manipulated,noman_downsampled])
print( downsampled.c_manipulator.value_counts())
y_downtrain = downsampled.c_manipulator
X_downtrain = downsampled.drop('c_manipulator',axis = 1)
ann(X_downtrain,y_downtrain,2)

#SMOTE Sampling
sm = SMOTE(random_state=1)
X_sm_train,y_sm_train = sm.fit_sample(X_train,y_train)
ann(X_sm_train,y_sm_train,3)