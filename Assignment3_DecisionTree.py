import pandas as pd
import scikitplot as skplt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn import tree
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv('C:/Users/DELL/Desktop/Second Sem/ML/Assignment 3/assignment3.csv')
features = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','ACCR','LEVI']
X = df[features]
y = df.c_manipulator
count = 0
#train & test wise split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1)

#decision tree classifier function
def decisiontree(X_train,y_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print(y_pred)
    y_pred_prob = clf.predict_proba(X_test)
    #print(y_pred_prob)
    calculate_score(y_pred,y_pred_prob,clf)
    confusion_matrix(y_pred)
    make_tree(clf)
    #count = count + 1

#calculate score function
def calculate_score(y_pred,y_pred_prob,clf):
    print("Accuracy Score:", round(100*metrics.accuracy_score(y_test, y_pred),2),"%")
    print("Precision Score:", round(100*metrics.precision_score(y_test, y_pred),2),"%")
    print("Recall Score:", round(100*metrics.recall_score(y_test, y_pred),2),"%")
    print("F1 Score:", round(100*metrics.f1_score(y_test, y_pred),2),"%")
    #print(metrics.classification_report(y_test,y_pred))
    #print("ROC Curve:",metrics.roc_curve(y_test,y_pred_prob[:,1]))
    print("Roc AUC score:",metrics.roc_auc_score(y_test,y_pred_prob[:,1]))
    print("Roc Curve _ scikiplot",skplt.metrics.plot_roc(y_test,y_pred_prob))
    plt.show()

#confusion matrix function
def confusion_matrix(y_pred):
    print("Confusion Matrix:")
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)
    cm_heatmap(cm)

#heatmap function
def cm_heatmap(cm):
    sn.heatmap(cm,annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.ylim(0,5)
    plt.show()

def make_tree(clf):
    tree.plot_tree(clf)
    plt.show()

print("---Model 1: Initial Decision Tree---")
decisiontree(X_train,y_train)

#oversampling - increasing 1s
print("---Model 2: Over Sampling---")
x = pd.concat([X_train,y_train],axis=1)
not_manipulated = x[x.c_manipulator == 0]
manipulated = x[x.c_manipulator == 1]
man_upsampled = resample(manipulated, replace=True,
                         n_samples=len(not_manipulated),random_state=1)
upsampled = pd.concat([not_manipulated,man_upsampled])
print(upsampled.c_manipulator.value_counts())
y_uptrain = upsampled.c_manipulator
X_uptrain = upsampled.drop('c_manipulator',axis = 1)
decisiontree(X_uptrain,y_uptrain)

#under sampling - decreasing 0s
print("---Model 3: Under Sampling---")
x = pd.concat([X_train,y_train],axis=1)
not_manipulated = x[x.c_manipulator == 0]
manipulated = x[x.c_manipulator == 1]
noman_downsampled = resample(not_manipulated, replace=True,
                         n_samples=len(manipulated),random_state=1)
downsampled = pd.concat([manipulated,noman_downsampled])
print( downsampled.c_manipulator.value_counts())
y_downtrain = downsampled.c_manipulator
X_downtrain = downsampled.drop('c_manipulator',axis = 1)
decisiontree(X_downtrain,y_downtrain)

#SMOTE Sampling
print("---Model 4: SMOTE Sampling---")
sm = SMOTE(random_state=1)
X_sm_train,y_sm_train = sm.fit_sample(X_train,y_train)
decisiontree(X_sm_train,y_sm_train)

