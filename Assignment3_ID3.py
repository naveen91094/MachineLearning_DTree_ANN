from chefboost import Chefboost as chef
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

prediction = []
actual = []
df = pd.read_csv('C:/Users/DELL/Desktop/Second Sem/ML/Assignment 3/assignment3_tree.csv')
#print(df.head(), df.size)
features = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','ACCR','LEVI']
X = df[features]
y = df.Decision
#train & test wise split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1)
df_train = pd.DataFrame(X_train)
df_train['Decision'] = y_train
#print(df_train.head(), df_train.size)

df_test = pd.DataFrame(X_test)
df_test['Decision'] = y_test
#print(df_test.head(), df_test.size)

#confusion matrix function
def confusion_matrix(pred,actual):
    print("Confusion Matrix:")
    cm = metrics.confusion_matrix(actual,pred)
    print(cm)
    cm_heatmap(cm)

#heatmap function
def cm_heatmap(cm):
    sn.heatmap(cm,annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.ylim(0,5)
    plt.show()

if __name__ == '__main__':
   config = {'algorithm': 'ID3', 'enableParallelism': True}
   model = chef.fit(df_train, config)
   #check outputs/rules file to see decision tree (if-else format)
   fi = chef.feature_importance()
   print(fi)

   for index, instance in df_test.iterrows():
       prediction.append(chef.predict(model, instance))
       actual.append(instance['Decision'])

   confusion_matrix(prediction,actual)

