#!/usr/bin/env python
# coding: utf-8

# # Mushroom Project

# Description:  This dataset describes 23 species of mushrooms. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and 
#     not recommended. The goal is to clasify editable and poisonous mushrooms. The type of Dataset is categorial 

# In[3]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve


# In[4]:


df=pd.read_csv('mushrooms.csv')
df.columns


# In[5]:


df.head()


# In[6]:


# As the dataset is catagorial so convert it to ordinal by using Label Encoder


# In[7]:


lb=LabelEncoder()
for column in df.columns:
    df[column]=lb.fit_transform(df[column])


# In[8]:


df.describe()


# In[9]:


#From above data it noticed that 'veil-type' is 0 thus need to remove it as its not contributing to the data


# In[10]:


df=df.drop(["veil-type"],axis=1)


# In[11]:


df.describe()


# # EDA

# In[12]:


# finding characteristics of dataset


# In[13]:


#Violin plot respresenting the distribution of classification characteristics.


# In[14]:


df_div = pd.melt(df, "class", var_name="Characteristics")
fig, ax = plt.subplots(figsize=(10,5))
p = sns.violinplot(ax = ax, x="Characteristics", y="value", hue="class", split = True, data=df_div, inner = 'quartile', palette = 'Set1')
df_no_class = df.drop(["class"],axis = 1)
p.set_xticklabels(rotation = 90, labels = list(df_no_class.columns));


# In[15]:


#check data is balanced or not ? By Barplot


# In[16]:


plt.figure()
pd.Series(df['class']).value_counts().sort_index().plot(kind='bar')
plt.ylabel("Count")
plt.xlabel("class")
plt.title('No. of Poisonous/edible mushrooms')
plt.show()


# In[17]:


# above graph shows that its balanced data


# # Findling Correlation by Heatmap

# In[18]:


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1, annot=True)
plt.yticks(rotation=0);
plt.title('correlation matrix')
plt.show()


# Splitting in X & Y

# In[19]:


X=df.drop(['class'], axis=1)
Y=df['class']


# In[20]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1)


# In[21]:


# The Algorith that we'll use here is Decision Tree Classifier


# In[22]:


clf=DecisionTreeClassifier()
clf=clf.fit(X_train,Y_train)


# In[ ]:


y_pred=clf.predict(X_test)

print("Decision Tree Classifier report \n", classification_report(Y_test, y_pred))
# In[ ]:


cfm=confusion_matrix(Y_test, y_pred)

sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None)
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label');


# In[ ]:


#Decision Tree Classifier shows 100% accuracy 


# In[ ]:


df1=pd.DataFrame(cvAccuracy)
df1.columns=['10-fold cv Accuracy']
df=df1.reindex(range(1,20))
df.plot()
plt.title("Decision Tree - 10-fold Cross Validation Accuracy vs Depth of tree")
plt.xlabel("Depth of tree")
plt.ylabel("Accuracy")
plt.ylim([0.8,1])
plt.xlim([0,20])
plt.show()


# For Evaluation classification model -Gaussian Naive Bayes (GaussianNB)

# In[ ]:


from sklearn.naive_bayes import GaussianNB

clf_GNB = GaussianNB()
clf_GNB = clf_GNB.fit(X_train, Y_train)


# In[ ]:


y_pred_GNB=clf_GNB.predict(X_test)
cfm=confusion_matrix(Y_test, y_pred_GNB)


# In[ ]:


sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None)
plt.title('Gaussian Naive Bayes confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[ ]:


print("Test data- Gaussian Naive Bayes report \n", classification_report(Y_test, y_pred_GNB))


# # AUC & ROC Curve

# In[ ]:


precision, recall, thresholds = precision_recall_curve(Y_test, y_pred_GNB)
area = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label = 'Area Under Curve = %0.3f'% area)
plt.legend(loc = 'lower left')
plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([-0.1, 1.1])
plt.xlim([-0.1, 1.1])
plt.show()


# In[ ]:


#higher the curve better the model


# In[ ]:


def roc_curve_acc(Y_test, Y_pred,method):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',label='%s AUC = %0.3f'%(method, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'b--')
    plt.ylim([-0.1, 1.1])
    plt.xlim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

roc_curve_acc(Y_test, y_pred_GNB, "Gaussian Naive Bayes")


# In[ ]:


plt.show()


# In[ ]:


#MOdel Saving


# In[ ]:


import pickle
lr=LogisticRegression()
filename='inhouse_mushroom.pkl'
pickle.dump(lr,open,filename,'wb')


# In[ ]:


#Conclusion


# In[ ]:


import numpy as np
a=np.array(Y_test)
predicted=np.array(lr.predict(X_test))
df_com=pd.DataFrame({"original":a,"predicted":predicted},index=range(len(a)))
df_com


# In[ ]:




