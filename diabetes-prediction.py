#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First we will import necessary libraries
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV


# # 1. EDA

# In[2]:


# Reading the data
df = pd.read_csv('diabetes-data.csv')
df.head()


# In[3]:


# Check data dimensions
df.shape


# In[4]:


df.describe()


# “Outcome” is the feature we are going to predict, 0 means No diabetes, 1 means diabetes.

# In[5]:


df.groupby(by='Outcome').size()
# In our dataset we have 500 patients without diabetes and 268 patients with diabetes
sns.countplot(df['Outcome'],label="Count")


# In[6]:


# Detailed distribution of the features in the dataset
sns.pairplot(data=df, hue='Outcome')


# # 2. Quality Checks

# In[7]:


df.isnull().sum()


# In[8]:


# Check how many 0 values are there in each feature
for field in df.columns:
    print('Count of 0 Entries in {feature}: {count}'.format(feature=field, count=np.count_nonzero(df[field]==0)))


# In[9]:


# Replacing zeroes with the mean value -> Giver poor results so not using
# features_with_zeros = df.columns[1:-1]
    
# df[features_with_zeros] = df[features_with_zeros].replace(0, np.nan)
# df[features_with_zeros] = df[features_with_zeros].fillna(df.mean())


# # 3. Feature Engineering

# In[10]:


features = df.columns[:8]
features


# In[11]:


X = df[features]
y = df.Outcome


# # 4. Correlation Matrix

# In[12]:


sns.heatmap(data=X.corr(),annot=True, cmap='RdYlGn')


# In[13]:


strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

logreg_model = LogisticRegression()

rfecv = RFECV(
    estimator=logreg_model,
    step=1,
    cv=strat_k_fold,
    scoring='accuracy'
)
rfecv.fit(X, y)

plt.figure()
plt.title('RFE with Logistic Regression')
plt.xlabel('Number of selected features')
plt.ylabel('10-fold Crossvalidation')

# grid_scores_ returns a list of accuracy scores
# for each of the features selected
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
print('rfecv.grid_scores_: {grid_scores}'.format(grid_scores=rfecv.grid_scores_))

# support_ is another attribute to find out the features
# which contribute the most to predicting
new_features = list(filter(
    lambda x: x[1],
    zip(features, rfecv.support_)
))

print('rfecv.support_: {support}'.format(support=rfecv.support_))

# Features are the most suitable for predicting the response class
new_features = list(map(operator.itemgetter(0), new_features))
print('\nThe most suitable features for prediction: {new_features}'.format(new_features=new_features))


# # 5. Data Standardization

# In[14]:


# Features chosen based on RFECV result
best_features = [
    'Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction'
]

X = StandardScaler().fit_transform(X[best_features])


# In[15]:


# Split your data into training and testing (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.20
)


# # 6. PCA

# In[16]:


pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio_)
print('PCA sum: {:.2f}%'.format(sum(pca.explained_variance_ratio_) * 100))


# # 7. Modelling and Evaluation

# In[17]:


# Model learning
log_reg = LogisticRegression(
    # Parameters chosen based on GridSearchCV result
    C=1,
    multi_class='ovr',
    penalty='l2',
    solver='newton-cg',
    random_state=42
)
log_reg.fit(X_train, y_train)

log_reg_predict = log_reg.predict(X_test)
log_reg_predict_proba = log_reg.predict_proba(X_test)[:, 1]


# In[18]:


# Model evaluation
print('Accuracy: {:.2f}%'.format(accuracy_score(y_test, log_reg_predict) * 100))
print('AUC: {:.2f}%'.format(roc_auc_score(y_test, log_reg_predict_proba) * 100))
print('Classification report:\n\n', classification_report(y_test, log_reg_predict))
print('Training set score: {:.2f}%'.format(log_reg.score(X_train, y_train) * 100))
print('Testing set score: {:.2f}%'.format(log_reg.score(X_test, y_test) * 100))


# In[19]:


# Confusion Matrix
outcome_labels = sorted(df.Outcome.unique())

sns.heatmap(
    confusion_matrix(y_test, log_reg_predict),
    annot=True,
    xticklabels=outcome_labels,
    yticklabels=outcome_labels
)

