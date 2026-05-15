#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#impoting dataset
dataset=pd.read_csv('ai_job_impact.csv')
features = [
    'Age','Gender','Education_Level','Industry','Job_Role',
    'Years_Experience','AI_Adoption_Level','Automation_Risk',
    'Upskilling_Required','Salary_Before_AI','Salary_After_AI',
    'Work_Hours_Per_Week','Remote_Work',
    'Job_Satisfaction','Productivity_Change_%']

x = dataset[features]
y = dataset['Job_Status']

#to check how many coloumns have nan values
nan_counts = x.isna().sum()
nan_counts = nan_counts[nan_counts > 0]
print(nan_counts)

#handeling categorical data with feature name retention
x_original = x.copy()
categorical_cols = ['Gender', 'Education_Level', 'Industry', 'Job_Role', 
                    'AI_Adoption_Level', 'Automation_Risk', 'Upskilling_Required','Remote_Work']
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first',sparse_output=False), categorical_cols)],remainder='passthrough')
ct.fit(x_original)
feature_names = ct.get_feature_names_out()
x = ct.fit_transform(x_original)
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
cols_to_scale = [37,38,39,40,41,42,43]
x[:, cols_to_scale] = scaler.fit_transform(x[:, cols_to_scale])

#spliting into train & test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

--------------------

-------------------------------------------------------------------------

#Fitting logistic regression to training set
from sklearn.svm import SVC
svm=SVC(kernel='linear',random_state=0,probability=True)
svm=svm.fit(x_train,y_train)

#predicting the test result
y_pred6=svm.predict(x_test)




from sklearn.metrics import classification_report

print("SVC kernel Liner",classification_report(y_test, y_pred6))

feature_names = ct.get_feature_names_out()
--------------------------------------------------------------------------
#Fitting logistic regression to training set
from sklearn.svm import SVC
K_svm=SVC(kernel='rbf',random_state=0,probability=True, class_weight='balanced')
K_svm=K_svm.fit(x_train,y_train)

#predicting the test result
y_pred7=K_svm.predict(x_test)


from sklearn.metrics import classification_report

print("SVC Kernal rbf",classification_report(y_test, y_pred7))

feature_names = ct.get_feature_names_out()

------------------------------------------------------------------------

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',
                             use_label_encoder=False,random_state=42)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred8 = classifier.predict(x_test)

from sklearn.metrics import classification_report

print("XGBOOST",classification_report(y_test, y_pred8))

feature_names = ct.get_feature_names_out()









# ============================================
# IMPORTING LIBRARIES
# ============================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ============================================
# SELECT MODEL
# ============================================

# Choose any model:
# svm       -> Linear SVM
# K_svm     -> RBF Kernel SVM
# classifier -> XGBoost

model = classifier
y_pred = y_pred8


# ============================================
# 1. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap='Blues')

plt.title('Confusion Matrix XG BOOST')
plt.show()



# ============================================
# 2. ROC CURVE AND AUC SCORE
# ============================================

# Convert multiclass labels into binary
y_test_bin = label_binarize(y_test, classes=[0,1,2])

# Probability predictions
y_prob = model.predict_proba(x_test)

plt.figure(figsize=(8,6))

for i in range(3):

    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i],
                                     y_prob[:, i])

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr,
             tpr,
             label='Class {} (AUC = {:.2f})'
             .format(i, roc_auc))


# Random diagonal line
plt.plot([0,1], [0,1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()

plt.show()



# ============================================
# 3. FEATURE IMPORTANCE GRAPH (XGBOOST ONLY)
# ============================================

# Run only if using XGBoost

importance = classifier.feature_importances_

# Sorting feature importance
indices = np.argsort(importance)[::-1]

# Top 15 important features
top_n = 15

plt.figure(figsize=(12,6))

plt.bar(range(top_n),
        importance[indices[:top_n]])

plt.xticks(range(top_n),
           feature_names[indices[:top_n]],
           rotation=90)

plt.xlabel('Features')
plt.ylabel('Importance Score')

plt.title('Top 15 Feature Importance - XGBoost')

plt.tight_layout()

plt.show()



# ============================================
# 4. MODEL ACCURACY COMPARISON
# ============================================

models = ['Linear SVM',
          'RBF Kernel SVM',
          'XGBoost']

accuracy = [0.92,
            0.93,
            0.94]

plt.figure(figsize=(7,5))

bars = plt.bar(models, accuracy)

plt.xlabel('Models')
plt.ylabel('Accuracy')

plt.title('Model Accuracy Comparison')

plt.ylim(0.85,1.0)

# Accuracy labels
for i in range(len(models)):

    plt.text(i,
             accuracy[i] + 0.003,
             str(accuracy[i]),
             ha='center')

plt.show()



# ============================================
# 5. F1 SCORE COMPARISON GRAPH
# ============================================

f1_scores = [0.73,
             0.86,
             0.79]

plt.figure(figsize=(7,5))

bars = plt.bar(models, f1_scores)

plt.xlabel('Models')
plt.ylabel('Macro Avg F1 Score')

plt.title('F1 Score Comparison')

plt.ylim(0.5,1.0)

# F1 labels
for i in range(len(models)):

    plt.text(i,
             f1_scores[i] + 0.01,
             str(f1_scores[i]),
             ha='center')

plt.show()









