```python
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as logr
from sklearn.model_selection import cross_val_score, LeaveOneOut, RepeatedKFold 
from sklearn.metrics import mean_squared_error, roc_curve, auc, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.exceptions import ConvergenceWarning 
from sklearn import tree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sns
from kneed import KneeLocator
```


```python
math_df = pd.read_csv('student-mat.csv')
por_df = pd.read_csv('student-por.csv')
```


```python
math_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GP</td>
      <td>F</td>
      <td>18</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>4</td>
      <td>4</td>
      <td>at_home</td>
      <td>teacher</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>2</td>
      <td>health</td>
      <td>services</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>15</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>3</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>6</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>390</th>
      <td>MS</td>
      <td>M</td>
      <td>20</td>
      <td>U</td>
      <td>LE3</td>
      <td>A</td>
      <td>2</td>
      <td>2</td>
      <td>services</td>
      <td>services</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>11</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>391</th>
      <td>MS</td>
      <td>M</td>
      <td>17</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>3</td>
      <td>1</td>
      <td>services</td>
      <td>services</td>
      <td>...</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>14</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>392</th>
      <td>MS</td>
      <td>M</td>
      <td>21</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>393</th>
      <td>MS</td>
      <td>M</td>
      <td>18</td>
      <td>R</td>
      <td>LE3</td>
      <td>T</td>
      <td>3</td>
      <td>2</td>
      <td>services</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>11</td>
      <td>12</td>
      <td>10</td>
    </tr>
    <tr>
      <th>394</th>
      <td>MS</td>
      <td>M</td>
      <td>19</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>other</td>
      <td>at_home</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>395 rows × 33 columns</p>
</div>




```python
math_df = pd.read_csv('student-mat.csv')
math = math_df.copy()
math = math.drop(['address', 'Pstatus', 'guardian', 'famsup','famsize','Medu','Fedu', 'Mjob', 'Fjob', 'reason', 'traveltime', 'nursery','romantic', "G1", 'G2'], axis=1)
math = pd.get_dummies(math, columns=['school', 'sex', 'schoolsup', 'paid', 'activities', 'internet', 'higher']) 
math
math = math.drop(['schoolsup_no', 'activities_no', 'paid_no', 'internet_no','higher_no', 'school_MS', 'sex_F'], axis = 1)
math
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>studytime</th>
      <th>failures</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G3</th>
      <th>school_GP</th>
      <th>sex_M</th>
      <th>schoolsup_yes</th>
      <th>paid_yes</th>
      <th>activities_yes</th>
      <th>internet_yes</th>
      <th>higher_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>390</th>
      <td>20</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>11</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>391</th>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>392</th>
      <td>21</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>393</th>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>394</th>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>395 rows × 18 columns</p>
</div>




```python
por = por_df.copy()
```


```python
por = por.drop(['address', 'Pstatus', 'guardian', 'famsup','famsize','Medu','Fedu', 'Mjob', 'Fjob', 'reason', 'traveltime', 'nursery','romantic', "G1", 'G2'], axis=1)
por = pd.get_dummies(por, columns=['school', 'sex', 'schoolsup', 'paid','activities', 'internet', 'higher'])
por = por.drop(['schoolsup_no', 'activities_no', 'paid_no', 'internet_no','higher_no', 'school_MS', 'sex_F'], axis = 1)
```


```python
student = pd.merge(math_df, por_df, on=['school', 'sex', 'age', 'address','famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery','internet'])
student.dtypes
student = student.drop(['address', 'Pstatus', 'guardian_x','guardian_y','famsup_x','famsup_y', 'famsize', 'Medu','Fedu', 'Mjob', 'Fjob', 'reason','traveltime_x','traveltime_y' ,'nursery', 'romantic_x', 'romantic_y', "G1_x", 'G2_x'], axis=1)
student = pd.get_dummies(student, columns=['school', 'sex', 'schoolsup_x','schoolsup_y' , 'paid_x', 'paid_y', 'activities_x', 'activities_y','internet', 'higher_x', 'higher_y'])
student['Weekly_Alc_Consumption'] = student['Dalc_x'] + student['Walc_x']
student = student.drop(['activities_x_no', 'activities_y_no', 'internet_no','higher_x_no', 'higher_y_no', 'paid_x_no', 'paid_y_no', 'schoolsup_x_no','schoolsup_y_no', "G1_y", 'G2_y', 'school_MS', 'sex_M', 'studytime_y','famrel_y', 'freetime_y', 'Walc_y', 'Dalc_y', 'health_y', 'higher_y_yes','activities_y_yes', 'schoolsup_y_yes', 'goout_y', 'Dalc_x','Walc_x'], axis=1)
student.columns
```




    Index(['age', 'studytime_x', 'failures_x', 'famrel_x', 'freetime_x', 'goout_x',
           'health_x', 'absences_x', 'G3_x', 'failures_y', 'absences_y', 'G3_y',
           'school_GP', 'sex_F', 'schoolsup_x_yes', 'paid_x_yes', 'paid_y_yes',
           'activities_x_yes', 'internet_yes', 'higher_x_yes',
           'Weekly_Alc_Consumption'],
          dtype='object')




```python
plt.figure(figsize=(20,20))
sns.heatmap(student.corr(),annot=True)
plt.show()
#correlation matrix to show if any of the predictors are highly related, this 
#correlation matrx is after we dropped the ones that had correlation >.7
```


    
![png](MachineLearningProject1_files/MachineLearningProject1_7_0.png)
    



```python
student.columns
```




    Index(['age', 'studytime_x', 'failures_x', 'famrel_x', 'freetime_x', 'goout_x',
           'health_x', 'absences_x', 'G3_x', 'failures_y', 'absences_y', 'G3_y',
           'school_GP', 'sex_F', 'schoolsup_x_yes', 'paid_x_yes', 'paid_y_yes',
           'activities_x_yes', 'internet_yes', 'higher_x_yes',
           'Weekly_Alc_Consumption'],
          dtype='object')




```python
new_column_names = {'studytime_x': 'studytime', 'failures_x': 'failures_math','famrel_x': 'family_rel_quality', 'freetime_x':'freetime','goout_x': "going_out", 'Dalc_x': 'Workday_Alc_consum','Walc_x': 'Weeekend_alc', 'health_x': 'health_status_',
                    'absences_x': 'abscences_math', 'G3_x':'final_grade_math',
                    'failures_x=y': 'failures_port', 'absences_y':'abscences_port', 'G3_y':'final_grade_port','school_GP': 'school', 'sex_F':'Sex', 'schoolsup_x_yes':"extra_edu_support", 'paid_x_yes':'extra_paid_classes','activities_x_yes': 'Extra_curricular', 'internet_yes':'internet_access', 'higher_x_yes':'higher_edu_desired'
}

student = student.rename(columns=new_column_names)
```


```python
import matplotlib.pyplot as plt 
num_cols = len(student.columns)
num_rows = (num_cols - 1) // 4 + 1
fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(16, 4*num_rows))
for i, column in enumerate(student.columns[1:], start=1): # Start from index 1 to skip the first column
    row_idx = i // 4
    col_idx = i % 4
    axes[row_idx, col_idx].hist(student[column], bins=20, density=True, alpha=0.6)
    axes[row_idx, col_idx].set_title(f'Distribution of {column}',fontdict={'fontsize': 10}) # Adjust fontsize here axes[row_idx, col_idx].set_xlabel(column) axes[row_idx, col_idx].set_ylabel('Density')

for i in range(num_cols, num_rows * 4): 
    fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
plt.show()
# this an be one of the visualization for the EDA portion, it shows the distrubion of all the predictors
```


    
![png](MachineLearningProject1_files/MachineLearningProject1_10_0.png)
    



```python
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

X = student.drop(['final_grade_math', 'final_grade_port'], axis=1) 
y_math = (student['final_grade_math'] >= 10).astype(int) # Binary classification for pass/fail in Math
y_portuguese = (student['final_grade_port'] >= 10).astype(int) # Binary classification for pass/fail in Portuguese

# Split the data into training and testing sets
X_train_math, X_test_math, y_train_math, y_test_math = train_test_split(X, y_math, test_size=0.2, random_state=42)
X_train_portuguese, X_test_portuguese, y_train_portuguese, y_test_portuguese = train_test_split(X, y_portuguese, test_size=0.2, random_state=42)

#DecisionTreeClassifier
dt_classifier_math = DecisionTreeClassifier(random_state=42)
dt_classifier_math.fit(X_train_math, y_train_math)
dt_classifier_portuguese = DecisionTreeClassifier(random_state=42)
dt_classifier_portuguese.fit(X_train_portuguese, y_train_portuguese)
# KNeighborsClassifier
knn_classifier_math = KNeighborsClassifier()
knn_classifier_math.fit(X_train_math, y_train_math)
knn_classifier_portuguese = KNeighborsClassifier()
knn_classifier_portuguese.fit(X_train_portuguese, y_train_portuguese)
# LogisticRegression
lr_classifier_math = LogisticRegression(random_state=42)
lr_classifier_math.fit(X_train_math, y_train_math)
lr_classifier_portuguese = LogisticRegression(random_state=42)
lr_classifier_portuguese.fit(X_train_portuguese, y_train_portuguese)
# Make predictions
dt_pred_math = dt_classifier_math.predict(X_test_math)
dt_pred_portuguese = dt_classifier_portuguese.predict(X_test_portuguese)
knn_pred_math = knn_classifier_math.predict(X_test_math)
knn_pred_portuguese = knn_classifier_portuguese.predict(X_test_portuguese)
lr_pred_math = lr_classifier_math.predict(X_test_math)
lr_pred_portuguese = lr_classifier_portuguese.predict(X_test_portuguese)
# Calculate accuracy scores
accuracy_dt_math = accuracy_score(y_test_math, dt_pred_math)
accuracy_dt_portuguese = accuracy_score(y_test_portuguese, dt_pred_portuguese)
accuracy_knn_math = accuracy_score(y_test_math, knn_pred_math)
accuracy_knn_portuguese = accuracy_score(y_test_portuguese, knn_pred_portuguese)
accuracy_lr_math = accuracy_score(y_test_math, lr_pred_math)
accuracy_lr_portuguese = accuracy_score(y_test_portuguese, lr_pred_portuguese)
# Print accuracy scores
print("Decision Tree Classifier - Math Accuracy:", accuracy_dt_math) 
print("Decision Tree Classifier - Portuguese Accuracy:", accuracy_dt_portuguese) 
print("\n")
print("KNeighbors Classifier - Math Accuracy:", accuracy_knn_math) 
print("KNeighbors Classifier - Portuguese Accuracy:", accuracy_knn_portuguese) 
print("\n")
print("Logistic Regression - Math Accuracy:", accuracy_lr_math)
print("Logistic Regression - Portuguese Accuracy:", accuracy_lr_portuguese)
```

    Decision Tree Classifier - Math Accuracy: 0.7272727272727273
    Decision Tree Classifier - Portuguese Accuracy: 0.8961038961038961
    
    
    KNeighbors Classifier - Math Accuracy: 0.6623376623376623
    KNeighbors Classifier - Portuguese Accuracy: 0.8831168831168831
    
    
    Logistic Regression - Math Accuracy: 0.7662337662337663
    Logistic Regression - Portuguese Accuracy: 0.9090909090909091


    /Users/emandabisrat/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /Users/emandabisrat/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /Users/emandabisrat/opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    /Users/emandabisrat/opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Random Forest classifier for math scores
rf_classifier_math = RandomForestClassifier(n_estimators=50, random_state=42)

# parameter grid for grid search
param_grid = {
    'max_depth': [3, 5, 7, 9],      
    'ccp_alpha': [0.001, 0.01, 0.1] 
}


grid_search_math = GridSearchCV(estimator=rf_classifier_math, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search_math.fit(X_train_math, y_train_math)
best_params_math = grid_search_math.best_params_
print("Best Hyperparameters for Math Scores:", best_params_math)


rf_pred_math = grid_search_math.predict(X_test_math)
accuracy_rf_math = accuracy_score(y_test_math, rf_pred_math)
print("Random Forest Accuracy - Math:", accuracy_rf_math)

# repeat the process for Portuguese scores
# Random Forest classifier for Portuguese scores
rf_classifier_portuguese = RandomForestClassifier(n_estimators=50, random_state=42)
grid_search_portuguese = GridSearchCV(estimator=rf_classifier_portuguese, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search_portuguese.fit(X_train_portuguese, y_train_portuguese)

# best hyperparameters
best_params_portuguese = grid_search_portuguese.best_params_
print("Best Hyperparameters for Portuguese Scores:", best_params_portuguese)

#best model to make predictions for Portuguese scores
rf_pred_portuguese = grid_search_portuguese.predict(X_test_portuguese)

#Caccuracy for Portuguese scores
accuracy_rf_portuguese = accuracy_score(y_test_portuguese, rf_pred_portuguese)
print("Random Forest Accuracy - Portuguese:", accuracy_rf_portuguese)

```

    Best Hyperparameters for Math Scores: {'ccp_alpha': 0.001, 'max_depth': 3}
    Random Forest Accuracy - Math: 0.8181818181818182
    Best Hyperparameters for Portuguese Scores: {'ccp_alpha': 0.001, 'max_depth': 3}
    Random Forest Accuracy - Portuguese: 0.8831168831168831



```python
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score

rf_classifier_d_math = RandomForestClassifier(n_estimators=50, max_depth=3, ccp_alpha=0.001, random_state=42)

rf_classifier_d_math.fit(X_train_math, y_train_math)
rf_pred_math_d = rf_classifier_d_math.predict(X_test_math)
#accuracy for math scores
accuracy_rf_math_d = accuracy_score(y_test_math, rf_pred_math_d)
#accuracy score for Model D - Math
print ("Random Forest Model D Accuracy - Math (Increased Estimators):", accuracy_rf_math_d)

#repeat the process for Portuguese scores

rf_classifier_d_portuguese = RandomForestClassifier(n_estimators=50, max_depth=8, ccp_alpha=0.001, random_state=42)
rf_classifier_d_portuguese.fit(X_train_portuguese, y_train_portuguese)
rf_pred_portuguese_d=rf_classifier_d_portuguese.predict(X_test_portuguese)
#accuracy for Portuguese scores
accuracy_rf_portuguese_d = accuracy_score(y_test_portuguese, rf_pred_portuguese_d)
#accuracy score for Model D - Portuguese
print ("Random Forest Model D Accuracy - Portuguese (Increased Estimators):", accuracy_rf_portuguese_d)
```

    Random Forest Model D Accuracy - Math (Increased Estimators): 0.8181818181818182
    Random Forest Model D Accuracy - Portuguese (Increased Estimators): 0.9090909090909091



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

rf_classifier_d_math = RandomForestClassifier(n_estimators=50, max_depth=5, ccp_alpha=0.01, random_state=42)
rf_classifier_d_math.fit(X_train_math, y_train_math)
rf_pred_math_d = rf_classifier_d_math.predict(X_test_math)

#accuracy for math scores
accuracy_rf_math_d = accuracy_score(y_test_math, rf_pred_math_d)

#accuracy score for Model D - Math
print("Random Forest Model D Accuracy - Math (Increased Estimators):", accuracy_rf_math_d)

#confusion matrix for math scores
plot_confusion_matrix(y_test_math, rf_pred_math_d, "Confusion Matrix - Math Scores")

#repeat the process for Portuguese scores

rf_classifier_d_portuguese = RandomForestClassifier(n_estimators=50, max_depth=9, ccp_alpha=0.00, random_state=42)
rf_classifier_d_portuguese.fit(X_train_portuguese, y_train_portuguese)
rf_pred_portuguese_d = rf_classifier_d_portuguese.predict(X_test_portuguese)

#accuracy for Portuguese scores
accuracy_rf_portuguese_d = accuracy_score(y_test_portuguese, rf_pred_portuguese_d)

#accuracy score for Model D - Portuguese
print("Random Forest Model D Accuracy - Portuguese (Increased Estimators):", accuracy_rf_portuguese_d)

#confusion matrix for Portuguese scores
plot_confusion_matrix(y_test_portuguese, rf_pred_portuguese_d, "Confusion Matrix - Portuguese Scores")

```

    Random Forest Model D Accuracy - Math (Increased Estimators): 0.8051948051948052



    
![png](MachineLearningProject1_files/MachineLearningProject1_14_1.png)
    


    Random Forest Model D Accuracy - Portuguese (Increased Estimators): 0.9090909090909091



    
![png](MachineLearningProject1_files/MachineLearningProject1_14_3.png)
    



```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the multi-output neural network model
input_layer = Input(shape=(X_train.shape[1],))
shared_hidden_layer = Dense(64, activation='relu')(input_layer)

output_math = Dense(1, activation='linear', name='output_math')(shared_hidden_layer)
output_portuguese = Dense(1, activation='linear', name='output_portuguese')(shared_hidden_layer)

model = Model(inputs=input_layer, outputs=[output_math, output_portuguese])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, [y_math_train, y_portuguese_train], epochs=50, batch_size=32, validation_split=0.2)

loss, math_loss, portuguese_loss = model.evaluate(X_test, [y_math_test, y_portuguese_test])
print(f"Mean Squared Error on Test Set (Math): {math_loss}")
print(f"Mean Squared Error on Test Set (Portuguese): {portuguese_loss}")

predictions_math, predictions_portuguese = model.predict(X_test)
```


```python
student['final_grade_math']
```




    0       6
    1       6
    2      10
    3      15
    4      10
           ..
    377     8
    378     0
    379     0
    380    16
    381    10
    Name: final_grade_math, Length: 382, dtype: int64




```python
y_math
```




    0      0
    1      0
    2      1
    3      1
    4      1
          ..
    377    0
    378    0
    379    0
    380    1
    381    1
    Name: final_grade_math, Length: 382, dtype: int64




```python
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test_math,dt_pred_math ) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Fail', 'Pass']) 
cm_display.plot()
plt.title('Confusion Matrix- Math Scores')
plt.show()
```


    
![png](MachineLearningProject1_files/MachineLearningProject1_18_0.png)
    



```python
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test_portuguese,dt_pred_portuguese )
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Fail', 'Pass']) 
cm_display.plot()
plt.title('Confusion Matrix- Portuguese Scores') 
plt.show()
```


    
![png](MachineLearningProject1_files/MachineLearningProject1_19_0.png)
    



```python
from sklearn.cluster import KMeans
# Features and number of clusters
features = ['Weekly_Alc_Consumption', 'final_grade_math'] 
n_clusters = 3
# Initialize KMeans
kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=10, max_iter=300) 
kmeans.fit(student[features])

print('The lowest SSE value found: %.3f' % kmeans.inertia_)
print('The number of iterations required to converge: %d' % kmeans.n_iter_)


```

    The lowest SSE value found: 2589.591
    The number of iterations required to converge: 11



```python
features = ['Weekly_Alc_Consumption', 'final_grade_math']
n_clusters = 3
kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=10, max_iter=300)
kmeans.fit(student[features])
print('The lowest SSE value found: %.3f'%kmeans.inertia_)
print('The number of iterations required to converge: %d'%kmeans.n_iter_)
fig, ax = plt.subplots(figsize=(8, 4.5))

#plotting
ax = sns.scatterplot(ax=ax, x=student[features[0]]+np.random.rand(len(student))*0.1, y=student[features[1]]+np.random.rand(len(student))*0.2,
hue=kmeans.labels_,palette=sns.color_palette('tab10', n_colors=n_clusters), legend=None, alpha = 0.33)
for n, [dur, qual] in enumerate(kmeans.cluster_centers_): 
    ax.scatter(dur, qual, s=100, c='#a8323e') 
    ax.annotate('Cluster %d'%(n+1), (dur, qual), fontsize=18,
                color='#a8323e', xytext=(dur, qual-1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#a8323e',lw=2),
                ha='center', va='center');
ax.set_xlabel('Weekly Alcohol Consumption')
ax.set_ylabel('Final Math Grade')
ax.set_title('Affect of Weekly Alcohol Consumption on Final Math Grade')
```

    The lowest SSE value found: 2589.591
    The number of iterations required to converge: 9





    Text(0.5, 1.0, 'Affect of Weekly Alcohol Consumption on Final Math Grade')




    
![png](MachineLearningProject1_files/MachineLearningProject1_21_2.png)
    



```python

```


```python
from sklearn.cluster import KMeans
# Features and number of clusters
features = ['Weekly_Alc_Consumption', 'final_grade_port'] # Using features from the math dataset 
n_clusters = 4
# Initialize KMeans
kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=10, max_iter=300) 
kmeans.fit(student[features])
# Print results
print('The lowest SSE value found: %.3f' % kmeans.inertia_)
print('The number of iterations required to converge: %d' % kmeans.n_iter_)


```

    The lowest SSE value found: 1435.715
    The number of iterations required to converge: 9



```python
features = ['Weekly_Alc_Consumption', 'final_grade_port']
n_clusters = 4
kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=10, max_iter=300)
kmeans.fit(student[features])
print('The lowest SSE value found: %.3f'%kmeans.inertia_)
print('The number of iterations required to converge: %d'%kmeans.n_iter_)
fig, ax = plt.subplots(figsize=(8, 4.5))
# plotting
ax = sns.scatterplot(ax=ax, x=student[features[0]]+np.random.rand(len(student))*0.1, y=student[features[1]]+np.random.rand(len(student))*0.2,
hue=kmeans.labels_,palette=sns.color_palette('tab10', n_colors=n_clusters), legend=None, alpha = 0.33)
for n, [dur, qual] in enumerate(kmeans.cluster_centers_): 
    ax.scatter(dur, qual, s=100, c='#a8323e') 
    ax.annotate('Cluster %d'%(n+1), (dur, qual), fontsize=18,
                color='#a8323e', xytext=(dur, qual-1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#a8323e',lw=2),
                ha='center', va='center');
ax.set_xlabel('Weekly Alcohol Consumption')
ax.set_ylabel('Final Portuguese Grade')
ax.set_title('Affect of Weekly Alcohol Consumption on Final Portuguese Grade')
```

    The lowest SSE value found: 1435.715
    The number of iterations required to converge: 13





    Text(0.5, 1.0, 'Affect of Weekly Alcohol Consumption on Final Portuguese Grade')




    
![png](MachineLearningProject1_files/MachineLearningProject1_24_2.png)
    



```python

```
