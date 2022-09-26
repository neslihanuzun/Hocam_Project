##########################
# TEAM MEMBERS INFORMATION
##########################

# Neslihan UZUN       https://github.com/neslihanuzun
# Melike ENGİNSOY     https://github.com/melikenginsoy
# Ebra ÇEPNİ          https://github.com/ebracepni/ebracepni
# Birgül CERYAN       https://github.com/birgulceryan

#######################################################################################
# A MACHINE LEARNING MODEL TO SUPPORT THE LEARNING EXPERIENCE FOR 8th GRADE STUDENTS  #
#######################################################################################

####################
# Business Problem #
####################
"""
The high number of students causes the student course status follow-up to take a long time.
Questions such as the causes of student failures and whether the source of incorrect answers
is student-centered or classroom-based is a complex area. In this project, it is aimed to try
to support teachers' student course status tracking through machine learning algorithms.
"""

###################
#    Features     #
###################

# STUDENT_ID: Student id
# CLASS: Student's class
# TEST_NUMBER: Test number
# QUESTION_NUMBER: Question of the test
# TOPIC: Main topics
# SUB_TOPIC: Sub-topics, achievements
# TAXONOMY_CLASS: The taxonomy class of the question
# ANSWER: True-False (1-0)

########################################################################################
#                           1.EXPLORATORY DATA ANALYSIS                                #
########################################################################################

#######################################
# * 1.1.Importing necessary libraries *
#######################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#pip install xlrd
from uuid import uuid4

#import cufflinks as cf
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate , GridSearchCV

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

###########################
# * 1.2.Read the dataset *
###########################

# * Converting student name to student id *
"""
df = pd.read_excel('student_list.xlsx')

STUDENT_NAME = df['STUDENT_NAME']
STUDENT_ID = []
for student in students:
    unique_id = str(uuid4())
    STUDENT_ID.append('{} : {}'.format(student, unique_id))

df = pd.DataFrame(STUDENT_ID, columns=['STUDENT_ID'])
print(df)

submission_df = df
submission_df.to_csv("submission_std_id.csv", index=False)
"""

# * Checking the data *

df = pd.read_excel("dataset_vbofinal.xlsx")


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())


check_df(df)

# * The Missing Values Analysis *

df.dropna(inplace=True)
df = df.reset_index(drop=True)
df.isnull().sum()

# * Creating New Feature Interactions *

df["TEST_QUESTION"] = df['TEST_NUMBER'].astype(str) +"_"+ df["QUESTION_NUMBER"].astype(str)

df.head()
df.nunique()

################################################################
# * 1.3.Define Numerical and Categorical variables of dataset *
################################################################


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=True):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)

###################################
# * 1.4.Target Variable Analysis *
###################################

df["ANSWER"].value_counts()


def target_summary_with_cat(df, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean(),
                        "Count": df[categorical_col].value_counts(),
                        "Ratio": 100 * df[categorical_col].value_counts() / len(df)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "ANSWER", col)

#############################
# * 1.5.Defined functions *
#############################


def class_test_mean(dataframe, Class, test_number):
    print(df[(df["CLASS"] == Class) & (df["TEST_NUMBER"] == test_number)]["ANSWER"].mean())


class_test_mean(df, "8C", 15)
class_test_mean(df, "8B", 16)


def taxonomy(dataframe, Class, taxonomy, test_number, answer=1):
    print(df[(df["CLASS"] == Class) & (df["TAXONOMY_CLASS"] == taxonomy) &
             (df["TEST_NUMBER"] == test_number) & (df["ANSWER"] == answer)]["ANSWER"].count())


taxonomy(df, "8B", "BİLGİ", 15, answer=1)
taxonomy(df, "8C", "KAVRAMA", 14, answer=0)


df.groupby(["CLASS", "TAXONOMY_CLASS"]).agg({"ANSWER": ["sum", "count", "mean"]})

df.groupby(["CLASS", "TAXONOMY_CLASS", "STUDENT_ID"]).agg({"ANSWER": ["sum", "count", "mean"]})

df.groupby(["CLASS", "TAXONOMY_CLASS", "STUDENT_ID"]).ANSWER.\
    agg(Toplam_Soru=('sum'), Toplam_Doğru_Sayısı=('count'), Puan=('mean'))

df.groupby(["CLASS", "TAXONOMY_CLASS"]).ANSWER.agg(['mean'])


def categorize(x):
    m = x.mean()
    return True if m < 0.50 else False


df.groupby(["CLASS", "TAXONOMY_CLASS", "STUDENT_ID"]).ANSWER.agg(['sum', 'count', 'mean', categorize])

df.groupby(["CLASS", "TAXONOMY_CLASS", "STUDENT_ID", "TEST_NUMBER"]).ANSWER.\
    agg(['sum', 'count', 'mean', categorize])

df.groupby(["CLASS", "TAXONOMY_CLASS"]).ANSWER.agg(['mean'])

df.groupby(["STUDENT_ID"]).ANSWER.agg(['mean']).sort_values(by=['mean'], ascending=True).head(10)

df.groupby(["STUDENT_ID"]).ANSWER.agg(['mean']).sort_values(by=['mean'], ascending=False).head(10)

df["TAXONOMY_CLASS"].value_counts()

df.groupby(["TAXONOMY_CLASS", "CLASS", "STUDENT_ID"]).ANSWER.agg(['sum'])

df.groupby(["STUDENT_ID", "CLASS", "TEST_QUESTION", "TAXONOMY_CLASS"]).\
    agg({"ANSWER": "mean"}).sort_values("TEST_QUESTION", ascending=False)

df.groupby(["STUDENT_ID", "SUB_TOPIC", "TAXONOMY_CLASS"]).agg({"ANSWER": "sum"}).\
    sort_values("STUDENT_ID", ascending=False)

df.groupby(["CLASS", "SUB_TOPIC", "TOPIC"]).agg({"ANSWER": "mean"})

df.groupby(["CLASS", "SUB_TOPIC", "TOPIC", "TEST_QUESTION"]).agg({"ANSWER": "mean"}).\
    sort_values("TEST_QUESTION", ascending=False)

#############################
# * 1.6.Data visualization *
#############################

cat_feat_constraints = ['#7FB3D5', '#76D7C4', '#F7DC6F', '#85929E', '#283747',
                        "#da70d6", "#ffa07a", "#ffc0cb", "#d3d3d3", "#40e0d0",
                        "#bc8f8f", "#e9967a", "#ffb6c1", "#9370db", "#c6fcfb",
                        "#88b378", "#8c0034", "#f0944d", "#aefd6c", "#fd5956",
                        "#eecffe", "#ffab0f", "#b6ffbb", "#cf6275", "#02ccfe",
                        "#82cafc", "#94ac02", "#d3b683"]


def categorical_variable_summary(df, column_name):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Countplot', 'Percentages'),
                        specs=[[{"type": "xy"}, {'type': 'domain'}]])

    fig.add_trace(go.Bar(y=df[column_name].value_counts().values.tolist(),
                         x=[str(i) for i in df[column_name].value_counts().index],
                         text=df[column_name].value_counts().values.tolist(),
                         textfont=dict(size=15),
                         name=column_name,
                         textposition='auto',
                         showlegend=False,
                         marker=dict(color=cat_feat_constraints,
                                     line=dict(color='#DBE6EC',
                                               width=1))),
                  row=1, col=1)

    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(),
                         values=df[column_name].value_counts().values,
                         textfont=dict(size=20),
                         textposition='auto',
                         showlegend=False,
                         name=column_name,
                         marker=dict(colors=cat_feat_constraints)),
                  row=1, col=2)

    fig.update_layout(title={'text': column_name,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')

    iplot(fig)


categorical_variable_summary(df, 'CLASS')


def summary_cat_features(dataframe, column_name, label):
    data = go.Bar(x=dataframe.groupby(column_name).agg({label: 'mean'}).reset_index()[column_name],
                  y=dataframe.groupby(column_name).agg({label: 'mean'}).reset_index()[label],
                  text=round(dataframe.groupby(column_name).agg({label: 'mean'}).reset_index()[label], 3),
                  textposition='auto',
                  marker=dict(color=cat_feat_constraints, line_color='white', line_width=1.5))

    layt = go.Layout(title={'text': f'Average {label} by {column_name} Categories', 'y': 0.9, 'x': 0.2,
                            'xanchor': 'center', 'yanchor': 'top'},
                     xaxis=dict(title=column_name),
                     yaxis=dict(title=label),
                     template='plotly_white')

    fig = go.Figure(data=data, layout=layt)
    iplot(fig)


new_cat_fea = ["CLASS", "TAXONOMY_CLASS", 'TOPIC', 'SUB_TOPIC', 'TEST_NUMBER']
for i in new_cat_fea:
    categorical_variable_summary(df, i)
    summary_cat_features(df, i, 'ANSWER')


x = [0.816, 0.866, 0.865, 0.789, 0.863]
labels = ['ANALİZ', 'BİLGİ', 'KAVRAMA', 'SENTEZ', 'UYGULAMA']

fig, ax = plt.subplots()
ax.pie(x, labels = labels, autopct='%.2f%%')
ax.set_title('The average of correct answers to the question by taxonomy class')
plt.show()


########################################################################################
#                           2.FEATURE ENGINEERING                                      #
########################################################################################

##################################################
# * 2.1.Processing Encoding and One-Hot Encoding *
##################################################

df.drop(['TEST_NUMBER'], axis=1, inplace=True)
df.drop(["QUESTION_NUMBER"], axis=1, inplace=True)
df.head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = ['TOPIC', 'TAXONOMY_CLASS', 'SUB_TOPIC', 'CLASS', 'TEST_QUESTION', "STUDENT_ID"]
df = one_hot_encoder(df, ohe_cols)
df.head()

########################################################################################
#                           3.MODELLING                                                #
########################################################################################

###########################################
# * 3.1.Create Modelling and Prediction *
###########################################

y = df["ANSWER"]
X = df.drop(["ANSWER"], axis=1)

################################
# * 3.1.1. Logistic Regression *
################################

log_model = LogisticRegression().fit(X, y)

log_model.intercept_

log_model.coef_

y_pred = log_model.predict(X)

# * Model Evaluation *


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

# * Model Validation: Holdout *

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

# * Model Validation: 5-Fold Cross Validation *

y = df["ANSWER"]
X = df.drop(["ANSWER"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_precision'].mean()

cv_results['test_recall'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

#####################################
# * 3.1.2. Classification ML Models *
#####################################

# * Model Validation: Holdout *

primitive_success = []
model_names = []
y = df['ANSWER']
X = df.drop(["ANSWER"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


def ML(algName):

    # Model Building / Training
    model = algName().fit(X_train,y_train)
    model_name = algName.__name__
    model_names.append(model_name)
    # Prediction
    y_pred = model.predict(X_test)
    # primitive-Success / Verification Score
    from sklearn.metrics import accuracy_score
    primitiveSuccess = accuracy_score(y_test, y_pred)
    primitive_success.append(primitiveSuccess)
    return primitive_success, model_names, model


models = [KNeighborsClassifier, SVC, MLPClassifier, DecisionTreeClassifier,RandomForestClassifier,
          GradientBoostingClassifier, XGBClassifier, LGBMClassifier]

for i in models:
    ML(i)

classification = pd.DataFrame(primitive_success, columns=['accuracy_Score'], index=[model_names]).\
    sort_values(by='accuracy_Score', ascending=False)

classification

########
# LGBM
########

# * Model Validation: 5-Fold Cross Validation *

lgbm_model = LGBMClassifier(random_state=17)

cv_results = cross_validate(lgbm_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

# * Model Optimization *

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

########
# GBM
########

# * Model Validation: 5-Fold Cross Validation *

gbm_model = GradientBoostingClassifier(random_state=17)

cv_results = cross_validate(gbm_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

# * Model Optimization *

gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "min_samples_split": [2, 5, 8],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

################################
# * 3.1.3. Feature İmportance *
################################


def plot_importance(model, features, num=len(X), save=True):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_final, X, 30)

plot_importance(gbm_final, X, 30)

###########################################
# * 3.2.Prediction for A New Observation *
###########################################

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)

random_student = X.sample(5, random_state=12)
log_model.predict(random_student)

random_student = X.sample(5, random_state=4)
log_model.predict(random_student)

random_student = X.sample(5, random_state=20)
log_model.predict(random_student)
