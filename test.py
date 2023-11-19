import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

x_train = pd.read_csv('data\Part2\mpg_X_train.csv')
y_train = pd.read_csv('data\Part2\mpg_y_train.csv')
x_test = pd.read_csv('data\Part2\mpg_X_test.csv')

print(x_train.head())
print(y_train.head())
print(x_test.head())

# 결과 값은 수험번호.csv로 저장하도록 한다.

print(x_train.info())

# 해당 데이터의 값은 8개 변수를 독립 변수로 두고 분석
# 해당 데이터는 이름을 제외한 모든 값이 실수 및 정수형이므로 별도의 데이터 타입 변경 작업 불필요

print(x_train.isna().sum())
# horsepower 값에 4개의 null 값 존재 => 삭제 또는 값 지정

x_train['horsepower'] = x_train['horsepower'].fillna(x_train['horsepower'].median())
x_test['horsepower'] = x_test['horsepower'].fillna(x_test['horsepower'].median())
# mean 또는 median을 이용해 수치형 적용
# mode를 이용해 범주형 적용

print(x_train.isna().sum())
print(x_test.isna().sum())
# null 값 0 확인

print(x_train.describe())

x_train = x_train.drop(labels="name",axis=1)
x_test = x_test.drop(labels="name",axis=1)

print(x_train.head())

from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.3)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
# x_train 과 y_train 값을 이용해 모델 학습

x_val_pred = rf.predict(x_val)

y_val_pred_probarf = rf.predict_proba(x_val)

from sklearn.metrics import roc_auc_score

print(y_val_pred_probarf)

print(roc_auc_score(y_val,y_val_pred_probarf[:,1]))

pred = rf.predict_proba(x_test)
print(pred[:,1])

pd.DataFrame({'isUSA':pred[:,1]}).to_csv('./003000000.csv',index=False)