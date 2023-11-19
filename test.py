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
