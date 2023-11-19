import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data\Train.csv')

print(df.head())