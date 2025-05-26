import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\Abhiram Vinod\Downloads\Titanic-Dataset.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
# Dropping the 'Cabin' column due to high number of missing values
df.drop(columns=['Cabin'], inplace=True)

# Filling missing values in 'Age' column with the median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Filling missing values in 'Embarked' column with the most frequent value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Verifying if there are any remaining missing values
print(df.isnull().sum())

# Encode verifying categorical variables
df= pd.get_dummies(df, columns =['Sex', 'Embarked'], drop_first=True)

# Normalize or standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

sns.pairplot(data=df[['Age','Fare']])
plt.show()

from scipy.stats import zscore
df = df[(np.abs(zscore(df[['Age', 'Fare']])) < 3).all(axis=1)]
df.to_csv('Titanic-Dataset-Processed.csv', index=False)