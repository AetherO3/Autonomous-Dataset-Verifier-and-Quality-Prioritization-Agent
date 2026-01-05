import pandas as pd
import numpy as np


# link to another dataset ** ---- https://amazon-reviews-2023.github.io/#quick-start ---- **
df = pd.read_csv('amazon.csv') # Kaggle Dataset

print(df.shape, "\n")
# 1465 X 16

print(df.head(), "\n")
print(df.info())


# Finding the missing parts of the dataset

missingCount = df.isna().sum()
print(missingCount, "\n")

missingPerc = df.isna().mean() * 100
print("The percentage of mising values in each columns is : \n\n",round(missingPerc, 2))


# Finding noise and inconsistencies

df.dropna(axis = 0)

df['discount_percentage'] = (df["discount_percentage"].astype(str).str.replace(r"[^\d.]","", regex = True).astype(float))

print(df['discount_percentage'].mean())