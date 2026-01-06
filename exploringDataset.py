import requests 
import pandas as pd
import numpy as np


# link to another dataset ** ---- https://amazon-reviews-2023.github.io/#quick-start ---- **
# dataset used ** ---- https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset ---- **
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

# Finding the active links

sample = df.sample(25, random_state = 0)

result = []

for idx, url in df["img_link"].items():
    try:
        res = requests.get(url, timeout = .1)
        stat = (res.status_code == 200) and ("image" in res.headers.get("Content-Type", ""))
        result.append((idx, int(stat)))
        print(idx , "\n")
    
    except:
        result.append((idx, 0))

df["img_available"] = 0
for idx, stat in result:
    df.loc[idx, "img_available"] = stat

success = sum(stat for _, stat in result)
print(f"{success} of 25 images were accessible.")
