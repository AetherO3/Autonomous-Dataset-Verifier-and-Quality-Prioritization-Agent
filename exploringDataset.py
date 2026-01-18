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
print("The percentage of mising values in each columns is : \n\n",round(missingPerc, 10))


# Finding noise and inconsistencies and fixing the datatypes

df = df.dropna(axis = 0)

df["product_id"] = (df["product_id"].astype(str))
df["product_name"] = (df["product_name"].astype(str))
df["category"] = (df["category"].astype(str))

#   for numerical data, conversion to float
df["discounted_price"] = (df["discounted_price"].astype(str).str.replace(r"[^\d.]","", regex = True).astype(float))
df["actual_price"] = (df["actual_price"].astype(str).str.replace(r"[^\d.]","", regex = True).astype(float))
df["discount_percentage"] = (df["discount_percentage"].astype(str).str.replace(r"[^\d.]","", regex = True).astype(float))
df["rating"] = (df["rating"].astype(str).str.replace(r"[^\d.]","", regex = True).replace("", np.nan).astype(float))
df["rating_count"] = (df["rating_count"].astype(str).str.replace(r"[^\d.]","", regex = True).replace("", np.nan).astype(float))

df["about_product"] = (df["about_product"].astype(str))
df["user_id"] = (df["user_id"].astype(str))
df["user_name"] = (df["user_name"].astype(str))
df["review_id"] = (df["review_id"].astype(str))
df["review_title"] = (df["review_title"].astype(str))
df["img_link"] = (df["img_link"].astype(str))
df["product_link"] = (df["product_link"].astype(str))

# Finding the active links

sample = df.sample(25, random_state = 0)

result = []

for idx, url in sample["img_link"].items():
    try:
        res = requests.get(url, timeout = 1)
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

print(sample.dtypes)