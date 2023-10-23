import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the csv file
df = pd.read_csv("HAM10000_metadata.csv")
data_visual = pd.read_csv('HAM10000_metadata.csv')

print(df.head())

# select independent and dependent variable
x = df[["age", "sex"]]
y = df["localization"]

# Split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Instantiate the model
classifier = RandomForestClassifier()

# fit the model
classifier.fit(x_train, y_train)

# Make the pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))