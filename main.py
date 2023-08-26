from pickle import NONE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('sonar_data.csv', header=None)
# Separate input and output
X = data.drop(columns=60)
y = data[60]

# Change random state to see the variation in the accuracy
rs = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=rs)
model = LogisticRegression()
model.fit(X_test, y_test)
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, y_test)
print(f'Accuracy = {testing_data_accuracy * 100}%')
# Input any sample data to test the trained model
input_data = (
    0.0262, 0.0582, 0.1099, 0.1083, 0.0974, 0.2280, 0.2431, 0.3771, 0.5598, 0.6194, 0.6333, 0.7060, 0.5544, 0.5320,
    0.6479,
    0.6931, 0.6759, 0.7551, 0.8929, 0.8619, 0.7974, 0.6737, 0.4293, 0.3648, 0.5331, 0.2413, 0.5070, 0.8533, 0.6036,
    0.8514,
    0.8512, 0.5045, 0.1862, 0.2709, 0.4232, 0.3043, 0.6116, 0.6756, 0.5375, 0.4719, 0.4647, 0.2587, 0.2129, 0.2222,
    0.2111,
    0.0176, 0.1348, 0.0744, 0.0130, 0.0106, 0.0033, 0.0232, 0.0166, 0.0095, 0.0180, 0.0244, 0.0316, 0.0164, 0.0095,
    0.0078)

input_data_as_numpy_array = np.asanyarray(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshape)
if (prediction[0] == 'R'):
    print('The object is a ROCK')
else:
    print('The object is a MINE')
