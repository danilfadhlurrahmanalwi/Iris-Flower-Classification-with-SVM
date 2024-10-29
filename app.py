# DataFlair Iris Classification on Streamlit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Streamlit Title
st.title("Iris Flower Classification with SVM")

# Load the data
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
df = pd.read_csv('iris.data', names=columns)

# Display data
st.subheader("Dataset")
st.write(df.head())

# Basic statistical analysis
st.subheader("Statistical Summary")
st.write(df.describe())

# Visualize the whole dataset
st.subheader("Data Visualization")
sns.pairplot(df, hue='Class_labels')
st.pyplot(plt)

# Separate features and target
data = df.values
X = data[:, 0:4]
Y = data[:, 4]

# Calculate average of each feature for all classes
Y_Data = np.array([np.average(X[:, i][Y == j].astype('float32')) for i in range(X.shape[1]) for j in np.unique(Y)])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns) - 1)
width = 0.25

# Plot the average
fig, ax = plt.subplots()
ax.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa')
ax.bar(X_axis + width, Y_Data_reshaped[1], width, label='Versicolor')
ax.bar(X_axis + width * 2, Y_Data_reshaped[2], width, label='Virginica')
ax.set_xticks(X_axis)
ax.set_xticklabels(columns[:4])
ax.set_xlabel("Features")
ax.set_ylabel("Value in cm.")
ax.legend(bbox_to_anchor=(1.3, 1))
st.pyplot(fig)

# Split the data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support Vector Machine model
svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset and calculate accuracy
predictions = svn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# Classification report
st.subheader("Classification Report")
report = classification_report(y_test, predictions, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Model Prediction
st.subheader("Predict New Samples")
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
prediction = svn.predict(X_new)
st.write("Prediction of Species for New Samples:")
st.write(prediction)

# Save the model
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)

# Load and use the model
st.subheader("Loaded Model Prediction")
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)
loaded_prediction = model.predict(X_new)
st.write("Loaded Model Predictions:")
st.write(loaded_prediction)
