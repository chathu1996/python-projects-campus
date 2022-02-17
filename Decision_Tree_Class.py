import pandas as pd;
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
# print(dataset)

x = dataset[["Age", "EstimatedSalary"]]
y = dataset["Purchased"]

# print(x.shape)
# print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

New_data = MinMaxScaler()
normalize_data =New_data.fit(x_train)

x_train_normalized = normalize_data.transform(x_train)
x_test_normalized = normalize_data.transform(x_test)

model = DecisionTreeClassifier()
model_dt = model.fit(x_train_normalized, y_train)

y_predict = model.predict(x_test_normalized)

accuracy = model.score(x_test_normalized, y_test)
# print(accuracy)

plt.scatter(x_test[y_test==0]["Age"], x_test[y_test==0]["EstimatedSalary"], c="red", alpha=0.5)
plt.scatter(x_test[y_test==1]["Age"], x_test[y_test==1]["EstimatedSalary"], c="blue", alpha=0.5)

plt.show()