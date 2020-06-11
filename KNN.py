import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd

data = pd.read_csv("corona_latest.csv", delimiter=',', names=['number', 'Country', 'Other', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths', 'TotalRecovered', 'ActiveCases', 'Serious', 'Critical', 'Tot Cases/1M pop', 'Deaths/1M pop'])
# print(data.head())

# Give Each label a numeric value so it can be worked as a classifier.
label = preprocessing.LabelEncoder()

number = label.fit_transform(list(data["number"]))
country = label.fit_transform(list(data["Country"]))
other = label.fit_transform(list(data["Other"]))
totalC = label.fit_transform(list(data["TotalCases"]))
newC = label.fit_transform(list(data["NewCases"]))
totalD = label.fit_transform(list(data["TotalDeaths"]))
newD = label.fit_transform(list(data["NewDeaths"]))
totalR = label.fit_transform(list(data["TotalRecovered"]))
active = label.fit_transform(list(data["ActiveCases"]))
serious = label.fit_transform(list(data["Serious"]))
critical = label.fit_transform(list(data["Critical"]))
totalperM = label.fit_transform(list(data["Tot Cases/1M pop"]))
deathperM = label.fit_transform(list(data["Deaths/1M pop"]))

predict = "class"

# X for Features & Y for Labels.
X = list(zip(number, other, totalC, newC, totalD, newD, totalR, active, serious, critical, totalperM, deathperM))
Y = list(country)

# Training & Testing Values and we used 0.1 to minimize the sacrifice of the data.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Begin Testing data.
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
predicted = model.predict(x_test)
x_train = model.transform(x_train)
x_test = model.transform(x_test)
# Get the predictions and the distances between data.
for x in range(len(predicted)):
    print("Country: ", number[x])
    print("Predicted Number: ", predicted[x], "Dataset Number: ", x_test[x], "Actual Number: ", y_test[x])
    distances = model.kneighbors([x_test[x]], 9, True)
    print("Distances between data: ", distances)

