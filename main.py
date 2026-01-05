import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score


data = pd.read_csv("dataset.csv")


X = data[['feature1', 'feature2']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_predictions = reg_model.predict(X_test)
print("Regression Mean Squared Error:", mean_squared_error(y_test, reg_predictions))



data['class'] = (data['target'] > data['target'].median()).astype(int)

Xc = data[['feature1', 'feature2']]
yc = data['class']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

clf_model = LogisticRegression()
clf_model.fit(Xc_train, yc_train)

clf_predictions = clf_model.predict(Xc_test)
print("Classification Accuracy:", accuracy_score(yc_test, clf_predictions))
