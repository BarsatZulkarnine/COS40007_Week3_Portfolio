import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('combined_with_composites.csv')
X = data.drop(columns=['class'])
y = data['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

svm_model = SVC(kernel='rbf')

svm_model.fit(X_train, y_train)
train_test_accuracy = svm_model.score(X_test, y_test)
print(f"Default SVM Train-Test Accuracy: {train_test_accuracy}")

cv_scores = cross_val_score(svm_model, X_scaled, y, cv=10)
print(f"Default SVM 10-Fold Cross-Validation Accuracy: {cv_scores.mean()}")
