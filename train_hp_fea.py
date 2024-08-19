import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('combined_with_composites.csv')
X = data.drop(columns=['class'])
y = data['class']

selector = SelectKBest(f_classif, k=10)
X_best = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.3, random_state=42)

best_model = SVC(C=0.1, gamma=0.1, kernel='rbf')

best_model.fit(X_train, y_train)
train_test_accuracy = best_model.score(X_test, y_test)
print(f'Train-Test Accuracy with feature selection and hyperparameter tuning: {train_test_accuracy}')

cv_scores = cross_val_score(best_model, X_best, y, cv=10)
print(f'10-Fold Cross-Validation Accuracy with feature selection and hyperparameter tuning: {cv_scores.mean()}')