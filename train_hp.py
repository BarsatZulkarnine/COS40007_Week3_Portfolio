import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('combined_with_composites.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101)

param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ['rbf']}

cv = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(SVC(), param_grid, cv=cv, verbose=3)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

train_test_accuracy = best_model.score(X_test, y_test)

cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)

print("Best parameters found: ", best_params)
print(f'Train-Test Accuracy with hyperparameter tuning: {train_test_accuracy}')
print(f'5-Fold Cross-Validation Accuracy with hyperparameter tuning: {cv_scores.mean()}')
