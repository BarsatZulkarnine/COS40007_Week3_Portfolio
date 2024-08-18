import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('processed_features.csv')
X = data.drop(columns=['class'])
y = data['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

classifiers = {
    'SVM': SVC(),
    'SGD': SGDClassifier(),
    'RandomForest': RandomForestClassifier(),
    'MLP': MLPClassifier()
}


for name, clf in classifiers.items():
    print(f"Training {name}...")
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(f"{name} Cross-Validation Accuracy: {scores.mean()}")


# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=10)
grid.fit(X_train, y_train)
print("Best parameters for SVM:", grid.best_params_)

# Feature selection
k_best = SelectKBest(f_classif, k=10)
X_train_best = k_best.fit_transform(X_train, y_train)
X_test_best = k_best.transform(X_test)

# PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train on selected features and PCA
print("Training SVM on selected features and PCA...")
clf_best = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'])
clf_best.fit(X_train_best, y_train)
clf_pca = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'])
clf_pca.fit(X_train_pca, y_train)

# Evaluate on test set
print(f"SVM Test Accuracy on selected features: {clf_best.score(X_test_best, y_test)}")
print(f"SVM Test Accuracy on PCA components: {clf_pca.score(X_test_pca, y_test)}")
