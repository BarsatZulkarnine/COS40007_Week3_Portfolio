import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('combined_with_composites.csv')
X = data.drop(columns=['class'])
y = data['class']

# PCA: Reduce to 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Train-Test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Initialize the model with best hyperparameters
best_model = SVC(C=0.1, gamma=0.1, kernel='rbf')

# Train on PCA components
best_model.fit(X_train, y_train)
train_test_accuracy = best_model.score(X_test, y_test)
print(f'Train-Test Accuracy with PCA: {train_test_accuracy}')

# 10-Fold Cross-Validation
cv_scores = cross_val_score(best_model, X_pca, y, cv=10)
print(f'10-Fold Cross-Validation Accuracy with PCA: {cv_scores.mean()}')