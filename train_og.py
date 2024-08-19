import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('combined_dataset.csv')
X = data.drop(columns=['class'])
y = data['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

classifiers = {
    'SGD': SGDClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42)
}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    
    clf.fit(X_train, y_train)
    train_test_accuracy = clf.score(X_test, y_test)
    print(f"{name} Train-Test Split Accuracy: {train_test_accuracy}")
    
    cross_val_scores = cross_val_score(clf, X_scaled, y, cv=10)
    print(f"{name} 10-Fold Cross-Validation Accuracy: {cross_val_scores.mean()}")

