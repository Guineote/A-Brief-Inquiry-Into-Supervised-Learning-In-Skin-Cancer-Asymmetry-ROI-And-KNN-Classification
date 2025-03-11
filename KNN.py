import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import json
import matplotlib.pyplot as plt
import seaborn as sns



file_path = 'HAM10000_metadata.csv'
df = pd.read_csv(file_path)
rgb_data = np.load('histogram_vectors_pca.npy') 
y = df['dx'] 
X = df.drop(columns=['dx'])
le = LabelEncoder()
y = le.fit_transform(y)

for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = LabelEncoder().fit_transform(X[column].astype(str))
    else:
        X[column] = pd.to_numeric(X[column], errors='coerce')


imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_combined = np.hstack((X, rgb_data))
with open('train_folds.json', 'r') as f:
    train_folds = json.load(f)

with open('validation_folds.json', 'r') as f:
    validation_folds = json.load(f)

with open('test_indices.json', 'r') as f:
    test_indices = json.load(f)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
knn = KNeighborsClassifier(n_neighbors=3)
validation_accuracies = []
test_accuracies = []
validation_f1_scores = []
test_f1_scores = []
all_y_true = []
all_y_pred = []

for fold, (train_indices, val_indices) in enumerate(zip(train_folds, validation_folds)):
    X_train, X_val = X_scaled[train_indices], X_scaled[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    knn.fit(X_train, y_train)
    val_pred = knn.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    validation_accuracies.append(val_accuracy)
    validation_f1_scores.append(val_f1)
    X_test = X_scaled[test_indices]
    y_test = y[test_indices]
    test_pred = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    test_accuracies.append(test_accuracy)
    test_f1_scores.append(test_f1)
    
    if fold == len(train_folds) - 1:
        all_y_true.extend(y_test)
        all_y_pred.extend(test_pred)

print("Resultados por split:")
for i in range(5):
    print(f"\nSplit {i+1}:")
    print(f"  Validation Accuracy: {validation_accuracies[i]:.4f}")
    print(f"  Validation F1-score: {validation_f1_scores[i]:.4f}")
    print(f"  Test Accuracy: {test_accuracies[i]:.4f}")
    print(f"  Test F1-score: {test_f1_scores[i]:.4f}")


print("\nPromedios:")
print(f"Validation Accuracy: {np.mean(validation_accuracies):.4f} ± {np.std(validation_accuracies):.4f}")
print(f"Validation F1-score: {np.mean(validation_f1_scores):.4f} ± {np.std(validation_f1_scores):.4f}")
print(f"Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
print(f"Test F1-score: {np.mean(test_f1_scores):.4f} ± {np.std(test_f1_scores):.4f}")
        
f1 = f1_score(all_y_true, all_y_pred, average='weighted')
precision = precision_score(all_y_true, all_y_pred, average='weighted')
recall = recall_score(all_y_true, all_y_pred, average='weighted')

print(f"\nMétricas adicionales en el conjunto de prueba:")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nInforme de clasificación detallado:")
print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy por Fold')
plt.legend()
cm = confusion_matrix(all_y_true, all_y_pred, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusión Normalizada')
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'F1-score', 'Precision', 'Recall']
test_mean = np.mean(test_accuracies)
values = [test_mean, f1, precision, recall]
plt.bar(metrics, values)
plt.ylabel('Score')
plt.title('Métricas de Rendimiento del Modelo')

for i, v in enumerate(values):
    plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()