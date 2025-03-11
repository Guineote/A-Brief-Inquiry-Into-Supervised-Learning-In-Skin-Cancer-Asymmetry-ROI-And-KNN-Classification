import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import json

categories = ['bkl', 'df', 'mel', 'vasc', 'bcc', 'nv', 'akiec']
category_counts = [1099, 115, 1113, 142, 514, 6705, 327]

data_size = sum(category_counts)

df = pd.DataFrame({
    'index': range(data_size),
    'label': np.repeat(categories, category_counts)
})


def count_categories(y):
    return pd.Series(y).value_counts().sort_index().to_dict()

print("Distribución original:", count_categories(df['label']))

for cat in categories:
    first_index = df[df['label'] == cat]['index'].iloc[0]
    print(f"Primer índice para {cat}: {first_index}")

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

print("\nDistribución en conjunto de prueba:", count_categories(test_df['label']))
print("Distribución en conjunto de entrenamiento:", count_categories(train_df['label']))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_indices_list = []
val_indices_list = []

for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df['label'])):
    train_indices = train_df.iloc[train_index]['index'].tolist()
    val_indices = train_df.iloc[val_index]['index'].tolist()
    
    train_indices_list.append(train_indices)
    val_indices_list.append(val_indices)
    
    print(f"\nFold {fold}:")
    print("Distribución en conjunto de entrenamiento:", count_categories(train_df.iloc[train_index]['label']))
    print("Distribución en conjunto de validación:", count_categories(train_df.iloc[val_index]['label']))

indices_data = {
    "train_folds": train_indices_list,
    "validation_folds": val_indices_list,
    "test_indices": test_df['index'].tolist()
}

with open('train_folds.json', 'w') as f:
    json.dump(indices_data["train_folds"], f)

with open('validation_folds.json', 'w') as f:
    json.dump(indices_data["validation_folds"], f)

with open('test_indices.json', 'w') as f:
    json.dump(indices_data["test_indices"], f)

print("\nÍndices guardados en archivos JSON.")