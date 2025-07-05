import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

train_data = pd.read_csv('data/processed/train_data_scaled.csv')

X = train_data.drop(columns=['loan_repaid'])
y = train_data['loan_repaid']

# split into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# build, compile, train & eval model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128),
    LeakyReLU(negative_slope=0.01),
    Dense(64),
    LeakyReLU(negative_slope=0.01),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

_ = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

probs = model.predict(X_val)
preds = (probs > 0.5).astype(int)
metrics = {
    'f1_score': f1_score(y_val, preds),
    'accuracy': accuracy_score(y_val, preds),
    'precision': precision_score(y_val, preds),
    'recall': recall_score(y_val, preds),
    'auc_roc': roc_auc_score(y_val, probs)
}

print("\nValidation Metrics:")
for metric, val in metrics.items():
    print(f"{metric}: {val:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, preds))

# save model
model.save('src/bin/model.h5')
print("\nModel saved successfully.")
