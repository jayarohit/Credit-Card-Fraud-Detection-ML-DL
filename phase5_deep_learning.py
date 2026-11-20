import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

df = pd.read_csv("data/processed/cleaned_creditcard.csv")

X = df.drop("Class",axis = 1)
Y = df["Class"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
# Separate validation data from training data
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train,
    Y_train,
    test_size=0.2,
    random_state=42,
    stratify=Y_train
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
smote = SMOTE(random_state=42)

X_train_smote, Y_train_smote = smote.fit_resample(
    X_train,
    Y_train
)

model = Sequential()
set_random_seed(42)

model.add(
    Dense(
        units = 67, 
        activation = "relu",
        input_shape =(30,)
    )
)
model.add(
    Dense(
        units=1,
        activation="sigmoid"
    )
)

model.summary()


model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)


history = model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, Y_val)
)

Y_pred_prob = model.predict(X_test).ravel()
Y_pred = (Y_pred_prob >= 0.5).astype(int)



cm = confusion_matrix(Y_test,Y_pred)
print(cm)

accuracy = accuracy_score(Y_test,Y_pred)
precision = precision_score(Y_test,Y_pred)
recall = recall_score(Y_test,Y_pred)
f1 = f1_score(Y_test,Y_pred)

roc_auc = roc_auc_score(Y_test,Y_pred_prob)
pr_auc = average_precision_score(Y_test,Y_pred_prob)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
