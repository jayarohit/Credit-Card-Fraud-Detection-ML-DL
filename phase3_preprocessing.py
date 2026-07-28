import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("data/processed/cleaned_creditcard.csv")


X = df.drop("Class",axis = 1)
Y = df["Class"]


print("Feature Shape:",X.shape)
print("Target Shape:",Y.shape)


print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(Y.head())


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(type(X_train))
print(type(X_test))
print(X_train.shape)
print(X_test.shape)


smote = SMOTE(random_state=42)
X_train,Y_train = smote.fit_resample(X_train,Y_train)

print(X_train.shape)
print(Y_train.shape)
print(Y_train.value_counts())


model = LogisticRegression(random_state=42)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print(Y_pred[:20])


cm = confusion_matrix(Y_test,Y_pred)
print(cm)


print("Accuracy:", accuracy_score(Y_test,Y_pred))
print("Precision:", precision_score(Y_test,Y_pred))
print("Recall:", recall_score(Y_test,Y_pred))
print("F1 Score:", f1_score(Y_test,Y_pred))
print("ROC-AUC:", roc_auc_score(Y_test,Y_pred))


dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10)
dt_model.fit(X_train,Y_train)
Y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree")
print(confusion_matrix(Y_test,Y_pred_dt))
print("Accuracy :", accuracy_score(Y_test, Y_pred_dt))
print("Precision:", precision_score(Y_test, Y_pred_dt))
print("Recall   :", recall_score(Y_test, Y_pred_dt))
print("F1 Score :", f1_score(Y_test, Y_pred_dt))
print("ROC-AUC  :", roc_auc_score(Y_test, Y_pred_dt))


rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=30,
    max_depth=10,
    n_jobs=-1
)
rf_model.fit(X_train,Y_train)
Y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest")
print(confusion_matrix(Y_test,Y_pred_rf))
print("Accuracy :", accuracy_score(Y_test, Y_pred_rf))
print("Precision:", precision_score(Y_test, Y_pred_rf))
print("Recall   :", recall_score(Y_test, Y_pred_rf))
print("F1 Score :", f1_score(Y_test, Y_pred_rf))
print("ROC-AUC  :", roc_auc_score(Y_test, Y_pred_rf))
