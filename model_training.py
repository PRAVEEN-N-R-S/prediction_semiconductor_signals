import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 🔥 AutoML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ---------------- LOAD DATA ---------------- #
data = pd.read_csv("dataset.csv")

# Fill missing values (only numeric)
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# ---------------- SPLIT ---------------- #
X = data.drop("Pass/Fail", axis=1)
y = data["Pass/Fail"]

# Keep only numeric
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.mean())

# ---------------- FEATURE SELECTION ---------------- #
print("Original Features:", X.shape[1])

# Temp model for feature importance
temp_model = RandomForestClassifier(n_estimators=100)
temp_model.fit(X, y)

importances = temp_model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Select top 20
top_features = feat_df.head(20)['Feature']
X = X[top_features]

print("Selected Top 20 Features:")
print(list(top_features))

# Save feature list
joblib.dump(top_features, "features.pkl")

# ---------------- SCALING ---------------- #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN TEST ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- 🔥 AUTOML ---------------- #

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "SVM": SVC()
}

best_model = None
best_score = 0
best_name = ""

for name, model_obj in models.items():
    model_obj.fit(X_train, y_train)
    score = model_obj.score(X_test, y_test)

    print(f"{name} Accuracy: {score}")

    if score > best_score:
        best_score = score
        best_model = model_obj
        best_name = name

print("\n🔥 Best Model:", best_name)
print("🔥 Best Accuracy:", best_score)

# ---------------- FINAL PREDICTION ---------------- #
y_pred = best_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=1))

# ---------------- SAVE ---------------- #
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Best Model, Scaler & Features saved successfully!")

# ---------------- VISUALIZATION ---------------- #

print("\nTop 10 Important Features:")
print(feat_df.head(10))

plt.figure(figsize=(10,6))
plt.barh(feat_df['Feature'][:10], feat_df['Importance'][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features")
plt.show()

import seaborn as sns

# ---------------- DASHBOARD ---------------- #

# Correlation Matrix
plt.figure(figsize=(10,8))
numeric_data = data.select_dtypes(include=[np.number])
corr = numeric_data.corr()

sns.heatmap(corr, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("static/correlation.png")
plt.close()

# Feature Distribution (example: first feature)
plt.figure()
sns.histplot(data[features[0]], kde=True)
plt.title(f"Distribution of Feature {features[0]}")
plt.savefig("static/distribution.png")
plt.close()

print("📊 Dashboard graphs saved!")