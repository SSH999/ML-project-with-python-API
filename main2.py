import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


file_path = 'data.csv'
df = pd.read_csv(file_path)


X = df.drop(['id', 'diagnosis'], axis=1) 
y = df['diagnosis']

# Encode labels (Malignant as 1, Benign as 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Handle missing values by imputing with mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)


y_pred = rf_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


target_names = ['Benign', 'Malignant']
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
