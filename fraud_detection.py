import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset of credit card
data = pd.read_csv('creditcard.csv')

# Display the first 5 rows of the dataset
print(data.head())
# Get a concise summary of the dataset
print(data.info())
print(data.isnull().sum())

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scaled_features = data.drop(columns=['Time', 'Amount'])
scaled_features = scalar.fit_transform(scaled_features)
scaled_features_df = pd.DataFrame(scaled_features, columns=[f'v{i}' for i in range(1, scaled_features.shape[1] + 1)])

# Add back 'Time', 'Amount', 'Class'
scaled_features_df['Time'] = data['Time']
scaled_features_df['Amount'] = data['Amount']
scaled_features_df['Class'] = data['Class']

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

x = scaled_features_df.drop(columns=['Class'])
y = scaled_features_df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(max_iter=1000)
model.fit(x_train_resampled, y_train_resampled)  

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5, scoring='f1')
print("Cross-validated F1 scores:", scores)


y_pred = model.predict(x_test)
print("y_test value counts:")
print(y_test.value_counts())


print("classification report:")
print(classification_report(y_test, y_pred))  

print("confusion matrix")
print(confusion_matrix(y_test, y_pred)) 

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

sns.countplot(x=y_test)
plt.title('Class Distribution in y_test')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


