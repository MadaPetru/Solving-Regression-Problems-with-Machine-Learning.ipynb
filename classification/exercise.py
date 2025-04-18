# Importarea librăriilor necesare
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Încărcarea dataset-ului
df = pd.read_csv('diabetes.csv')  # Asigură-te că fișierul diabetes.csv este în același folder

# Afișarea primelor rânduri
print("Primele 5 rânduri din dataset:")
print(df.head())

# Verificarea dimensiunii și a valorilor lipsă
print("\nDimensiunea dataset-ului:", df.shape)
print("\nValori lipsă inițiale:\n", df.isnull().sum())

# Coloane cu valori 0 imposibile
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Verificare după înlocuire
print("\nValori lipsă după curățare:\n", df.isnull().sum())

# Separarea features și target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Împărțirea în set de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Antrenarea modelului (Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predicții
y_pred = clf.predict(X_test_scaled)

# Evaluarea performanței
print("\nAcuratețea modelului:", accuracy_score(y_test, y_pred))
print("\nRaport de clasificare:\n", classification_report(y_test, y_pred))
print("\nMatrice de confuzie:\n", confusion_matrix(y_test, y_pred))

# Graficul matricei de confuzie
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confuzie')
plt.xlabel('Predicții')
plt.ylabel('Valori reale')
plt.show()

#pip install pandas numpy matplotlib seaborn scikit-learn
#pip install fsspec
