import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset (taruh Depresi.csv di folder yang sama)
df = pd.read_csv("Depresi.csv")

# bersihkan target
df = df.dropna(subset=['Depression'])

# fitur & target
X = df.drop('Depression', axis=1)
y = df['Depression']

# encoding
X = pd.get_dummies(X, drop_first=True)

# simpan kolom
columns = X.columns.tolist()

# handle missing
X = X.fillna(X.mean())

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(columns, "columns.pkl")

print("MODEL BERHASIL DISIMPAN")