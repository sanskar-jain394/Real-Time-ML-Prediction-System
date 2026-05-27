import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import joblib


# Dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 5],
    'Price': [30, 35, 45, 50, 60, 65, 75]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Area', 'Bedrooms']]
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'house_price_model.pkl')

print("Model Saved Successfully")