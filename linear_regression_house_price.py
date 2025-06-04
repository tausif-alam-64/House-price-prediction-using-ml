import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    "Area": [1000, 1500, 2000, 2500],
    "Price": [20000, 25000, 30000, 35000]
}
df = pd.DataFrame(data)

x = df[["Area"]] # keeping as 2d, scikit-learn expects for features
y = df["Price"] # keeping as 1d, representing the target value

model = LinearRegression() 
model.fit(x,y)  # Train model using our dataset

y_pred = model.predict(x)
accuracy = r2_score(y, y_pred)
print(f"Model R² Score (Accuracy): {accuracy:.4f}")

area_range = np.linspace(df['Area'].min(), df['Area'].max(), 100).reshape(-1,1)
price_pridictions = model.predict(area_range)

plt.figure(figsize=(8, 5))
plt.scatter(df["Area"], df["Price"], color="blue", label='Actual Data')

plt.plot(area_range, price_pridictions, color="red", linewidth=2, label="Regression Line")
predicted_price = model.predict([[1700]])
plt.scatter(1800, predicted_price, color="green", s=100, label=f'Prediction(1800 sq ft: ${float(predicted_price[0]):,.0f})')

plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Price Prediction with Linear Regression")
plt.legend()
plt.grid(True)
plt.show()