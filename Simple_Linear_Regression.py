import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

veri = {
    "Reklam Harcamaları": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "Satışlar": [200, 220, 240, 260, 280, 300, 320, 340 ,360, 380]
}

df = pd.DataFrame(veri)

X = df[["Reklam Harcamaları"]].values
y = df["Satışlar"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.scatter(X, y, color = "blue", label = "Gerçek Veriler")
plt.plot(X, model.predict(X), color = "red", label = "Tahmin (Regresyon Doğrusu)")
plt.title("Reklam Harcalamaları İle Satışlar Arasındaki İlişki")
plt.xlabel("Reklam Harcamaları")
plt.ylabel("Satışlar")
plt.legend()
plt.show()
