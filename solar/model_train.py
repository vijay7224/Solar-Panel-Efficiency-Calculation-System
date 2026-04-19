import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset

data1=pd.read_csv(r"E:\Plant_1_Generation_Data.csv")
data2=pd.read_csv(r"E:\Plant_1_Weather_Sensor_Data.csv")
data1["DATE_TIME"] = pd.to_datetime(data1["DATE_TIME"])
data2["DATE_TIME"] = pd.to_datetime(data2["DATE_TIME"])

# merge
data = pd.merge(data1, data2, on=["DATE_TIME", "PLANT_ID"])


# Fix zero irradiation

data["EFFICIENCY"] = (
    data["AC_POWER"] / data["DC_POWER"].replace(0, np.nan)
) * 100


# Remove invalid values

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Features & target
X = data[[
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "DC_POWER",
]]

y = data["EFFICIENCY"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
from joblib import dump

dump(model, "model.pkl", compress=5)

print("Model trained and saved as model.pkl")