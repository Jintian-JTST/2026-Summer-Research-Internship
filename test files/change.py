import pandas as pd

df = pd.read_csv("Data copy.csv")

print("Before:")
print(df["Time_us"].head())

df["Time_us"] = (df["Time_us"] * 1e-6).round(2)

print("After:")
print(df["Time_us"].head())

df.to_csv("Data_NEW.csv", index=False)