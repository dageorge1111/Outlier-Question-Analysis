import pandas as pd

# Specify an alternative encoding
data = pd.read_csv("voirdiredata.csv", encoding="ISO-8859-1")

# Filter rows where 'q_type' ends with '2Jr'
filtered_data = data[data['q_type'].str.endswith("2Jr", na=False)]

# Display the filtered data
print(filtered_data)

# Optionally, save the filtered data to a new CSV file
filtered_data.to_csv("filtered_data.csv", index=False)

