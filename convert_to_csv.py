import pickle
import csv


pickle_path = "./outputs/exp_results.pkl"

with open(pickle_path, 'rb') as f:
    analysis = pickle.load(f)

# Prepare the data for CSV
csv_data = []
csv_data.append(['set_number', 'timestep', 'mse', 'ssim'])  # Header row

# Iterate through the nested dictionary to extract the data
for set_number, timesteps in analysis.items():
    for timestep, values in timesteps.items():
        # Extract metrics (mse and ssim) from the 'metrics' key
        mse, ssim = values['metrics']
        
        # Append the data to the list
        csv_data.append([set_number, timestep, mse, ssim])

# Save the data to a CSV file
with open('./outputs/exp_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print("CSV file 'exp_results.csv' has been created.")
