import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = 'C:\\Users\\Annora Sundararajan\\PycharmProjects\\pythonProject1\\sallen_key_data.csv'
df = pd.read_csv(file_path)
channel_1 = df['channel 1']
channel_2 = df['channel 2']
frequency = df['frequency']
df['log_frequency'] = np.log(df['frequency'])
df['z'] = -20 * np.log10(channel_1 / channel_2)

# Subset the data to include only frequencies between 100 and 1000 Hz
subset_df = df[(df['log_frequency'] >= 5.5) & (df['log_frequency'] <= 8)]

# Define the linear model function
def linear_model(log_frequency, a, b):
    return a * log_frequency + b

# Fit the model to the subset data
popt, pcov = curve_fit(linear_model, subset_df['log_frequency'], subset_df['z'])

# Extract the gradient (slope) from the fitting parameters
gradient = popt[0]

# Generate a smooth line over the frequency range for the fitted curve
frequency_range = np.linspace(5.5, 8, 10000)
fitted_z = linear_model(frequency_range, *popt)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['log_frequency'], df['z'], label='Log Frequency vs z', color='dodgerblue', marker='.')

# Plot the fitted linear line within the range 100 to 1000 Hz
ax.plot(frequency_range, fitted_z, label=f'Fitted Linear Line log(5.5-8)kHz\nGradient = {gradient:.4f}', color='red', linestyle='--')

ax.set_xlabel('Log Frequency (kHz)', fontsize=12)
ax.set_ylabel('z (dB)', fontsize=12, labelpad=20)
ax.set_title('Sallen-Key Filter Response with Fitted Linear Line log(5.5-8)kHz', fontsize=14)
ax.grid(True, linestyle='-', alpha=0.5)
ax.legend()
plt.tight_layout()
fig.subplots_adjust(left=0.15)
plt.savefig(f"plots\\Sallen-Key Filter Response with Fitted Linear Line log(5.5-8)kHz.png", dpi=1200)
plt.show()
