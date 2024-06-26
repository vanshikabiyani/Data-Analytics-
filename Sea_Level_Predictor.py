import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Define the years range
years = np.arange(1880, 2021)

# Generate synthetic sea level data
np.random.seed(0)
sea_levels = np.random.uniform(low=-2, high=10, size=len(years))

# Create a DataFrame
data = {
    'Year': years,
    'CSIRO Adjusted Sea Level': sea_levels
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('epa-sea-level.csv', index=False)

