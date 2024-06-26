import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data to create medical_examination.csv
data = {
    'age': [23345, 20228, 18821, 17623, 19741],
    'height': [168, 156, 178, 165, 167],
    'weight': [62.0, 85.0, 76.0, 59.0, 65.0],
    'gender': [2, 1, 1, 2, 1],
    'ap_hi': [120, 140, 130, 150, 110],
    'ap_lo': [80, 90, 85, 100, 70],
    'cholesterol': [1, 2, 1, 3, 1],
    'gluc': [1, 1, 2, 1, 3],
    'smoke': [0, 1, 0, 1, 0],
    'alco': [0, 1, 0, 0, 1],
    'active': [1, 0, 1, 1, 0],
    'cardio': [0, 1, 0, 1, 0]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('medical_examination.csv', index=False)

# Task 1: Import the data
df = pd.read_csv('medical_examination.csv')

# Task 2: Add 'overweight' column
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (df['BMI'] > 25).astype(int)
df.drop(columns=['BMI'], inplace=True)

# Task 3: Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Task 4: Draw Categorical Plot
def draw_cat_plot():
# Task 5: Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
# Task 6: Group and reformat the data to split it by 'cardio'
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
# Task 7: Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar', height=5, aspect=1)
    
# Task 8: Get the figure for the output
    fig = g.fig

    return fig

# Task 9: Draw Heat Map
def draw_heat_map():
# Task 10: Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Task 11: Calculate the correlation matrix
    corr = df_heat.corr()

    # Task 12: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Task 13: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Task 14: Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})

    return fig

# Test the functions (you can comment these lines when running unit tests)
cat_plot = draw_cat_plot()
cat_plot.savefig('catplot.png')

heat_map = draw_heat_map()
heat_map.savefig('heatmap.png')

