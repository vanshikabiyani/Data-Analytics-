    

# Prepare the data
data_dict = {
    'age': [39, 50, 38, 53, 28],
    'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
    'fnlwgt': [77516, 83311, 215646, 234721, 338409],
    'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors'],
    'education-num': [13, 13, 9, 7, 13],
    'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Married-civ-spouse'],
    'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners', 'Prof-specialty'],
    'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife'],
    'race': ['White', 'White', 'White', 'Black', 'Black'],
    'sex': ['Male', 'Male', 'Male', 'Male', 'Female'],
    'capital-gain': [2174, 0, 0, 0, 0],
    'capital-loss': [0, 0, 0, 0, 0],
    'hours-per-week': [40, 13, 40, 40, 40],
    'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'Cuba'],
    'salary': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K']
}

# Create the DataFrame
data = pd.DataFrame(data_dict)

# Save the DataFrame to a CSV file
data.to_csv('data.csv', index=False)

# Load the dataset from the CSV file
data = pd.read_csv('data.csv')

# 1. How many people of each race are represented in this dataset?
race_count = data['race'].value_counts()

# 2. What is the average age of men?
average_age_men = round(data[data['sex'] == 'Male']['age'].mean(), 1)

# 3. What is the percentage of people who have a Bachelor's degree?
total_people = data.shape[0]
bachelors_count = data[data['education'] == 'Bachelors'].shape[0]
percentage_bachelors = round((bachelors_count / total_people) * 100, 1)

# 4. What percentage of people with advanced education (Bachelors, Masters, or Doctorate) make more than 50K?
advanced_education = data[data['education'].isin(['Bachelors', 'Masters', 'Doctorate'])]
percentage_advanced_education_rich = round((advanced_education[advanced_education['salary'] == '>50K'].shape[0] / advanced_education.shape[0]) * 100, 1)

# 5. What percentage of people without advanced education make more than 50K?
non_advanced_education = data[~data['education'].isin(['Bachelors', 'Masters', 'Doctorate'])]
percentage_non_advanced_education_rich = round((non_advanced_education[non_advanced_education['salary'] == '>50K'].shape[0] / non_advanced_education.shape[0]) * 100, 1)

# 6. What is the minimum number of hours a person works per week?
min_work_hours = data['hours-per-week'].min()

# 7. What percentage of the people who work the minimum number of hours per week have a salary of more than 50K?
num_min_workers = data[data['hours-per-week'] == min_work_hours].shape[0]
rich_min_workers = data[(data['hours-per-week'] == min_work_hours) & (data['salary'] == '>50K')].shape[0]
rich_percentage_min_workers = round((rich_min_workers / num_min_workers) * 100, 1)

# 8. What country has the highest percentage of people that earn >50K and what is that percentage?
country_counts = data['native-country'].value_counts()
country_rich_counts = data[data['salary'] == '>50K']['native-country'].value_counts()
highest_earning_country_percentage = round((country_rich_counts / country_counts * 100).max(), 1)
highest_earning_country = (country_rich_counts / country_counts * 100).idxmax()

# 9. Identify the most popular occupation for those who earn >50K in India.
india_occupation_counts = data[(data['native-country'] == 'India') & (data['salary'] == '>50K')]['occupation'].value_counts()
top_IN_occupation = india_occupation_counts.idxmax() if not india_occupation_counts.empty else None

# Create the function to return the results
def demographic_data_analyzer():
    return {
        'race_count': race_count,
        'average_age_men': average_age_men,
        'percentage_bachelors': percentage_bachelors,
        'percentage_advanced_education_rich': percentage_advanced_education_rich,
        'percentage_non_advanced_education_rich': percentage_non_advanced_education_rich,
        'min_work_hours': min_work_hours,
        'rich_percentage_min_workers': rich_percentage_min_workers,
        'highest_earning_country': highest_earning_country,
        'highest_earning_country_percentage': highest_earning_country_percentage,
        'top_IN_occupation': top_IN_occupation
    }

# Output the results
results = demographic_data_analyzer()
print(results)
