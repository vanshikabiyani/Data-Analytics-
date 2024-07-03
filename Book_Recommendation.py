Book Recommendation Engine using KNN


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Mock data creation (for demonstration)
data = {
    'userID': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'bookTitle': [
        'Book A', 'Book B', 'Book C', 'Book D', 'Book E'
    ],
    'bookRating': [5, 4, 3, 2, 1]
}

# Create DataFrame
ratings = pd.DataFrame(data)

# Assume data is already filtered for sufficient ratings
# Create pivot table for user ratings
rating_pivot = ratings.pivot_table(index='bookTitle', columns='userID', values='bookRating').fillna(0)

# Convert pivot table to sparse matrix for memory efficiency
rating_matrix = csr_matrix(rating_pivot.values)

# Initialize Nearest Neighbors model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(rating_matrix)

# Define recommendation function
def get_recommends(book_title):
    # Find index of book in pivot table
    book_idx = rating_pivot.index.get_loc(book_title)
    
    # Find nearest neighbors (set n_neighbors <= number of books in dataset)
    n_neighbors = min(5, len(rating_pivot))
    distances, indices = model_knn.kneighbors(rating_pivot.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=n_neighbors)
    
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        recommended_books.append([rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]])
    
    return [book_title, recommended_books]

# Testing the recommendation function
print(get_recommends('Book A'))
