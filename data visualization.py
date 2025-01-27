import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Open the files
ratings_file = open(r"C:\Users\gawai\OneDrive\Desktop\ml-1m\ml-1m\ratings.dat")
users_file = open(r"C:\Users\gawai\OneDrive\Desktop\ml-1m\ml-1m\users.dat")
movies_file = open(r"C:\Users\gawai\OneDrive\Desktop\ml-1m\ml-1m\movies.dat")

# Load the datasets into pandas DataFrames
ratings = pd.read_csv(
    ratings_file, 
    sep='::', 
    names=['UserID', 'MovieID', 'Rating', 'Timestamp'], 
    engine='python'
)

users = pd.read_csv(
    users_file, 
    sep='::', 
    names=['UserID', 'Gender', 'Age', 'Occupation', 'ZipCode'], 
    engine='python'
)

movies = pd.read_csv(
    movies_file, 
    sep='::', 
    names=['MovieID', 'Title', 'Genres'], 
    engine='python'
)

# Convert the timestamp to datetime
ratings['Timestamp'] = pd.to_datetime(ratings['Timestamp'], unit='s')
ratings['Year'] = ratings['Timestamp'].dt.year

# Merge the datasets for analysis
merged_data = pd.merge(pd.merge(ratings, movies, on='MovieID'), users, on='UserID')

# Extract release year from the movie title
merged_data['ReleaseYear'] = merged_data['Title'].str.extract(r'\((\d{4})\)').astype(float)

# One-hot encode genres for analysis
genres_dummies = merged_data['Genres'].str.get_dummies('|')

# 1. Distribution of Ratings by Genres
plt.figure(figsize=(12, 6))
genre_ratings = genres_dummies.multiply(merged_data['Rating'], axis=0).sum().sort_values(ascending=False)
genre_ratings.plot(kind='bar', color=sns.color_palette("pastel"))
plt.title('Distribution of Ratings by Genres', fontsize=16)
plt.xlabel('Genres', fontsize=12)
plt.ylabel('Total Ratings', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Ratings by Year
plt.figure(figsize=(12, 6))
merged_data.groupby('Year')['Rating'].mean().plot(kind='line', color='red', linestyle='--', marker='o')
plt.title('Average Ratings by Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.tight_layout()
plt.show()

# 3. Popular Genres by Gender
plt.figure(figsize=(12, 6))
gender_genre = merged_data.groupby('Gender')['Genres'].apply(lambda x: '|'.join(x)).str.get_dummies('|').sum()
gender_genre.plot(kind='bar', stacked=True, colormap='cool')
plt.title('Popular Genres by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Total Ratings', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Popular Genres by Age Group
age_groups = {
    1: 'Under 18', 
    18: '18-24', 
    25: '25-34', 
    35: '35-44', 
    45: '45-49', 
    50: '50-55', 
    56: '56+'
}
merged_data['AgeGroup'] = merged_data['Age'].map(age_groups)

age_genre = merged_data.groupby('AgeGroup')['Genres'].apply(lambda x: '|'.join(x)).str.get_dummies('|').sum()

plt.figure(figsize=(12, 6))
age_genre.plot(kind='bar', stacked=True, colormap='Spectral')
plt.title('Popular Genres by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Total Ratings', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Correlation Heatmap
numerical_columns = ['Rating', 'Age', 'Occupation']
correlation_data = pd.concat([merged_data[numerical_columns], genres_dummies], axis=1)
corr_matrix = correlation_data.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, linecolor='white')
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

numerical_columns = ['Rating', 'Age', 'Occupation']
correlation_data = pd.concat([merged_data[numerical_columns], genres_dummies], axis=1)
corr_matrix = correlation_data.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white')
plt.title("Correlation Heatmap: Ratings, Age, Occupation, and Genres", fontsize=16)
plt.tight_layout()
plt.show()

# 6. Heatmap of Ratings Distribution by Genres and Years
genre_year_pivot = merged_data.pivot_table(
    index='Genres', 
    columns='Year', 
    values='Rating', 
    aggfunc='count'
).fillna(0)

plt.figure(figsize=(16, 10))
sns.heatmap(
    genre_year_pivot, 
    cmap='YlGnBu', 
    annot=False, 
    cbar_kws={'label': 'Number of Ratings'}
)
plt.title('Heatmap of Ratings Distribution by Genres and Years', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Genres')
plt.tight_layout()
plt.show()

# 7. Popular Genres by User Activity
genre_popularity = merged_data.groupby('Genres')['Rating'].count().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    y=genre_popularity.index, 
    x=genre_popularity.values, 
    palette='viridis'
)
plt.title('Popular Genres by User Activity', fontsize=16)
plt.xlabel('Number of Ratings')
plt.ylabel('Genres')
plt.tight_layout()
plt.show()

