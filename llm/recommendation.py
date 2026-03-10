import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import requests
import zipfile
import io
import os

def download_and_extract_movielesn():
    """Loads MoveLens100K dataset"""
    if not os.path.exists('ml-100k'):
        print("Downloading MovieLens 100k dataset...")
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print("Download and extraction completed.")
    else:
        print("MovieLens 100k dataset already exists.")

download_and_extract_movielesn()

ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
print(f"Dataset shape: {ratings_df.shape}")
print(f"Number of unique users: {ratings_df['user_id'].nunique()}")
print(f"Number of unique movies: {ratings_df['item_id'].nunique()}")
print(f"Range of ratings: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)
model.fit(trainset)

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(f"Average RMSE: {cv_results['test_rmse'].mean():.4f}")
print(f"Average MAE: {cv_results['test_mae'].mean():.4f}")


movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                        names=['item_id','title','release_date','video_release_date','IMDb_URL'] + list(range(19)))
movies_df = movies_df[['item_id','title']]

def recommend_movies(user_id, n=10):
    all_movies = movies_df['item_id'].unique()
    
    rated_movies = ratings_df[ratings_df['user_id'] == user_id]['item_id'].values
    
    unrated_movies = np.setdiff1d(all_movies, rated_movies)
    
    predictions = []
    for item_id in unrated_movies:
        predicted_rating = model.predict(user_id, item_id).est
        predictions.append((item_id, predicted_rating))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    top_recommendations = predictions[:n]
    
    recommendations = pd.DataFrame(top_recommendations, columns=['item_id', 'predicted_rating'])
    recommendations = recommendations.merge(movies_df, on='item_id')
    
    return recommendations

user_id = 42
recommendations = recommend_movies(user_id)

print(f"Top recommendations for user {user_id}:")
print(recommendations[['title', 'predicted_rating']])