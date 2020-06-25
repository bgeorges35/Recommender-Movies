import sys
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np


def pivot_csv():
    try:
        chunk = pd.read_csv("datasets/pivot.csv", chunksize=100000)
        df = pd.concat(chunk)
        df.set_index(["title", "weight_rating"], inplace=True)
    except:
        df = merge_csv()
        print("Creating datasets/pivot.csv...")
        df = pd.pivot_table(df, values="rating", index=["title", "weight_rating"], columns=["userId"],
                            fill_value=0)
        df.to_csv("datasets/pivot.csv")
    return df


def merge_csv():
    try:
        df = pd.read_csv("datasets/merge.csv", usecols=["index", "title", "userId", "weight_rating", "rating"],
                         dtype={"index": int, "userId": int, "rating": float, "title": str, "weight_rating": float})
        df.set_index(['index'], inplace=True)
    except:
        print("Creating datasets/merge.csv...")
        try:
            df_links = pd.read_csv("datasets/links.csv", header=0, usecols=[0, 2],
                                   names=['movieId', 'id']).dropna().drop_duplicates(subset="id").astype(int)

            df_data = pd.read_csv("datasets/movies_metadata.csv",
                                  usecols=["id", "title", "vote_average", "vote_count"]).dropna().drop_duplicates(
                subset="id")

            df_rating = pd.read_csv("datasets/ratings_small.csv",
                                    usecols=['userId', 'movieId', 'rating']).astype({"userId": int, "movieId": int,
                                                                                     "rating": float})

            df_data = df_data[df_data.id.apply(lambda x: x.isnumeric())].astype({"id": int, "vote_count": int,
                                                                                 "title": str, 'vote_average': float})

            m = df_data["vote_count"].quantile(0.9)
            C = df_data["vote_average"].mean()
            df_data = df_data[(df_data[["vote_count"]] >= m).all(axis=1)]
            WR = weight_rating(df_data["vote_count"], df_data["vote_average"], m, C)
            df_data["weight_rating"] = pd.Series(WR, index=df_data.index)
            df_links = pd.merge(df_links, df_data[['id', 'title', 'weight_rating']], on="id", how="inner")
            df = pd.merge(df_rating, df_links[['title', 'weight_rating', 'movieId']], on="movieId", how="inner")
            df.index.name = 'index'
            df.to_csv("datasets/merge.csv")
        except Exception as e:
            sys.exit(print("{}: {}".format(type(e).__name__, e)))
    return df


def pickle_model(movie_features_df_matrix):
    try:
        model_knn = pickle.load(open('knn.pkl', 'rb'))
        print("Pickle knn.pkl loaded...")
    except:
        print("Creating pickle knn.pkl...")
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(movie_features_df_matrix)
        knnPickle = open('knn.pkl', 'wb')
        # source, destination
        pickle.dump(model_knn, knnPickle)

    return model_knn


def weight_rating(vote_count, vote_average, m, C):
    """
    IMDB weighted rating
    R = average for the movie(mean) = (rating)
    v = number of votes for the movie = (votes)
    m = minimum votes required to be listed in the Top Rated list(currently 25, 000)
    C = the mean vote across the whole report
    weighted rating (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C
    """
    v = vote_count.to_numpy(dtype=int)
    R = vote_average.to_numpy(dtype=int)
    WR = np.round((v / (v + m) * R) + (m / (m + v) * C), 4)
    return WR


if __name__ == "__main__":
    df = pivot_csv()
    movie_features_df_matrix = csr_matrix(df.values)
    model_knn = pickle_model(movie_features_df_matrix)

    query_index = np.random.choice(movie_features_df_matrix.shape[0])
    distances, indices = model_knn.kneighbors(movie_features_df_matrix[query_index], n_neighbors=11)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('\n\nRecommendations for \033[91m {}\033[00m: weight rated: {:.2f}/10\n'.format(df.index[query_index][0],
                                                                                 df.index[query_index][1]))
        else:
            print('{}: \033[91m {}\033[00m weight rated: {:.2f}/10, with distance of: {:.4f}'.format(i,
                                                                                    df.index[indices.flatten()[i]][0],
                                                                                    df.index[indices.flatten()[i]][1],
                                                                                    distances.flatten()[i]))
