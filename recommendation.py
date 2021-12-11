import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_data():
    book_data = pd.read_csv('dataset/data.csv')
    book_data['title'] = book_data['title'].str.lower()
    return book_data


def combine_data(data):
    data_recommend = data.drop(columns=['id', 'title', 'desc'])
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    data_recommend = data_recommend.drop(columns=['author', 'genres'])
    return data_recommend


def transform_data(data_combine, data_desc):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['combine'])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_desc['desc'])

    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')

    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)

    return cosine_sim


def recommend_movies(title, data, combine, transform):
    indices = pd.Series(data.index, index=data['title'])
    index = indices[title]

    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]

    book_indices = [i[0] for i in sim_scores]

    book_id = data['id'].iloc[book_indices]
    book_title = data['title'].iloc[book_indices]
    book_genres = data['genres'].iloc[book_indices]

    recommendation_data = pd.DataFrame(columns=['Id', 'Name', 'Genres'])

    recommendation_data['Id'] = book_id
    recommendation_data['Name'] = book_title
    recommendation_data['Genres'] = book_genres

    return recommendation_data


def results(book_name):
    book_name = book_name.lower()

    find_book = get_data()
    combine_result = combine_data(find_book)
    transform_result = transform_data(combine_result, find_book)

    if book_name not in find_book['title'].unique():
        return 'Book not in Database'

    else:
        recommendations = recommend_movies(book_name, find_book, combine_result, transform_result)
        return recommendations.to_dict('records')
