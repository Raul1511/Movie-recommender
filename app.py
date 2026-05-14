import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")

@st.cache_data
def load_data():
    movies = pd.DataFrame(pickle.load(open("movies_dict.pkl", "rb")))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    cv = pickle.load(open("vectorizer.pkl", "rb"))
    vectors = pickle.load(open("vectors.pkl", "rb"))
    return movies, similarity, cv, vectors

movies, similarity, cv, vectors = load_data()

st.title("Movie Recommender System")

tab1, tab2 = st.tabs(["Recommend by title", "Recommend by interest"])

with tab1:
    movie = st.selectbox("Select a movie", sorted(movies["title"].values))
    n = st.slider("Number of recommendations", 3, 10, 5)
    if st.button("Recommend", key="btn1"):
        idx = movies[movies["title"] == movie].index[0]
        distances = similarity[idx]
        results = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n+1]
        for rank, (i, score) in enumerate(results, 1):
            st.write(f"**{rank}. {movies.iloc[i].title}**")
            st.progress(float(score))

with tab2:
    interest = st.text_input("Describe what you want to watch", "crime mystery detective")
    n2 = st.slider("Number of recommendations", 3, 10, 5, key="s2")
    if st.button("Search", key="btn2"):
        user_vec = cv.transform([interest.lower()]).toarray()
        scores = cosine_similarity(user_vec, vectors)[0]
        results = sorted(list(enumerate(scores)), reverse=True, key=lambda x: x[1])[:n2]
        for rank, (i, score) in enumerate(results, 1):
            st.write(f"**{rank}. {movies.iloc[i].title}**")
            st.progress(float(min(score, 1.0)))
