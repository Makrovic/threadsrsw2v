import pandas as pd
import numpy as np
import gradio as gr
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from custom_cleaning import CustomCleaning
from threads import Threads
from threads_interface import ThreadsInterface
from collections import Counter

threadsnet = Threads()
# threadsnet = ThreadsInterface()
cleaning = CustomCleaning()

## BOOK

df = pd.read_csv("books.csv")

df["desc_cleaned"] = [cleaning.clean(desc) for desc in df["description"]]
df["desc_cleaned"] = [cleaning.remove_stopword(desc) for desc in df["desc_cleaned"]]
df["desc_cleaned"] = [cleaning.stem(desc) for desc in df["desc_cleaned"]]

df["desc_cleaned"] = df["desc_cleaned"].str.split()

book_model = Word2Vec(
    sentences=df["desc_cleaned"],
    vector_size=300,
    window=5,
    workers=4,
    epochs=30,
    min_count=1,
    sg=1,
)


## USER
def build_user_profile(username):
    try:
        user_id = threadsnet.public_api.get_user_id(username)
        # user_id = threadsnet.retrieve_user_id(username)
    except:
        raise gr.Error("Username Not Found!")

    user_threads = threadsnet.public_api.get_user_threads(user_id)
    # user_threads = threadsnet.retrieve_user_threads(user_id)

    threads = user_threads["data"]["mediaData"]["threads"]

    user_caption = []
    for thread in threads:
        if thread["thread_items"][0]["post"]["caption"] is not None:
            text = thread["thread_items"][0]["post"]["caption"]["text"]
            text = cleaning.clean(text)  # clean sentence from unnecessary characters
            text = cleaning.remove_stopword(text)  # remove stopwords from sentence
            text = cleaning.stem(text)  # stem
            user_caption.append(text)

    user_sent = " ".join(user_caption)

    user_sent_processed = user_sent.split()
    return user_sent_processed


## EMBEDDING


def get_vector(synopsis):
    vectors = [book_model.wv[word] for word in synopsis if word in book_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)  # sum of vectors / n
    else:
        return None


# Embed the synopses into vectors
desc_vectors = []
for desc in df["desc_cleaned"]:
    desc_vector = get_vector(desc)
    desc_vectors.append(desc_vector)

df["desc_vector"] = [vector for vector in desc_vectors if vector is not None]

## GET RECOMMENDATION


def recommend(username):
    user_sent_processed = build_user_profile(username)
    user_query_vector = get_vector(user_sent_processed)
    similarity_scores = cosine_similarity(
        [user_query_vector], df["desc_vector"].tolist()
    )[0]
    similar_indices = similarity_scores.argsort()[-5:][::-1]

    similar_books = df.loc[
        similar_indices,
        ["book_id", "title"],
    ]
    similar_books["similarity_score"] = similarity_scores[similar_indices]

    word_counts = Counter(user_sent_processed)
    most_common_words = word_counts.most_common(10)

    data_dicts = [{"token": row[0], "frekuensi": row[1]} for row in most_common_words]
    freq_df = pd.DataFrame(data_dicts)

    return similar_books, freq_df


# print(recommend("salimafillah"))

app = gr.Interface(fn=recommend, inputs="text", outputs=["dataframe", "dataframe"])
app.launch(share=True)
