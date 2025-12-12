from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

DATA_PROCESSED = Path("data/processed/takayasu_annotated.csv")

def main():
    df = pd.read_csv(DATA_PROCESSED)
    texts = df["abstract"].fillna("").tolist()

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)

    # LDA topics (e.g., 5 topics)
    lda = LatentDirichletAllocation(
        n_components=5,
        learning_method="batch",
        random_state=42
    )
    topic_dist = lda.fit_transform(X)
    df["lda_topic"] = topic_dist.argmax(axis=1)

    # k-means on the same TF-IDF space (e.g., 6 clusters)
    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(X)

    out_path = Path("data/processed/takayasu_annotated_clustered.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved clustered dataset to {out_path}")

if __name__ == "__main__":
    main()
