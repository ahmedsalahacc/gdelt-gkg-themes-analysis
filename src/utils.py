"""Util functions for GKG themes preprocessing."""

import re
import numpy as np

from typing import Dict, List, Tuple, Optional

from fastembed import TextEmbedding

from sklearn.cluster import KMeans

from tqdm import tqdm


def load_raw_themes(filepath: str) -> Dict[str, int]:
    """Loads raw themes from txt file into a dictionary.

    Text file must be like the one in http://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT

    Args:
        filepath (str): Path to the txt file containing the themes.

    Returns:
        Dict[str:int]: Dictionary with the themes and their respective ids.
    """
    themes = {}
    with open(filepath, mode="r") as f:
        for line in f:
            theme, theme_id = line.strip().split()
            themes[theme] = int(theme_id)

    return themes


def filter_theme_prefixes(themes: List[str]) -> List[str]:
    """Removes the GDELT GKG prefixes from the themes.

    Args:
        themes (List[str]): List of themes with prefixes.

    Returns:
        List[str]: List of themes without prefixes.
    """
    filtered = []
    for theme in themes:
        # filter tax_
        theme = re.sub(r"^tax_", "", theme)
        # filter wb_(numeric)
        theme = re.sub(r"^wb_\d+", "", theme)
        # filter econ_
        theme = re.sub(r"^econ_", "", theme)
        # filter soc_
        theme = re.sub(r"^soc_", "", theme)
        # filter epu_
        theme = re.sub(r"^epu_", "", theme)

        filtered.append(theme)

    return filtered


def embed(words: List[str], model_name) -> np.ndarray:
    """Embeds a list of words into a numpy array.

    Args:
        words (List[str]): List of words to be embedded.

    Returns:
        np.ndarray: Numpy array with the embeddings.
    """
    model = TextEmbedding(
        model_name=model_name,
    )
    embeddings = model.embed(tqdm(words, desc="Embedding"))
    return np.array(list(embeddings))


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Clusters the embeddings into n clusters.

    Args:
        embeddings (np.ndarray): Numpy array with the embeddings.
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Numpy array with the cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_


def bucket_clusters(
    embeddings: np.ndarray, keywords: List[str], clusters: List[int]
) -> Tuple[Dict[int, List[str]], Dict[int, List[np.ndarray]]]:
    """Buckets the clusters into a dictionary.

    Args:
        embeddings (np.ndarray): List of GKG themes embeddings.
        keywords (List[str]): List of GKG themes.
        clusters (List[int]): Clusters from the clustering algorithm.

    Returns:
        [type]: [description]
    """
    words_buckets = {}
    embeddings_buckets = {}

    for cluster, keyword, embedding in zip(clusters, keywords, embeddings):
        if cluster not in words_buckets:
            words_buckets[cluster.item()] = []
            embeddings_buckets[cluster.item()] = []

        words_buckets[cluster.item()].append(keyword)
        embeddings_buckets[cluster.item()].append(embedding.tolist())

    return words_buckets, embeddings_buckets


def find_nearest(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    batch_size: Optional[int] = None,
    top_k: int = 10,
) -> np.ndarray:
    """Find the top k similar embeddings to a query embedding.

    Args:
        query (np.ndarray): Query embedding.
        reference_embeddings (np.ndarray): Reference embeddings.
        batch_size (int, optional): Batch size for the similarity search. Defaults to 1000.
        top_k (int, optional): Number of top similar embeddings to return. Defaults to 10.

    Returns:
        np.ndarray: Numpy array with the top k similar embeddings.
    """
    results = np.empty((reference_embeddings.shape[0]))
    query_norm = np.linalg.norm(query_embedding)

    for i in range(0, reference_embeddings.shape[0], batch_size):
        # batched calculations to fit in memory
        batch = reference_embeddings[i : i + batch_size]

        # Compute cosine similarity
        reference_norms = np.linalg.norm(batch, axis=1).reshape(1, -1)

        cosine_score = np.dot(batch, query_embedding.T).reshape(1, -1)
        cosine_score /= query_norm * reference_norms

        # Store results
        results[i : i + batch_size] = cosine_score

    # Argsort the results
    argsort = np.argsort(results)[::-1]
    print(results.shape)
    return argsort[:top_k], results[argsort[:top_k]]
