"""Main module to run the application."""

import os

import json

import yaml as yml

import numpy as np

import utils


def main():
    # Get config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config.yml"), "r") as f:
        config = yml.safe_load(f)

    # Get the themes file and load it
    parent_dir = os.path.dirname(current_dir)
    themes_filepath = os.path.join(parent_dir, config.get("themes_path"))

    themes = utils.load_raw_themes(themes_filepath)
    themes_keywords = list(themes.keys())

    # load theme embeddings and cluster
    theme_embeddings = np.load(os.path.join(parent_dir, config.get("embedding_file")))

    # Query the user for what to do next
    action = input("What do you want to do? (1:cluster, 2:query):")

    if int(action) == 1:
        clusters = utils.cluster_embeddings(
            theme_embeddings, n_clusters=config.get("n_clusters")
        )

        # view clusters
        words_buckets, embeddings_buckets = utils.bucket_clusters(
            theme_embeddings, themes_keywords, clusters
        )

        json.dump(
            words_buckets,
            open(os.path.join(parent_dir, config.get("words_bucket_file")), "w"),
        )
        json.dump(
            embeddings_buckets,
            open(os.path.join(parent_dir, config.get("embedding_bucket_file")), "w"),
        )
    elif int(action) == 2:
        # Query the user to find the nearest theme to a given keyword
        query = input("Enter a keyword:")

        # Embed the query
        query_embedding = utils.embed(
            [query], model_name=config.get("embedding_model_name")
        )
        # Load the theme embeddings and find the nearest neighbors for the query
        indices, scores = utils.find_nearest(
            query_embedding, theme_embeddings, batch_size=10_000, top_k=10
        )

        # Find matching themes
        nearest_themes = [themes_keywords[i.item()] for i in indices]

        # Print the results
        print("Given query:", query)
        for theme, score in zip(nearest_themes, scores):
            print("Found:", theme, "With score", score)
    else:
        raise ValueError("Invalid choice, must be either 1 or 2.")


if __name__ == "__main__":
    main()
