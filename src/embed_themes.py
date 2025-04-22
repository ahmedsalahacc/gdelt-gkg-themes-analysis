"""This module contains the code for embedding the themes."""

import os
import yaml as yml
import numpy as np

import utils


def main():
    # Get config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    with open(os.path.join(parent_dir, "config.yml"), "r") as f:
        config = yml.safe_load(f)

    # Get the themes file and load it
    themes_filepath = os.path.join(parent_dir, config.get("themes_path"))

    themes = utils.load_raw_themes(themes_filepath)

    # embed keywords and save it in cache
    keywords = list(themes.keys())
    keywords = utils.filter_theme_prefixes(keywords)

    print("Embedding", len(keywords), "keywords...")
    embeddings = utils.embed(keywords, model_name=config.get("embedding_model_name"))
    print("Finished with dim", embeddings.shape)
    np.save(os.path.join(parent_dir, config.get("embedding_file")), embeddings)


if __name__ == "__main__":
    main()
