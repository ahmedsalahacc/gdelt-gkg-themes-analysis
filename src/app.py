import os
import yaml as yml
import numpy as np

import utils

def main():
    # Get config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'config.yml'), 'r') as f:
        config = yml.safe_load(f)
    
    # Get the themes file and load it
    parent_dir = os.path.dirname(current_dir)
    themes_filepath = os.path.join(parent_dir, config.get('themes_path'))

    themes = utils.load_raw_themes(themes_filepath)
    
    # load theme embeddings and cluster
    embeddings = np.load(os.path.join(parent_dir, config.get('embedding_file')))
    clusters = utils.cluster_embeddings(embeddings, n_clusters=config.get('n_clusters'))

    # view clusters
    words_buckets, _ = utils.bucket_clusters(
        embeddings, 
        themes,
        clusters
    )

    print(words_buckets)

    
if __name__ == '__main__':
    main()