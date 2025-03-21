# Supply Chain Disruptions Analysis

A Python application for analyzing supply chain disruptions using GDELT GKG themes through embedding and clustering.

## Features

- Theme embedding using BAAI/bge-large-en-v1.5 model
- Clustering of theme embeddings using K-means 
- Semantic search across themes

## Setup

1. Clone the repository:
```sh
git clone https://github.com/ahmedsalahacc/supplychain_disruptions.git
cd supplychain_disruptions
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Download the GDELT GKG themes file and place it in `cache/LOOKUP-GKGTHEMES.TXT`

## Usage

### Generate Theme Embeddings

Run the embedding script:
```sh
python src/embed_themes.py
```

This will:
- Load themes from the GDELT file
- Generate embeddings using the configured model
- Save embeddings to `cache/GKGThemeEmbeddings.npy`

### Run Analysis

Run the main application:
```sh
python src/app.py
```

The app provides two main functions:

1. Clustering (Option 1):
   - Clusters themes into configured number of groups
   - Saves results to JSON files in cache directory

2. Theme Search (Option 2): 
   - Enter keywords to find semantically similar themes
   - Shows top 10 matches with similarity scores

## Configuration

Edit `config.yml` to modify:

- File paths and cache locations
- Embedding model selection
- Number of clusters
- Other parameters

## Docker Support

Build and run using Docker:

```sh
docker build -t supplychain_disruptions .
docker run -it supplychain_disruptions
```
