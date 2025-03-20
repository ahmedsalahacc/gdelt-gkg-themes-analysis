"""Util functions for GKG themes preprocessing."""
import re
import numpy as np

from typing import Dict, List

from fastembed import TextEmbedding

from tqdm import tqdm

def load_raw_themes(filepath:str)->Dict[str,int]:
    """Loads raw themes from txt file into a dictionary. 
    
    Text file must be like the one in http://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT

    Args:
        filepath (str): Path to the txt file containing the themes.

    Returns:
        Dict[str:int]: Dictionary with the themes and their respective ids.
    """
    themes = {}
    with open(filepath, mode='r') as f:
        for line in f:
            theme, theme_id = line.strip().split()
            themes[theme] = int(theme_id)
    
    return themes

def filter_theme_prefixes(themes:List[str])->List[str]:
    """Removes the GDELT GKG prefixes from the themes.

    Args:
        themes (List[str]): List of themes with prefixes.

    Returns:
        List[str]: List of themes without prefixes.
    """
    filtered = []
    for theme in themes:
        # filter tax_
        theme = re.sub(r'^tax_', '', theme)
        # filter wb_(numeric)
        theme = re.sub(r'^wb_\d+', '', theme)
        # filter econ_
        theme = re.sub(r'^econ_', '', theme)
        # filter soc_
        theme = re.sub(r'^soc_', '', theme)
        # filter epu_
        theme = re.sub(r'^epu_', '', theme)

        filtered.append(theme)

    return filtered

def embed(words:List[str], model_name)->np.ndarray:
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