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

    print(themes)



if __name__ == '__main__':
    main()