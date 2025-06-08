import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from sklearn.model_selection import train_test_split

class FashionDatasetLoader:
    generalized_color_map = {
        'blue': ['blue', 'navy blue', 'teal'],
        'black': ['black'],
        'grey': ['grey', 'charcoal', 'steel', 'grey melange', 'silver', 'metallic', 'taupe'],
        'white': ['white', 'off white', 'beige', 'cream'],
        'beige': ['beige', 'khaki', 'tan', 'off white'],
        'brown': ['brown', 'bronze', 'copper', 'rust', 'coffee brown', 'skin', 'nude', 'mushroom brown'],
        'yellow': ['yellow', 'gold', 'mustard'],
        'red': ['red', 'maroon', 'burgundy'],
        'orange': ['orange', 'peach'],
        'green': ['green', 'olive', 'sea green', 'lime green', 'fluorescent green'],
        'purple': ['purple', 'lavender', 'mauve'],
        'pink': ['pink', 'magenta', 'rose'],
        'multi': ['multi'],
        'nan': ['nan'],
        'turquoise': ['turquoise blue'],
    }

    def __init__(self, dataset_path, metadata_file):
        self.dataset_path = dataset_path
        if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
            try:
                self.metadata = pd.read_csv(metadata_file, on_bad_lines='skip')  # Handle malformed rows
            except pd.errors.ParserError as e:
                raise ValueError(f"Error parsing metadata file: {e}")
            # List of subcategories to remove
            for i in ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'gender', 'season', 'usage']:
                self.metadata[i] = self.metadata[i].astype(str).str.strip().str.lower()
            unwanted_subcategories = ['lips', 'nails', 'wallets',
                'skin care', 'makeup', 'free gifts', 'skin',
                'beauty accessories', 'water bottle', 'eyes', 'bath and body', 'shoe accessories', 'cufflinks', 'sports equipment', 'hair',
                 'perfumes', 'home furnishing', 'umbrellas', 'fragrance', 'vouchers']
            unwanted_maincategories = ['personal care', 'home', 'free items']
            unwanted_articleTypes = ['baby dolls', 'deodorant', 'perfume and body mist', 'laptop bag', 'trolley bag', 'duffel bag', 'travel accessory',
                'mobile pouch', 'messenger bag', 'accessory gift set', 'tablet sleeve', 'footballs', 'hair colour',
                'cushion covers', 'key chain', 'umbrellas', 'water bottle', 'ipad', 'wallets', 'waist pouch', 'hair accessory', 'cufflinks']
            self.metadata = self.metadata[~self.metadata['subCategory'].isin(unwanted_subcategories)]
            self.metadata = self.metadata[~self.metadata['masterCategory'].isin(unwanted_maincategories)]
            self.metadata = self.metadata[~self.metadata['articleType'].isin(unwanted_articleTypes)]

            # Apply generalizations here:
            self.metadata['season'] = self.metadata['season'].apply(self._generalize_season)
            self.metadata['baseColour'] = self.metadata['baseColour'].apply(self._generalize_base_color)
        else:
            raise FileNotFoundError(f"Metadata file {metadata_file} not found or empty")
        
    def _generalize_season(self, season_value):
        # Group summer and spring as "summer/spring"
        if pd.isna(season_value) or season_value == 'nan':
            return 'nan'
        season_value = season_value.lower().strip()
        if season_value in ['summer', 'spring']:
            return 'summer/spring'
        # keep fall and winter as is
        if season_value in ['fall', 'winter']:
            return season_value
        # fallback - keep original
        return season_value

    def _generalize_base_color(self, base_color_value):
        if pd.isna(base_color_value) or base_color_value == 'nan':
            return 'nan'
        base_color_value = base_color_value.lower().strip()
        # find the generalized key for this base color
        for gen_color, variants in self.generalized_color_map.items():
            if base_color_value in variants:
                return gen_color
        # fallback - if no match, return original
        return base_color_value
        
    def load_image(self, image_id):
        image_path = os.path.join(self.dataset_path, f"{image_id}.jpg")
        try:
            with Image.open(image_path) as img:
                return img.convert("RGB").copy()  # forces full load into memory
        except FileNotFoundError:
            print(f"[SKIP] Image not found: {image_path}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None
    
    def get_metadata(self, image_id):
        return self.metadata[self.metadata['id'] == image_id].iloc[0].to_dict()
    
    def get_all_metadata(self):
        return self.metadata
    
    def filter_by_article_type(self, article_type, max_samples=500):
        filtered = self.metadata[self.metadata['articleType'] == article_type]
        filtered = filtered.sample(min(len(filtered), max_samples))
        return filtered