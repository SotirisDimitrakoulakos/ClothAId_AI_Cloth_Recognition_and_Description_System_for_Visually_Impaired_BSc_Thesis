import mmcv
from mmfashion.apis import (init_model, inference_recognizer, show_prediction)
import os

class MMFashionPredictor:
    def __init__(self):
        # Initialize MMFashion models
        self.config_file = '/content/drive/MyDrive/AI_Infrastructure/MMFashion/configs/attribute_predict/global_predictor_vgg_attr.py'
        self.checkpoint_file = '/content/drive/MyDrive/AI_Infrastructure/MMFashion/checkpoints/latest.pth'  
        self.model = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        
        # Define which attributes to predict for which article types
        self.attribute_mapping = {
            'shirts': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'jeans': ['pattern', 'fabric', 'length'],
            'track pants': ['pattern', 'fabric', 'length'],
            'tshirts': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'socks': ['pattern', 'fabric'],
            'casual shoes': ['pattern', 'fabric'],
            'flip flops': ['pattern', 'fabric'],
            'tops': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'bra': ['pattern', 'fabric'],
            'sandals': ['pattern', 'fabric'],
            'sweatshirts': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'formal shoes': ['pattern', 'fabric'],
            'flats': ['pattern', 'fabric'],
            'kurtas': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'waistcoat': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'sports shoes': ['pattern', 'fabric'],
            'shorts': ['pattern', 'fabric', 'length'],
            'briefs': ['pattern', 'fabric'],
            'sarees': ['pattern', 'fabric', 'length'],
            'heels': ['pattern'],
            'innerwear vests': ['pattern', 'fabric'],
            'scarves': ['pattern', 'fabric'],
            'rain jacket': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'dresses': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'night suits': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'skirts': ['pattern', 'fabric', 'length'],
            'blazers': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'kurta sets': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'shrug': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'trousers': ['pattern', 'fabric', 'length'],
            'camisoles': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'boxers': ['pattern', 'fabric'],
            'dupatta': ['pattern', 'fabric'],
            'capris': ['pattern', 'fabric', 'length'],
            'bath robe': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'mufflers': ['pattern', 'fabric'],
            'tunics': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'jackets': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'lounge pants': ['pattern', 'fabric', 'length'],
            'sports sandals': ['pattern', 'fabric'],
            'sweaters': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'tracksuits': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'swimwear': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'nightdress': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'ties': ['pattern', 'fabric'],
            'leggings': ['pattern', 'fabric', 'length'],
            'kurtis': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'jumpsuit': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'robe': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'salwar and dupatta': ['pattern', 'fabric', 'length'],
            'patiala': ['pattern', 'fabric', 'length'],
            'stockings': ['pattern', 'fabric'],
            'tights': ['pattern', 'fabric'],
            'churidar': ['pattern', 'fabric', 'length'],
            'lounge tshirts': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'lounge shorts': ['pattern', 'fabric', 'length'],
            'gloves': ['pattern', 'fabric'],
            'stoles': ['pattern', 'fabric'],
            'shapewear': ['pattern', 'fabric'],
            'nehru jackets': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'salwar': ['pattern', 'fabric', 'length'],
            'jeggings': ['pattern', 'fabric', 'length'],
            'rompers': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'booties': ['pattern', 'fabric'],
            'lehenga choli': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'clothing set': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length'],
            'rain trousers': ['pattern', 'fabric', 'length'],
            'suits': ['pattern', 'sleeve_length', 'neckline', 'fabric', 'length']
        }
    
    def predict(self, image_path, article_type):
        # Determine which attributes to predict
        attributes_to_predict = self.attribute_mapping.get(
            article_type.lower(), 
            ['pattern', 'fabric']  # Default attributes
        )
        
        # Perform prediction
        results = inference_recognizer(self.model, image_path)
        
        # Filter results to only include relevant attributes
        filtered_results = {}
        for attr in attributes_to_predict:
            if attr in results:
                filtered_results[attr] = results[attr]
        
        return filtered_results