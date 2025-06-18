import mmcv
import os
import sys
import json
from mmfashion.apis import init_model, inference_recognizer

class MMFashionPredictor:
    def __init__(self):
        # Initialize MMFashion models
        self.config_file = 'configs/global_predictor_vgg_attr.py'
        self.checkpoint_file = 'checkpoints/latest.pth' 
        self.model = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        
        # Define which attributes to predict for which article types
        self.attribute_mapping = {
            'shirts': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'jeans': ['pattern', 'fabric' ],
            'track pants': ['pattern', 'fabric' ],
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
            'kurtas': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'waistcoat': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'sports shoes': ['pattern', 'fabric'],
            'shorts': ['pattern', 'fabric' ],
            'briefs': ['pattern', 'fabric'],
            'sarees': ['pattern', 'fabric' ],
            'heels': ['pattern'],
            'innerwear vests': ['pattern', 'fabric'],
            'scarves': ['pattern', 'fabric'],
            'rain jacket': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'dresses': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'night suits': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'skirts': ['pattern', 'fabric' ],
            'blazers': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'kurta sets': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'shrug': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'trousers': ['pattern', 'fabric' ],
            'camisoles': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'boxers': ['pattern', 'fabric'],
            'dupatta': ['pattern', 'fabric'],
            'capris': ['pattern', 'fabric' ],
            'bath robe': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'mufflers': ['pattern', 'fabric'],
            'tunics': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'jackets': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'lounge pants': ['pattern', 'fabric' ],
            'sports sandals': ['pattern', 'fabric'],
            'sweaters': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'tracksuits': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'swimwear': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'nightdress': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'ties': ['pattern', 'fabric'],
            'leggings': ['pattern', 'fabric' ],
            'kurtis': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'jumpsuit': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'robe': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'salwar and dupatta': ['pattern', 'fabric' ],
            'patiala': ['pattern', 'fabric' ],
            'stockings': ['pattern', 'fabric'],
            'tights': ['pattern', 'fabric'],
            'churidar': ['pattern', 'fabric' ],
            'lounge tshirts': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'lounge shorts': ['pattern', 'fabric' ],
            'gloves': ['pattern', 'fabric'],
            'stoles': ['pattern', 'fabric'],
            'shapewear': ['pattern', 'fabric'],
            'nehru jackets': ['pattern', 'sleeve_length', 'neckline', 'fabric'],
            'salwar': ['pattern', 'fabric' ],
            'jeggings': ['pattern', 'fabric' ],
            'rompers': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'booties': ['pattern', 'fabric'],
            'lehenga choli': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'clothing set': ['pattern', 'sleeve_length', 'neckline', 'fabric' ],
            'rain trousers': ['pattern', 'fabric' ],
            'suits': ['pattern', 'sleeve_length', 'neckline', 'fabric' ]
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
    
    def main():
        if len(sys.argv) != 3:
            print("Usage: python predict_mmfashion.py <image_path> <article_type>")
            sys.exit(1)

        image_path = sys.argv[1]
        article_type = sys.argv[2]

        predictor = MMFashionPredictor()
        preds = predictor.predict(image_path, article_type)
        print(json.dumps(preds))

    if __name__ == "__main__":
        main()