import sys
import json
import torch
from mmfashion.apis import init_model, inference_recognizer

class MMFashionPredictor:
    # Your raw attribute list (attribute_name + attribute_type)
    ATTR_DATA = """
floral               1
graphic              1
striped              1
embroidered          1
pleated              1
solid                1
lattice              1
long_sleeve          2
short_sleeve         2
sleeveless           2
maxi_length          3
mini_length          3
no_dress             3
crew_neckline        4
v_neckline           4
square_neckline      4
no_neckline          4
denim                5
chiffon              5
cotton               5
leather              5
faux                 5
knit                 5
tight                6
loose                6
conventional         6
"""

    def __init__(self):
        # Load model config and checkpoint
        self.config_file = 'configs/global_predictor_vgg_attr.py'
        self.checkpoint_file = 'checkpoints/latest.pth'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = init_model(self.config_file, self.checkpoint_file, device=self.device)

        # Parse attribute data and build groups
        self.attribute_mapping = self.build_attribute_mapping()
        
        # Mapping articleType → which attribute categories to predict (your original mapping)
        self.article_type_to_categories = {
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
    
    def build_attribute_mapping(self):
        # Build attribute mapping dict: category -> list of attribute names
        attr_lines = self.ATTR_DATA.strip().split('\n')
        attr_type_map = {1: 'pattern', 2: 'sleeve_length', 4: 'neckline', 5: 'fabric'}
        grouped_attrs = {'pattern': [], 'sleeve_length': [], 'neckline': [], 'fabric': []}

        for line in attr_lines:
            parts = line.strip().split()
            if len(parts) == 2:
                name, atype = parts[0], int(parts[1])
                if atype in attr_type_map:
                    grouped_attrs[attr_type_map[atype]].append(name)

        return grouped_attrs

    def predict(self, image_path, article_type):
        # Get categories to predict for this article_type
        categories = self.article_type_to_categories.get(article_type.lower(), ['pattern', 'fabric'])

        # Build list of attribute names to predict
        attrs_to_predict = []
        for cat in categories:
            attrs_to_predict.extend(self.attribute_mapping.get(cat, []))

        # Run inference on image (returns dict attr_name -> score)
        results = inference_recognizer(self.model, image_path)

        # Filter to only attributes relevant for this article type
        filtered_results = {attr: results[attr] for attr in attrs_to_predict if attr in results}

        return filtered_results


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_mmfashion.py <image_path> <article_type>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    article_type = sys.argv[2]

    predictor = MMFashionPredictor()
    preds = predictor.predict(image_path, article_type)

    print(json.dumps(preds))


if __name__ == "__main__":
    main()