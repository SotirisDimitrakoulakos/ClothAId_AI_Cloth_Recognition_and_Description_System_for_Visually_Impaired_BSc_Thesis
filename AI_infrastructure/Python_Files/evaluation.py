import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def evaluate(self, X_test, y_test, history=None, training_time=None, num_epochs=None):
        # Encode test labels
        y_test_encoded = {}
        for attr, labels in y_test.items():
            y_test_encoded[attr] = self.label_encoders[attr].transform(labels)
        
        # Predict
        y_pred_raw = self.model.predict(X_test)

        # Handle dict or list prediction outputs
        if isinstance(y_pred_raw, dict):
            y_pred = y_pred_raw
        elif isinstance(y_pred_raw, list):
            if len(y_pred_raw) != len(y_test):
                raise ValueError("Number of model outputs does not match number of attributes.")
            y_pred = {attr: preds for attr, preds in zip(y_test.keys(), y_pred_raw)}
        else:
            raise TypeError("Model prediction output must be dict or list.")
        
        results = {}

        # Evaluate each attribute
        for attr in y_test.keys():
            if attr not in y_pred:
                raise ValueError(f"Missing predictions for attribute: {attr}")
            true_labels = y_test_encoded[attr]
            pred_labels = np.argmax(y_pred[attr], axis=1)

            # Classification report
            report = classification_report(
                true_labels,
                pred_labels,
                target_names=self.label_encoders[attr].classes_,
                output_dict=True
            )

            # Confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)

            results[attr] = {
                'report': report,
                'confusion_matrix': cm
            }

        # Add training history if available
        if history:
            results['training'] = {
                'train_accuracy': history.history.get('accuracy', [0,0])[-1],
                'val_accuracy': history.history.get('val_accuracy', [0,0])[-1],
                'train_loss': history.history.get('loss', [0,0])[-1],
                'val_loss': history.history.get('val_loss', [0,0])[-1]
            }
        if training_time:
            results['training_time'] = training_time
        if num_epochs:
            results['num_epochs'] = num_epochs

        return results

    def print_summary(self, results):
        # Print training summary
        if 'training' in results:
            print("\n=== Training Summary ===")
            for k, v in results['training'].items():
                print(f"{k.replace('_', ' ').title()}: {v:.4f}" if isinstance(v, float) else f"{k.replace('_', ' ').title()}: {v}")

        # Print evaluation per attribute
        for attr, metrics in results.items():
            if attr in ['training', 'training_time', 'num_epochs']:
                continue

            report = metrics['report']
            print(f"\n=== Attribute: {attr.upper()} ===")
            print(f"Accuracy: {report['accuracy']:.4f}")
            print("Precision / Recall / F1 (macro):",
                  f"{report['macro avg']['precision']:.4f} /",
                  f"{report['macro avg']['recall']:.4f} /",
                  f"{report['macro avg']['f1-score']:.4f}")

    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def plot_all_confusion_matrices(self, results):
        for attr, metrics in results.items():
            if attr in ['training', 'training_time', 'num_epochs']:
                continue
            cm = metrics.get('confusion_matrix')
            classes = self.label_encoders[attr].classes_
            title = f"Confusion Matrix for {attr.upper()}"
            self.plot_confusion_matrix(cm, classes, title)
