from utils import *
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    average_precision_score
)

class Metric:
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, preds, targets):
        # Convert inputs to numpy arrays for consistent handling
        preds = np.array(preds)
        targets = np.array(targets)
        
        # Input validation
        if len(preds) != len(targets):
            raise ValueError(f"Predictions and targets must have same length. Got {len(preds)} and {len(targets)}")
        if len(preds) == 0:
            raise ValueError("Empty predictions and targets")
        if not np.all(np.isin(preds, [0, 1])) or not np.all(np.isin(targets, [0, 1])):
            raise ValueError("All values must be binary (0 or 1)")
            
        self.preds.extend(preds)
        self.targets.extend(targets)

    def accuracy(self):
        return np.mean(np.array(self.preds) == np.array(self.targets))
    
    def precision(self):
        try:
            return precision_score(self.targets, self.preds)
        except Exception:
            return 0.0
    
    def recall(self):
        try:
            return recall_score(self.targets, self.preds)
        except Exception:
            return 0.0
    
    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def roc_auc(self):
        try:
            return roc_auc_score(self.targets, self.preds)
        except Exception:
            return None
    
    def average_precision(self):
        try:
            return average_precision_score(self.targets, self.preds)
        except Exception:
            return None
    
    def confusion_matrix(self):
        return confusion_matrix(self.targets, self.preds)

    def class_distribution(self):
        targets = np.array(self.targets)
        preds = np.array(self.preds)
        
        return {
            'target_distribution': {
                'class_0': np.mean(targets == 0),
                'class_1': np.mean(targets == 1)
            },
            'prediction_distribution': {
                'class_0': np.mean(preds == 0),
                'class_1': np.mean(preds == 1)
            }
        }

    def reset(self):
        """
        Reset the accumulated predictions and targets.
        """
        self.preds = []
        self.targets = []

    def compute(self):
        if not self.preds:
            raise ValueError("No predictions accumulated yet")
            
        # Calculate confusion matrix values
        tn, fp, fn, tp = self.confusion_matrix().ravel()
        
        return {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1_score(),
            'confusion_matrix': self.confusion_matrix(),
            'detailed_metrics': {
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp,
            },
            'class_distribution': self.class_distribution()
        }

if __name__ == '__main__':
    # Test the metrics with a simple example
    metric = Metric()
    
    # Basic test case
    metric.update([0, 1, 0, 1, 1], [0, 0, 0, 1, 1])
    results = metric.compute()
    print("Basic test metrics:", results)