from pathlib import Path
import flammkuchen as fl
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, f1_score, precision_score, recall_score)
import calcium_event_classifier as cec


def test_logistic_regression_model():
    """
    Test a saved logistic regression model on a test dataset.
    Finds optimal threshold based on F1 score (same approach as CNN).
    """
    
    # Load the saved model
    model_path = Path(
        r"C:/Users/dcupolillo/Projects/calcium_event_classifier/"
        r"models/251120_LR_model.h5")
    
    checkpoint = fl.load(model_path)
    lr_model = checkpoint['model']
    
    # Load test dataset
    test_data_path = Path(
        r"C:/Users/dcupolillo/Projects/calcium_event_classifier/"
        r"datasets/251114_test_dataset.h5")
    test_data = fl.load(test_data_path)
    
    # Create dataset and extract features
    test_dataset = cec.DffDataset(test_data, augment=False)
    
    X_test = np.array(
        [test_dataset[i][0].squeeze().numpy() 
         for i in range(len(test_dataset))])
    y_test = np.array(
        [test_dataset[i][1].item() 
         for i in range(len(test_dataset))])
    
    print(f"\nTest dataset: {len(y_test)} samples")
    print(f"  Class 0: {(y_test == 0).sum()}")
    print(f"  Class 1: {(y_test == 1).sum()}")
    
    # Get predicted probabilities
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold based on F1 score
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Make predictions with default (0.5) and optimal thresholds
    y_pred_default = lr_model.predict(X_test)  # Uses 0.5 threshold
    y_pred_optimal = (y_prob >= best_threshold).astype(int)
    
    # Evaluate with both thresholds
    test_acc_default = accuracy_score(y_test, y_pred_default)
    test_acc_optimal = accuracy_score(y_test, y_pred_optimal)
    
    test_f1_default = f1_score(y_test, y_pred_default)
    test_f1_optimal = f1_score(y_test, y_pred_optimal)
    
    test_precision_default = precision_score(y_test, y_pred_default)
    test_precision_optimal = precision_score(y_test, y_pred_optimal)
    
    test_recall_default = recall_score(y_test, y_pred_default)
    test_recall_optimal = recall_score(y_test, y_pred_optimal)
    
    print("\n" + "="*60)
    print("Test Set Performance")
    print("="*60)
    print(f"Default threshold (0.5):")
    print(f"  Accuracy: {test_acc_default:.4f} ({test_acc_default*100:.1f}%)")
    print(f"  Precision: {test_precision_default:.4f} ({test_precision_default*100:.1f}%)")
    print(f"  Recall: {test_recall_default:.4f} ({test_recall_default*100:.1f}%)")
    print(f"  F1 Score: {test_f1_default:.4f}")
    print(f"\nOptimal threshold ({best_threshold:.4f}):")
    print(f"  Accuracy: {test_acc_optimal:.4f} ({test_acc_optimal*100:.1f}%)")
    print(f"  Precision: {test_precision_optimal:.4f} ({test_precision_optimal*100:.1f}%)")
    print(f"  Recall: {test_recall_optimal:.4f} ({test_recall_optimal*100:.1f}%)")
    print(f"  F1 Score: {test_f1_optimal:.4f}")
    print("="*60)
    
    print("\nClassification Report (Optimal Threshold):")
    print(classification_report(y_test, y_pred_optimal, 
                                target_names=["No Event", "Event"]))
    
    print("\nConfusion Matrix (Optimal Threshold):")
    cm = confusion_matrix(y_test, y_pred_optimal)
    print(cm)
    print(f"  [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
    print(f"   [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    class0_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    class0_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    class1_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    class1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\nPer-class metrics:")
    print(f"Class 0 (No Event):")
    print(f"  Precision: {class0_precision:.3f} ({class0_precision*100:.1f}%)")
    print(f"  Recall: {class0_recall:.3f} ({class0_recall*100:.1f}%)")
    print(f"Class 1 (Event):")
    print(f"  Precision: {class1_precision:.3f} ({class1_precision*100:.1f}%)")
    print(f"  Recall: {class1_recall:.3f} ({class1_recall*100:.1f}%)")
    
    return lr_model, X_test, y_test, y_pred_optimal, y_prob, best_threshold


if __name__ == "__main__":
    test_logistic_regression_model()
