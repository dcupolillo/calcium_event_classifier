from pathlib import Path
import flammkuchen as fl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, confusion_matrix, classification_report)
import calcium_event_classifier as cec


def threshold_crosing_test():
    """
    """

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
    
    std_window = (5, 15)
    detection_window = (15, 20)

    stds = np.array([np.std(trace[std_window[0]:std_window[1]]) for trace in X_test])

    crossing_196_sigma = np.zeros_like(X_test)
    crossing_25_sigma = np.zeros_like(X_test)

    for n in range(len(X_test)):
        crossing_196_sigma[n] = X_test[n] > 1.96 * stds[n]
        crossing_25_sigma[n] = X_test[n] > 2.5 * stds[n]

    # Classify as event if crossing at least twice consecutively in the detection window,
    y_pred_196_sigma = np.array([
        np.any(np.convolve(
            crossing_196_sigma[n][detection_window[0]:detection_window[1]],
            np.array([1, 1]), mode="valid") == 2)
        for n in range(len(X_test))]).astype(int)
    y_pred_25_sigma = np.array([
        np.any(np.convolve(
            crossing_25_sigma[n][detection_window[0]:detection_window[1]],
            np.array([1, 1]), mode="valid") == 2)
        for n in range(len(X_test))]).astype(int)
    
    # Calculate accuracy, precision, recall
    accuracy_196_sigma = accuracy_score(y_test, y_pred_196_sigma)
    accuracy_25_sigma = accuracy_score(y_test, y_pred_25_sigma)

    # Get confusion matrices for per-class metrics
    cm_196 = confusion_matrix(y_test, y_pred_196_sigma)
    cm_25 = confusion_matrix(y_test, y_pred_25_sigma)
    
    # Calculate per-class metrics for 1.96σ
    tn_196, fp_196, fn_196, tp_196 = cm_196.ravel()
    precision_class0_196 = tn_196 / (tn_196 + fn_196) if (tn_196 + fn_196) > 0 else 0
    recall_class0_196 = tn_196 / (tn_196 + fp_196) if (tn_196 + fp_196) > 0 else 0
    precision_class1_196 = tp_196 / (tp_196 + fp_196) if (tp_196 + fp_196) > 0 else 0
    recall_class1_196 = tp_196 / (tp_196 + fn_196) if (tp_196 + fn_196) > 0 else 0
    
    # Calculate per-class metrics for 2.5σ
    tn_25, fp_25, fn_25, tp_25 = cm_25.ravel()
    precision_class0_25 = tn_25 / (tn_25 + fn_25) if (tn_25 + fn_25) > 0 else 0
    recall_class0_25 = tn_25 / (tn_25 + fp_25) if (tn_25 + fp_25) > 0 else 0
    precision_class1_25 = tp_25 / (tp_25 + fp_25) if (tp_25 + fp_25) > 0 else 0
    recall_class1_25 = tp_25 / (tp_25 + fn_25) if (tp_25 + fn_25) > 0 else 0
    
    # Print comprehensive results
    print("\n" + "="*70)
    print("THRESHOLD CROSSING METHOD RESULTS")
    print("="*70)
    
    print("\n1.96σ Threshold:")
    print(f"  Overall Accuracy: {accuracy_196_sigma:.4f} ({accuracy_196_sigma*100:.1f}%)")
    print(f"  Class 0 (No Event) - Precision: {precision_class0_196:.3f} ({precision_class0_196*100:.1f}%), Recall: {recall_class0_196:.3f} ({recall_class0_196*100:.1f}%)")
    print(f"  Class 1 (Event)    - Precision: {precision_class1_196:.3f} ({precision_class1_196*100:.1f}%), Recall: {recall_class1_196:.3f} ({recall_class1_196*100:.1f}%)")
    print(f"  Confusion Matrix: TN={tn_196}, FP={fp_196}, FN={fn_196}, TP={tp_196}")
    
    print("\n2.5σ Threshold:")
    print(f"  Overall Accuracy: {accuracy_25_sigma:.4f} ({accuracy_25_sigma*100:.1f}%)")
    print(f"  Class 0 (No Event) - Precision: {precision_class0_25:.3f} ({precision_class0_25*100:.1f}%), Recall: {recall_class0_25:.3f} ({recall_class0_25*100:.1f}%)")
    print(f"  Class 1 (Event)    - Precision: {precision_class1_25:.3f} ({precision_class1_25*100:.1f}%), Recall: {recall_class1_25:.3f} ({recall_class1_25*100:.1f}%)")
    print(f"  Confusion Matrix: TN={tn_25}, FP={fp_25}, FN={fn_25}, TP={tp_25}")
    
    print("="*70)
    
    return {
        '196_sigma': {
            'accuracy': accuracy_196_sigma,
            'class0_precision': precision_class0_196,
            'class0_recall': recall_class0_196,
            'class1_precision': precision_class1_196,
            'class1_recall': recall_class1_196,
            'confusion_matrix': cm_196
        },
        '25_sigma': {
            'accuracy': accuracy_25_sigma,
            'class0_precision': precision_class0_25,
            'class0_recall': recall_class0_25,
            'class1_precision': precision_class1_25,
            'class1_recall': recall_class1_25,
            'confusion_matrix': cm_25
        }
    }


if __name__ == "__main__":
    threshold_crosing_test()