from pathlib import Path
from datetime import datetime
import flammkuchen as fl
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import calcium_event_classifier as cec



def logistic_regression():
    """
    Train and evaluate a logistic regression baseline on raw dFF traces.
    
    Returns
    -------
    lr_model : LogisticRegression
        Fitted logistic regression model with optimized hyperparameters.
    X_train : np.ndarray
        Training data, shape (n_train_samples, trace_length).
    X_valid : np.ndarray
        Validation data, shape (n_valid_samples, trace_length).
    y_train : np.ndarray
        Training labels, shape (n_train_samples,).
    y_valid : np.ndarray
        Validation labels, shape (n_valid_samples,).
    lr_prob : np.ndarray
        Predicted probabilities for class 1 on validation set, shape (n_valid_samples,).
    lr_pred : np.ndarray
        Binary predictions (0 or 1) on validation set, shape (n_valid_samples,).
    """
    
    # Get today's date YYMMDD
    today = datetime.now().strftime("%y%m%d")

    data_path = Path(
        r"C:/Users/dcupolillo/Projects/calcium_event_classifier/"
        r"datasets/251114_dataset.h5")
    data = fl.load(data_path)

    dataset = cec.DffDataset(
        data,
        augment=False,
    )

    X = np.array([dataset[i][0].squeeze().numpy() for i in range(len(dataset))])
    y = np.array([dataset[i][1].item() for i in range(len(dataset))])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'penalty': ['l2'],             # or ['l1', 'l2'] depending on solver
        'solver': ['liblinear'],       # required for L1 penalty
    }
    grid_search = GridSearchCV(
        LogisticRegression(class_weight='balanced', random_state=42),
        param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    # Best estimator is already fitted during grid search
    lr_model = grid_search.best_estimator_

    train_acc = accuracy_score(y_train, lr_model.predict(X_train))
    valid_acc = accuracy_score(y_valid, lr_model.predict(X_valid))
    
    print("="*60)
    print("Logistic Regression Performance (on raw dFF traces)")
    print("="*60)
    print(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"Valid Accuracy: {valid_acc:.4f} ({valid_acc*100:.1f}%)")
    print("="*60)
    
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_valid, lr_model.predict(X_valid), 
                                target_names=["No Event", "Event"]))

    lr_prob = lr_model.predict_proba(X_valid)[:, 1]
    lr_pred = lr_model.predict(X_valid)

    # Save the model
    save_path = Path(
        r"C:/Users/dcupolillo/Projects/calcium_event_classifier/"
        rf"models/{today}_LR_model.h5")

    fl.save(save_path, {
        'model': lr_model,
        'best_params': grid_search.best_params_,
        'train_accuracy': train_acc,
        'valid_accuracy': valid_acc,
        'dataset_path': str(data_path),
        'date': today,
    })
    
    print(f"\nModel saved to: {save_path}")

    return lr_model, X_train, X_valid, y_train, y_valid, lr_prob, lr_pred