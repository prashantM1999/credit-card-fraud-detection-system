from sklearn.metrics import (
    classification_report,
    roc_auc_score
)


def monitoring_summary(y_true, y_pred, y_prob):
    """
    Print key monitoring metrics.
    """

    print("ROC-AUC:", roc_auc_score(y_true, y_prob))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))