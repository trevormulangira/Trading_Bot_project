from sklearn.metrics import classification_report, confusion_matrix
def inspect_mistakes(y_test, predictions):
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))
