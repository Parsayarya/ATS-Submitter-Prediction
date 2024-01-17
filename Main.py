import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, make_scorer, roc_curve, auc, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import joblib

def load_data(file_path):
    """
    Load data from a CSV file and preprocess it.

    Args:
    file_path (str): Path to the CSV file containing the data.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    df = df[(df['Type'] == "wp") | (df['Type'] == "ip")]
    df.reset_index(drop=True)
    return df

def preprocess_submitted_by(df):
    """
    Preprocess the 'Submitted By' column in the DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with 'Submitted By' column preprocessed.
    """
    df['Submitted By'] = df['Submitted By'].apply(lambda x: x.split(', '))
    return df

def preprocess_text_data(df, threshold=10):
    """
    Preprocess text data using TF-IDF and filter labels based on a threshold.

    Args:
    df (pd.DataFrame): Input DataFrame with text data and labels.
    threshold (int): Threshold value for label frequency filtering.

    Returns:
    pd.DataFrame: Filtered DataFrame for training and testing.
    """
    y = mlb.fit_transform(df['Submitted By'])
    label_frequencies = y.sum(axis=0)
    valid_labels = [label for label, frequency in enumerate(label_frequencies) if frequency >= threshold]

    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.25, random_state=42)

    X_train_filtered = X_train[y_train[:, valid_labels].any(axis=1)]
    y_train_filtered = y_train[y_train[:, valid_labels].any(axis=1)][:, valid_labels]

    X_test_filtered = X_test[y_test[:, valid_labels].any(axis=1)]
    y_test_filtered = y_test[y_test[:, valid_labels].any(axis=1)][:, valid_labels]

    return X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered

def create_pipeline():
    """
    Create a pipeline with a TF-IDF vectorizer and a OneVsRestClassifier with a RandomForestClassifier.

    Returns:
    sklearn.pipeline.Pipeline: Model pipeline.
    """
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', OneVsRestClassifier(RandomForestClassifier(
            n_estimators=60, max_features=None, class_weight='balanced_subsample',
            min_samples_split=2, min_samples_leaf=5, max_depth=500, random_state=42, n_jobs=-1
        )))
    ])
    return pipeline

def evaluate_model(pipeline, X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered):
    """
    Train the model, predict probabilities, and evaluate performance.

    Args:
    pipeline (sklearn.pipeline.Pipeline): Model pipeline.
    X_train_filtered (pd.Series): Filtered training text data.
    y_train_filtered (pd.DataFrame): Filtered training labels.
    X_test_filtered (pd.Series): Filtered testing text data.
    y_test_filtered (pd.DataFrame): Filtered testing labels.
    """
    pipeline.fit(X_train_filtered, y_train_filtered)

    predicted_probabilities = pipeline.predict_proba(X_test_filtered)

    confusion_matrices = multilabel_confusion_matrix(y_test_filtered, predicted_probabilities.round())

    for i, label in enumerate(valid_labels):
        print(f"Confusion Matrix for class {mlb.classes_[label]}:")
        print(confusion_matrices[i])

    fig, ax = plt.subplots(figsize=(40, 20))
    for i, label in enumerate(valid_labels):
        fpr, tpr, _ = roc_curve(y_test_filtered[:, i], predicted_probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(mlb.classes_[label], roc_auc))

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc='lower right')
    plt.show()

    y_train_pred = pipeline.predict(X_train_filtered)

    recall_score_train = recall_score(y_train_filtered, y_train_pred, average='macro')

    threshold = 0.4
    predicted_probabilities = pipeline.predict_proba(X_test_filtered)
    y_test_pred = (predicted_probabilities >= threshold).astype(int)
    recall_score_test = recall_score(y_test_filtered, y_test_pred, average='macro')
    f1 = f1_score(y_test_filtered, y_test_pred, average='macro')

    print("Training Recall:", recall_score_train)

if __name__ == "__main__":
    file_path = 'Data/FinalCorpus4.csv'
    df = load_data(file_path)
    df = preprocess_submitted_by(df)

    mlb = MultiLabelBinarizer()
    tfidf = TfidfVectorizer(max_features=10000, norm="l2", smooth_idf=True, stop_words='english')

    X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = preprocess_text_data(df)

    pipeline = create_pipeline()

    evaluate_model(pipeline, X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered)

    joblib.dump(pipeline, 'ModelNO2RFC.pkl')
