import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Description:
        Loading data from database
    Arguments:
        database_filepath: the file path for database
    Returns:
        X, y: pandas dataframes for X (independent variable) and y (dependent variable)
        category_names: names for the catagories.
    """
    engine = create_engine("sqlite:///%s"%database_filepath)
    df = pd.read_sql("clean_msg", engine)
    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    y=y.astype(int)
    #print(y.head(10))

    category_names = y.columns
    return X, y, category_names

stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
punctuation_words = str.maketrans('', '', string.punctuation)
def tokenize(text):
    """
    Description:
        For a dataframe with massages, do following procedures.
        - Normalization to lowercase.
        - Remove punctuation characters.
        - Tokenization,lemmatization, and stop word removal.

    Arguments:
        text: text
    Returns:
        tokens: a list of tockenized text.
    """
    # normalize case and remove punctuation
    text = text.translate(punctuation_words).lower()
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Description:
        Building ML pipline with grid search
    Arguments:
        None
    Returns:
        cv: Defined model
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('forest', MultiOutputClassifier( RandomForestClassifier() ))
    ])

    parameters = {
    'tfidf__norm': ['l1', 'l2'],
    'tfidf__sublinear_tf': [True, False]
    }

    cv_model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=4, n_jobs=-1)
    return cv_model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description:
        Evaluate the performance of the model
    Arguments:
        model: fitted models
        X_test, Y_test: pandas dataframes for testing set
        category_names: names for msg catagories
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    y_pred_t = np.transpose(y_pred)
    i = 0
    while i<len(category_names):
        print(category_names[i])
        print(classification_report(Y_test[category_names[i]], y_pred_t[i]))
        i = i+1

def save_model(model, model_filepath):
    """
    Description:
        Save trained model to a file
    Arguments:
        model: save model object
        model_filepath: pickle file path for saving the model
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building and training model...')
        model = build_model()

        print(model)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
