import pickle
import re
import sys

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_response_db_table", con=engine.connect())
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


def tokenize(text):
    # lower case
    text = text.lower()
    # strip all punctuation
    text = re.sub(r"[^\w\d]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = nltk.word_tokenize(text)
    return tokens


def build_model():
    pipeline = Pipeline([
    ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # pipeline1.get_params().keys()
    parameters_grid = {
                'classifier__estimator__n_estimators': [10]}


    cv = GridSearchCV(pipeline, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
