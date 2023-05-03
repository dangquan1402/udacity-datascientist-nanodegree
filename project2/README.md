# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


## File descriptions

```
app
├── run.py
└── templates
    ├── go.html
    └── master.html
```

```
data
├── categories.csv
├── disaster_response_db.db
├── messages.csv
└── process_data.py
```


```

```

## Components

### ETL Pipeline
`process_data.py`, prepare data for training machine learning model, including following steps:
- loads and merges data from messages and categories field
- cleans data
- store data into sqlite

`train_classifier.py`, writes machine learning pipeline includes:
- loads data from sqlite database
- splits loaded data into train and test set
- builds machine learning pipeline
- tunes hyperparameters using `GridSearchCV`
- evaluates model on test set
- dumps trained model to a pickle file

## Flask app
