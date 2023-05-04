import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """load and merge messages and categories data

    Args:
        messages_filepath (_type_): messages datapath
        categories_filepath (_type_): categories datapath

    Returns:
        dataframe: merged dataframe from two data files
    """
    # load dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge dataset
    df = pd.merge(messages, categories, on="id")
    
    return df


def clean_data(df):
    """clean given dataframe

    Args:
        df (pandas dataframe): input dataframe

    Returns:
        pandas dataframe: cleaned dataframe
    """

    categories = df['categories'].str.split(expand=True, pat=";")
    # select the first row of the categories dataframe
    row = categories.iloc[0,:].values

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [f[:-2] for f in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    df['related'] = df['related'].astype("int")

    return df


def save_data(df, database_filename):
    """save dataframe into sqlitedatabase

    Args:
        df (dataframed): input dataframe
        database_filename (string): database filename
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response_db_table', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
