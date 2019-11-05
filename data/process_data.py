# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Description:
        Loading .csv files into dataframes
    Arguments:
        messages_filepath: file path for disaster_messages.csv
        categories_filepath: file path for disaster_categories.csv
    Returns:
        messages, categories_raw: two dataframes.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories_raw = pd.read_csv(categories_filepath)

    return messages, categories_raw

def clean_data(messages, categories_raw):
    """
    Description:
        Merge messages and catagories data into one dataframe and clean data, including:
        - Splite columns into multiples
        - Proper naming
        - Convert values to binary
        - Drop duplicates
    Arguments:
        messages, categories_raw: two dataframes
    Returns:
        df: cleand dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = categories_raw.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.map(lambda x: str(x)[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: str(x)[-1])

    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    categories["id"] = categories_raw.id

    # merge dataframes
    df = messages.merge(categories, how='outer', on=['id'])
    df= df.sort_values(['id'])

    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)

    return df


def save_data(df, database_filename):
    """
    Description:
        Save processed data into database.
    Arguments:
        df: clean dataframe
        database_filename: the file path for datavase file
    Returns:
        None
    """
    engine = create_engine('sqlite:///%s'%database_filename)
    df.to_sql('clean_msg', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories_raw = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories_raw)

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
