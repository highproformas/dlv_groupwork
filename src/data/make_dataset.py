# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from imblearn.under_sampling import RandomUnderSampler


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    SEED = 1337
    df = pd.read_csv(os.path.join(input_filepath, 'creditcard.csv'))
    df = df.drop('Time',axis=1)
    X = df.drop('Class',axis=1).values 
    y = df['Class'].values
    rus = RandomUnderSampler(random_state=SEED)
    X, y = rus.fit_sample(X, y)
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=SEED, test_size=0.2)
    files = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test}
    for k, v in files.items():
        pd.DataFrame(v).to_csv(os.path.join(output_filepath, f'{k}.csv'), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
