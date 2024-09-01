import numpy as np
import pandas as pd
import random
import json
import os
from tqdm import tqdm
from data.raw.person import Customer, Fraudster
from config import n_customers, n_fraudsters, n_sessions, \
    fraud_session_rate, raw_data_output_dir, save_formats, seed
from pathlib import Path


# set seed
random.seed(seed)
np.random.seed(seed)


# generate sessions
def generate_sessions(customers, fraudsters):
    """
    Generate sessions for customers and fraudsters.

    Parameters
    ----------
    customers : list of Customer
    fraudsters : list of Fraudster
    """

    print("Generating sessions...")
    sessions = []
    for _ in tqdm(range(n_sessions)):
        # fraud case
        if np.random.uniform() < fraud_session_rate:
            user = np.random.choice(fraudsters)
        # legit case
        else:
            user = np.random.choice(customers)
        session = user.make_session()
        sessions.append(session)
    return sessions


def flatten_sessions(sessions):
    """
    Flatten a list of sessions into a dataframe of events.

    Parameters
    ----------
    sessions : list of dict
        Each dict is a session.
    """

    dfs = []
    for session in sessions:
        dfs.append(pd.DataFrame.from_records(session))
    return pd.concat(dfs)


def generate_customer_info_table(customers):
    """
    Generate a table of customer info.

    Parameters
    ----------
    customers : list of Customer
    """

    records = [c.make_customer_info_record() for c in customers]
    df = pd.DataFrame.from_records(records)
    return df


def run():
    """
    Generate mock data and save to disk.
    """
    # generate customers (legit and fraud)
    customers = [Customer(i) for i in range(n_customers)]
    fraudsters = [Fraudster() for _ in range(n_fraudsters)]

    # generate sessions
    sessions = generate_sessions(customers, fraudsters)
    if 'csv' in save_formats or 'json' in save_formats:
        # flatten sessions into a dataframe
        df = flatten_sessions(sessions)
        # generate customer info table
        cust_df = generate_customer_info_table(customers)

    # save to disk
    print("Saving data to disk...")
    path = Path(raw_data_output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # save to csv
    if 'csv' in save_formats:
        df.to_csv(path / "events.csv", index=False)
        cust_df.to_csv(path / "customer_info.csv", index=False)
    
    # save to json with all keys on all records
    if 'json_full' in save_formats:
        df.to_json(path / "events_fulls.json", orient="records", lines=True)
    
    # save to json with only keys from each record
    if 'json' in save_formats:
        file_path = path / "events.json"
        # remove file if it already exists
        if os.path.exists(file_path):
            os.remove(file_path)
        # save to json
        with open(file_path, "a") as outfile:
            for session in tqdm(sessions):
                for event in session:
                    json.dump(event, outfile)
                    outfile.write('\n')

    if 'json' in save_formats or 'json_full' in save_formats:
        # save customer info table
        cust_df.to_json(path / "customer_info.json", orient="records", lines=True)


if __name__ == "__main__":
    run()
