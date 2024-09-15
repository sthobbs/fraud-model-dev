from config import mongo_client_uri, db_name, profile_dir, raw_data_dir
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pymongo
import json


def load_new_line_json(file_path):
    """load data from a newline deliminated JSON file"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def run():
    mongo_client = pymongo.MongoClient(mongo_client_uri)
    db = mongo_client[db_name]

    # clear collections if they already exists
    collections = ['events', 'customer_info', 'profiles']
    for col in collections:
        db[col].drop()

    # load customer_info data into MongoDB
    file_path = Path(raw_data_dir) / 'customer_info.json'
    data = load_new_line_json(file_path)
    db['customer_info'].insert_many(data)

    # load monthly profile data into MongoDB
    profile_paths = glob(f'{profile_dir}/*')
    for file_path in tqdm(profile_paths):
        data = load_new_line_json(file_path)
        db['profiles'].insert_many(data)   


    # create indexes
    db.events.create_index([("customerId", pymongo.ASCENDING),
                            ("sessionId", pymongo.ASCENDING),
                            ("action", pymongo.ASCENDING),
                            ("timestamp", pymongo.ASCENDING)])
    db.customer_info.create_index([("customerId", pymongo.ASCENDING)], unique=True)
    db.profiles.create_index([("profileDate", pymongo.ASCENDING),
                            ("customerId", pymongo.ASCENDING)], unique=True)


if __name__ == '__main__':
    run()
