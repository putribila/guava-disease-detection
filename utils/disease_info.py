import json

INFO_PATH = "data/disease_information.json"

with open(INFO_PATH, "r") as f:
    disease_database = json.load(f)


def get_disease_info(disease_name):

    return disease_database[disease_name]
