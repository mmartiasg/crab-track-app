import json


class Config:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as config_file:
            self.config = json.load(config_file)

    def add_new_key(self, key, value):
        self.config[key] = value

    def save(self, config_file_path):
        with open(config_file_path, "w") as config_file:
            json.dump(self.config, config_file)

    def get_value(self, key):
        return self.config[key]
