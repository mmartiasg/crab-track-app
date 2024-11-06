import yaml

class Config:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    @property
    def get_config(self):
        return self.config
