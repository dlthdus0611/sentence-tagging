import json
import os

class Configure(object):
    def __init__(self, config=None, config_json_file=None):
        if config_json_file:
            assert os.path.isfile(config_json_file), "Error: Configure file not exists!!"
            with open(config_json_file, 'r') as fin:
                self.dict = json.load(fin)
            self.update(self.dict)
        if config:
            self.update(config)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __contains__(self, key):
        return key in self.dict.keys()

    def add(self, k, v):
        self.__dict__[k] = v

    def items(self):
        return self.dict.items()

    def update(self, config):
        assert isinstance(config, dict), "Configure file should be a json file and be transformed into a Dictionary!"
        for k, v in config.items():
            if isinstance(v, dict):
                config[k] = Configure(v)
            elif isinstance(v, list):
                config[k] = [Configure(x) if isinstance(x, dict) else x for x in v]
        self.__dict__.update(config)
