import importlib
import logging


class MInterface():
    def __init__(self, conf):
        # self.lightning_model
        self.conf = conf
        self.lightning_model = self.init_lightning_model(self.conf.method_name)
        self.model = self.instancialize_lightning_model(self.lightning_model, self.conf)

    def init_lightning_model(self, name):
        return getattr(importlib.import_module(f'model.{name}.lightning_model'), f'{name}_Lightning_Model')

    def instancialize_lightning_model(self, model, conf):
        return model(conf)