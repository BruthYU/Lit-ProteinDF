import importlib
import logging


class SInterface():
    def __init__(self, conf):
        # self.lightning_model
        self.conf = conf
        self.sampler_module = self.init_sampler_module(self.conf.method_name)
        self.sampler = self.instancialize_lightning_model(self.sampler_module, self.conf)

    def init_sampler_module(self, name):
        return getattr(importlib.import_module(f'sampler.{name}.sampler'), f'{name}_Sampler')

    def instancialize_lightning_model(self, sampler, conf):
        return sampler(conf)