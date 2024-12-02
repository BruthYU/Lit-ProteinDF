# Lightning Workspace
Main workspace of Lit-ProteinDF, which contains the training and inference implementations of 
**Protein Structure Generation Diffusion Models**. 

## Folders and Files
<details>

- **config**: Lit-ProteinDF manages complex configuration with the [hydra](https://github.com/facebookresearch/hydra) framework. 
This folder contains default settings of integrated methods. Specifically, `train.yaml` and `inference.yaml` select method for training or inference
by setting the value `default`, corresponding configurations are loaded from the folder `config/method`. For example, if we want to run training of **FoldFlow**, we can set the `train.yaml` as
    ```yaml
    # config/train.yaml
    defaults:
      - method: foldflow
    ```
   and simply run `main_train.py` (similar to the inference).
- **data**: With loaded `lmdb` cache, methods further extract features (e.g. frame with t-step diffusion) to determine the dataloader for training and inference.
In every folder for each method, a `lightning_datamodule.py` are implemented to align the interface `ligtning/data/data_interface.py`. Note that restrict datamodule class names
are required (`{}_Lightning_Datamodule`).
   ```python
   class DInterface():
    def __init__(self, conf):
        # self.lightning_model
        self.conf = conf
        self.lightning_datamodule = self.init_lightning_datamodule(self.conf.method_name)
        self.datamodule = self.instancialize_lightning_model(self.lightning_datamodule, self.conf)

    def init_lightning_datamodule(self, name):
        return getattr(importlib.import_module(f'data.{name}.lightning_datamodule'), f'{name}_Lightning_Datamodule')

    def instancialize_lightning_model(self, datamodule, conf):
        return datamodule(conf)
   ```
- **model**: In line with the deep learning framework [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/), both model architecture and training details (e.g. training step and loss function) are placed in this folder.
In every folder for each method, a `lightning_model.py` are implemented to align the interface `ligtning/model/model_interface.py`. Note that restrict model class names
are required (`{}_Lightning_Model`).
  ```python
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
     ```
- **sampler**: For convenient usage of pre-trained model, we develop this folder supporting checkpoint loading and protein sampling.
In every folder for each method, a `sampler_module.py` are implemented to align the interface `ligtning/sampler/sampler_interface.py`. Note that restrict sampler class names
are required (`{}_Sampler`).
   ```Python
   class SInterface():
       def __init__(self, conf):
           # self.lightning_model
           self.conf = conf
           self.sampler_module = self.init_sampler_module(self.conf.method_name)
           self.sampler = self.instancialize_lightning_model(self.sampler_module, self.conf)
   
       def init_sampler_module(self, name):
           return getattr(importlib.import_module(f'sampler.{name}.sampler_module'), f'{name}_Sampler')
   
       def instancialize_lightning_model(self, sampler, conf):
           return sampler(conf)
   ```

</details>

## Tutorials

