from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        print("TYPE OF DATASET THAT WAS LOAD_DATASET()-ED INSIDE INIT OF ABC DATALOADER:", type(dataset))
        self.train = dataset['train']
        print("type(self.train): ", type(self.train))
        print([x for x in self.train.keys()][:5], type([x for x in self.train.keys()][0]))
        print([x for x in self.train.values()][:5], type([x for x in self.train.values()][0][0]))
        self.val = dataset['val']
        print("type(self.val): ", type(self.val))
        print([x for x in self.val.keys()][:5])
        print([x for x in self.val.values()][:5])
        self.test = dataset['test']
        print("type(self.test): ", type(self.test))
        print([x for x in self.test.keys()][:5])
        print([x for x in self.test.values()][:5])
        self.umap = dataset['umap']
        print("umap: ", max([x for x in self.umap]))
        self.smap = dataset['smap']
        print("smap: ", max([x for x in self.smap]))
        self.user_count = len(self.umap)
        print(self.user_count)
        self.item_count = len(self.smap)
        print(self.item_count)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
