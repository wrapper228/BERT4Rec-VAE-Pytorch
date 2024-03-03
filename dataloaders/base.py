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
        print(self.train.keys())
        self.val = dataset['val']
        print("type(self.val): ", type(self.val))
        print(self.val.keys())
        self.test = dataset['test']
        print("type(self.test): ", type(self.test))
        print(self.test.keys())
        assert False
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
