from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, data):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = "Data\preprocessed\ml-1m_min_rating0-min_uc5-min_sc0-splitleave_one_out"
        self.train = data['train']
        print("type(self.train): ", type(self.train))
        print([x for x in self.train.keys()][:5], type([x for x in self.train.keys()][0]))
        print([x for x in self.train.values()][:5], type([x for x in self.train.values()][0][0]))
        self.val = data['val']
        print("type(self.val): ", type(self.val))
        print([x for x in self.val.keys()][:5])
        print([x for x in self.val.values()][:5])
        self.test = data['test']
        print("type(self.test): ", type(self.test))
        print([x for x in self.test.keys()][:5])
        print([x for x in self.test.values()][:5])
        self.umap = data['umap']
        print("umap: ", min([x for x in self.umap]), max([x for x in self.umap]))
        self.smap = data['smap']
        print("smap: ", min([x for x in self.smap]), max([x for x in self.smap]))
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
