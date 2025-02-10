from typing import Sequence
import torch
from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    def __init__(self, dataset:Dataset):
        self.original_dataset = dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        return (self.original_dataset[idx], idx)
    
class IndexedRepeatedDataset(IndexedDataset):
    def __init__(self, dataset:Dataset, repeats:int):
        super().__init__(dataset)
        self.repeats = repeats
    
    def __getitem__(self, idx):
        data_list = []
        label_list = []
        for _ in range(self.repeats):
            (data, target) = self.original_dataset[idx]
            data_list.append(data)
            label_list.append(target)
        return ((torch.stack(data_list), torch.tensor(label_list)), idx)

class TagedRepeatedDataset(IndexedRepeatedDataset):
    def __init__(self, dataset:Dataset, repeats:int, tags:Sequence[int]=None):
        super().__init__(dataset, repeats)
        if tags:
            if len(tags) != len(dataset):
                raise ValueError("Length of tags must be equal to length of dataset")
        self.tags = tags
    
    def __getitem__(self, idx):
        if self.tags is None:
            return super().__getitem__(idx)
        data_list = []
        label_list = []
        for _ in range(self.repeats):
            (data, target) = self.original_dataset[idx]
            data_list.append(data)
            label_list.append(target)
        return ((torch.stack(data_list), torch.tensor(label_list)), self.tags[idx])
    
class TaggedDataset(IndexedDataset):
    def __init__(self, dataset:Dataset, tags:Sequence[int]):
        super().__init__(dataset)
        if len(tags) != len(dataset):
            raise ValueError("Length of tags must be equal to length of dataset")
        self.tags = tags
    
    def __getitem__(self, idx):
        return (self.original_dataset[idx], self.tags[idx])
    
class InMemoryDataset(Dataset):
    def __init__(self, dataset):
        """
        :param file_paths: Data file path list
        :param transform: Data enhancement or transformation methods
        """
        self.data = [dataset[idx] for idx in range(len(dataset))]  # Load all data into memory at once
        self.length = len(dataset)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.length