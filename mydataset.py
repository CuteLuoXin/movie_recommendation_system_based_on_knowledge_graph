from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, rating):
        super(Dataset, self).__init__()
        self.user = rating['user_id']
        self.movie = rating['item_id']
        self.rating = rating['rating']

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, item):
        return self.user[item], self.movie[item], self.rating[item]

