########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction

########################################################################################################################
########## Import
########################################################################################################################

import torch
import random
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import logging
logger = logging.getLogger(__name__)


########################################################################################################################
########## Functions
########################################################################################################################


class CustomDataset(Dataset):
    def __init__(self, tri_list , shuffle=False):
        self.tri_list = tri_list

        if shuffle:
            random.shuffle(self.tri_list)

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):
        d1_samples = []
        d2_samples = []
        labels = []
        for d1, d2, label in batch:
            d1_samples.append(d1)
            d2_samples.append(d2)
            labels.append(label)

        d1_samples = Batch.from_data_list(d1_samples)
        d2_samples = Batch.from_data_list(d2_samples)
        labels = torch.tensor(labels, dtype=torch.float32)

        return d1_samples, d2_samples, labels


class CustomDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def set_log(path_output, log_message):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(path_output / log_message),
            logging.StreamHandler()
        ]
    )


def set_random_seeds(seed: int):
    pyg.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"set seed: {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)