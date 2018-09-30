import numpy as np
import torch
from torch.utils.data import Dataset


class MixupDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.beta_dist = torch.distributions.beta.Beta(0.2, 0.2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx_a = idx
        idx_b = np.random.randint(len(self))

        image_a, label_a = self.get_oneitem(idx_a)
        image_b, label_b = self.get_oneitem(idx_b)

        if label_a == label_b:
            image = image_a
            oh_label = self.onehot_encode(label_a)
        else:
            mix_rate = self.beta_dist.sample()
            if mix_rate < 0.5:
                mix_rate = 1. - mix_rate

            image = mix_rate * image_a + (1. - mix_rate) * image_b
            oh_label = mix_rate * self.onehot_encode(label_a) + (1. - mix_rate) * self.onehot_encode(label_b)

        sample = (image, label_a, oh_label)
        return sample

    def get_oneitem(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return image, label

    def onehot_encode(self, label, n_class=10):
        diag = torch.eye(n_class)
        oh_vector = diag[label].view(n_class)
        return oh_vector