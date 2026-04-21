import torch
from torch.utils.data import Dataset
from PIL import Image

class OralCancerBinaryDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Must contain columns ['image_path', 'label']
            transform (callable, optional): Transformations for images
        """
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {"cancer": 1, "non_cancer": 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, "image_path"]
        label_name = self.data.loc[idx, "label"]

        # Convert label to numeric
        label = self.class_to_idx[label_name]

        # Load image safely
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
