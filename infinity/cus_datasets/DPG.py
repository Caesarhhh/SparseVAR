import os
import json
import torch
from torch.utils.data import Dataset

class DPGDataset(Dataset):
    def __init__(self, prompt_path):
        """
        Initialize the dataset loader with the path to the JSON file.
        
        Args:
            json_path (str): Path to the JSON file containing prompts.
        """
        self.prompt_path = prompt_path
        self.data = []
        self.load_data()

    def load_data(self):
        """
        Load the dataset from the JSON file.
        """
        try:
            filenames=os.listdir( self.prompt_path)
            for filename in filenames:
                with open(f"{self.prompt_path}/{filename}") as f:
                    line=f.readlines()[0]
                self.data.append(
                    {
                        "prompt":line,
                        "filename":filename
                    }
                )
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {e}")

    def __len__(self):
        """
        Return the total number of prompts in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a specific prompt by its index.

        Args:
            idx (int): The index of the prompt to retrieve.

        Returns:
            dict: The prompt at the specified index.
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[idx]