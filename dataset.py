from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import os

class PTDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the class folders.
        """
        self.root_dir = root_dir
        self.data_files = []
        self.labels = []
        self.class_to_idx = {}
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset and assigns integer labels to each class folder, 
        excluding the folder named '????' without affecting the label range.
        """
        class_folders = [d for d in sorted(os.listdir(self.root_dir)) if os.path.isdir(os.path.join(self.root_dir, d)) and d != '????']
        
        for idx, class_name in enumerate(class_folders):
            class_folder = os.path.join(self.root_dir, class_name)
            
            # Assign an integer label for this class
            self.class_to_idx[class_name] = idx
            
            # Load all .pt files in the current class folder
            for file_name in os.listdir(class_folder):
                if file_name.endswith('.pt'):
                    file_path = os.path.join(class_folder, file_name)
                    self.data_files.append(file_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        data = torch.load(data_path)  # Load the .pt file
        label = self.labels[idx]  # Get the corresponding label
        return data, label

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        data = torch.load(data_path)  # Load the .pt file
        label = self.labels[idx]  # Get the corresponding label
        return data, label

class POWDataset(Dataset):
    def __init__(self, audio_dir, csv_file, target_length=5.0, sample_rate=16000, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            audio_dir (string): Directory with all the audio files.
            target_length (float): Target length in seconds for padding/truncation.
            sample_rate (int): Sample rate for loading audio.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(os.path.join(audio_dir, csv_file))
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.transform = transform

        # Filter out species with less than 2 examples
        species_counts = self.data['Species'].value_counts()
        valid_species = species_counts[species_counts >= 2].index
        self.data = self.data[self.data['Species'].isin(valid_species)]

        # Create a mapping from species to integer labels
        self.species_to_idx = {species: idx for idx, species in enumerate(valid_species)}
        self.data['label'] = self.data['Species'].map(self.species_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row for the current sample
        row = self.data.iloc[idx]
        
        # Load the audio file
        file_path = os.path.join(self.audio_dir, row['Filename'])
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            sr = self.sample_rate

        # Calculate the start and end frames
        start_frame = int(row['Begin Time (s)'] * sr)
        end_frame = int(row['End Time (s)'] * sr)

        # Extract the segment of the audio
        segment = waveform[:, start_frame:end_frame]

        # Pad or truncate to the target length (5 seconds in this case)
        target_num_samples = int(self.target_length * sr)
        if segment.size(1) < target_num_samples:
            # Pad with zeros if the segment is too short
            padding = target_num_samples - segment.size(1)
            segment = torch.nn.functional.pad(segment, (0, padding))
        elif segment.size(1) > target_num_samples:
            # Truncate if the segment is too long
            segment = segment[:, :target_num_samples]

        # Get the label (integer) for the species
        label = row['label']

        if self.transform:
            segment = self.transform(segment)

        return segment, label