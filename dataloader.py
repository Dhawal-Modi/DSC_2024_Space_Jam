import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from cardiac_ml_tools import *
from torch.utils.data import Dataset
import torch


class EcgActivation(Dataset):
    def normalize_ecg(self, ecg):
        normalized_ecg = np.zeros_like(ecg)
        for i in range(ecg.shape[0]):
            lead_min = np.min(ecg[i])
            lead_max = np.max(ecg[i])
            lead_range = lead_max - lead_min

            if lead_range != 0:
                normalized_ecg[i] = (ecg[i] - lead_min) / lead_range
            else:
                normalized_ecg[i] = ecg[i] - lead_min  # If the lead is constant, just center it at 0
        return normalized_ecg

    def __getitem__(self, idx):
        ecg_file, vm_file = self.file_pairs[idx]

        # Load ECG data
        ecg_data = np.load(ecg_file)
        ecg_data = get_standard_leads(ecg_data)  # Convert to 12-lead
        ecg_data = self.normalize_ecg(ecg_data)

        ecg_data = ecg_data.T

        # Load and process Vm data
        vm_data = np.load(vm_file)

        ActTime = get_activation_time(vm_data)
        ActTime = ActTime.T

        vm_data = self.normalize_ecg(vm_data)
        vm_data = vm_data.T

        return torch.FloatTensor(ecg_data), torch.FloatTensor(vm_data), torch.FloatTensor(ActTime)

    def __init__(self, file_pairs):
        self.file_pairs = file_pairs


if __name__ == "__main__":
    data_dirs = []
    regex = r'data_hearts_dd_0p2*'

    DIR = 'intracardiac_dataset'  # This should be the path to the intracardiac_dataset, it can be downloaded using data_science_challenge_2023/download_intracardiac_dataset.sh
    for x in os.listdir(DIR):  # os.listdir gets all the folders in a directory
        if re.match(regex, x):
            data_dirs.append(DIR + x)
    file_pairs = read_data_dirs(data_dirs)
    print('Number of file pairs: {}'.format(len(file_pairs)))
    # example of file pair
    print("Example of file pair:")
    print("{}\n{}".format(file_pairs[0][0], file_pairs[0][1]))
