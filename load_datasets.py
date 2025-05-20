import datasets
import torchvision.transforms as transforms
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
from torch.utils.data import random_split, Subset

BATCH_SIZE = 32
CLIENT_NUM = 5
DATA_DIR = "chest_xray"
def partition_dataset(dataset, partition_id):
    #Determines the size of the dataset and the subsets
    division_count = CLIENT_NUM
    total_size = len(dataset)
    subset_size = total_size // division_count
    remainder = total_size % division_count
    #Calculates how the dataset will be split based on the previously determined sizes
    lengths = []
    for n in range(division_count):
        lengths.append(subset_size)
    lengths[division_count-1] = lengths[division_count-1] + remainder
    generator = torch.Generator().manual_seed(42)
    random_dataset = random_split(dataset, lengths, generator=generator)[partition_id]
    return random_dataset

def partition_dataset_non_iid_2(dataset, partition_id):
    # Determines the size of the dataset and the subsets
    division_count = int(CLIENT_NUM/2)
    total_size = len(dataset)
    subset_size = total_size // division_count
    remainder = total_size % division_count
    # Calculates how the dataset will be split based on the previously determined sizes
    lengths = []
    for n in range(division_count):
        lengths.append(subset_size)
    lengths[division_count-1] = lengths[division_count-1] + remainder
    generator = torch.Generator().manual_seed(42)
    # Finds all indices of a particular label
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if dataset[i]['label'] == partition_id%2:
            indices.append(i)
    # Splits the dataset and filters by indices of a particular label
    random_dataset=random_split(dataset, lengths, generator=generator)[int(partition_id/2)]
    matching_indices = [random_dataset.indices[i] for i in range(len(random_dataset))
        if random_dataset.indices[i] in indices]
    subset = Subset(dataset, matching_indices)
    return subset

def load_datasets(partition_id: int):
    pytorch_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), v2.RGB(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #pytorch_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), v2.RGB(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Loads train data
    dataset_dict_train = load_dataset("imagefolder", data_dir="./"+DATA_DIR+"/train")
    dataset_length = int(len(dataset_dict_train["train"]))
    dataset_train = dataset_dict_train["train"].select(range(dataset_length)) #This can be used to select part of the dataset
    # Loads test data
    dataset_dict_test = load_dataset("imagefolder", data_dir="./"+DATA_DIR, split="test")
    dataset_length = int(len(dataset_dict_test))
    dataset_test = dataset_dict_test.select(range(dataset_length)) #This can be used to select part of the dataset
    # Used to apply transform to dataset
    def apply_transforms(batch):
        batch['image'] = [pytorch_transforms(img) for img in batch['image']]
        batch['label'] = [label for label in batch['label']]
        return batch

    # Partition dataset and then applies transform
    if partition_id == -1:
        # Uses full train set for evaluation set
        train_partition = dataset_train
        train_partition.set_transform(apply_transforms)
        dataset_test.set_transform(apply_transforms)
    else:
        # Uses partition for clients
        train_partition = partition_dataset(dataset_train, partition_id)
        #train_partition = partition_dataset_non_iid(dataset_train, partition_id)
        train_partition.dataset.set_transform(apply_transforms)
        dataset_test.set_transform(apply_transforms)


    # Create train/val for each partition and wrap it into DataLoader
    trainloader = DataLoader(
        train_partition, batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(dataset_test, batch_size=BATCH_SIZE)
    testloader = DataLoader(dataset_test, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader