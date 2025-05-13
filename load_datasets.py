import datasets
import torchvision.transforms as transforms
from datasets import load_dataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.partitioner import PathologicalPartitioner
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
from torch.utils.data import random_split
import numpy as np

BATCH_SIZE = 32
CLIENT_NUM = 10

def partition_dataset(dataset):
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
    return random_split(dataset, lengths, generator=generator)

def partition_dataset_split_class(dataset):
    #Determines the size of the dataset and the subsets
    division_count = int(CLIENT_NUM/2)
    #for
    total_size = len(dataset)
    subset_size = total_size // division_count
    remainder = total_size % division_count
    client1_train_indices = np.where(np.isin(dataset.targets[dataset.indices].numpy(), [1, 2, 3]))[0]
    #Calculates how the dataset will be split based on the previously determined sizes
    lengths = []
    for n in range(division_count):
        lengths.append(subset_size)
    lengths[division_count-1] = lengths[division_count-1] + remainder
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, lengths, generator=generator)


def load_datasets(partition_id: int):
    #pytorch_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), v2.RGB(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    pytorch_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), v2.RGB(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_dict_train = load_dataset("imagefolder", data_dir="./images/train")
    dataset_length = int(len(dataset_dict_train["train"])/1) #Selects half the training dataset
    print(dataset_length)
    dataset_train = dataset_dict_train["train"].select(range(dataset_length))

    dataset_dict_test = load_dataset("imagefolder", data_dir="./images", split="test")
    dataset_length = int(len(dataset_dict_test)/1)
    print(dataset_length)
    dataset_test = dataset_dict_test.shuffle(seed=22).select(range(dataset_length)) #Selects half the testing dataset
    #dataset_test = dataset_dict_test.select(range(dataset_length))  # Selects half the testing dataset
    #partitioner = IidPartitioner(num_partitions=10)
    #partitioner.dataset = dataset_train
    #train_partition = partitioner.load_partition(partition_id)
    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        #print(batch)
        batch['image'] = [pytorch_transforms(img) for img in batch['image']]
        batch['label'] = [label for label in batch['label']]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    dataset_train.set_transform(apply_transforms)
    dataset_test.set_transform(apply_transforms)
    #partitioner = IidPartitioner(num_partitions=2)
    train_partition = partition_dataset(dataset_train)[partition_id]
    #partitioner = PathologicalPartitioner(num_partitions=2,partition_by="label",num_classes_per_partition=1)
    trainloader = DataLoader(
        train_partition, batch_size=BATCH_SIZE, shuffle=True #partition_train_test["train"]
    )
    valloader = DataLoader(dataset_train, batch_size=BATCH_SIZE) #partition_train_test["test"]
    testset = dataset_test
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader