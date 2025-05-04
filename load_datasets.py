import torchvision.transforms as transforms
from datasets import load_dataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import v2

BATCH_SIZE = 32
def load_datasets(partition_id: int):
    pytorch_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), v2.RGB(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    pytorch_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), v2.RGB(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_dict_train = load_dataset("imagefolder", data_dir="./images/train")
    dataset_train = dataset_dict_train["train"]
    dataset_dict_test = load_dataset("imagefolder", data_dir="./images", split="test")
    dataset_test = dataset_dict_test
    partitioner = IidPartitioner(num_partitions=10)
    partitioner.dataset = dataset_train

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch['image'] = [pytorch_transforms(img) for img in batch['image']]
        batch['label'] = [label for label in batch['label']]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    dataset_train.set_transform(apply_transforms)
    dataset_test.set_transform(apply_transforms)
    trainloader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True #partition_train_test["train"]
    )
    valloader = DataLoader(dataset_train, batch_size=BATCH_SIZE) #partition_train_test["test"]
    testset = dataset_test
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader