import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


#------------------------------------------------------------------------------
# Custom function to create non-IID dataset split
#------------------------------------------------------------------------------
def create_noniid_datasets(dataset, num_clients, majority_ratio=0.8, verbose=False):
    # dataset
    targets = np.array(dataset.targets)
    indices = np.arange(len(targets))

    # specs
    num_classes = 10
    majority_split_size = num_clients // num_classes   
    minority_split_size = num_clients - (num_clients // num_classes)
    if verbose:
        print(majority_split_size, minority_split_size)

    # split into pools
    all_majority_splits = {}
    all_minority_splits = {}
    for class_id in range(num_classes):
        # indices of this class
        class_indices = indices[targets == class_id]

        # Calculate the split point
        split_point = int(len(class_indices) * majority_ratio)
    
        # Break down the array into majority and minority pool
        majority_pool = class_indices[:split_point]
        minority_pool = class_indices[split_point:]

        # sub-split into N splits
        majority_splits = np.array_split(majority_pool, majority_split_size)
        minority_splits = np.array_split(minority_pool, minority_split_size)

        # add to dict
        all_majority_splits[class_id] = majority_splits
        all_minority_splits[class_id] = minority_splits

    # create the dataset split
    client_datasets = []
    
    for class_id in range(num_classes):
        majority_splits = all_majority_splits[class_id]
        for client_id in range(majority_split_size):
            # Sample majority data
            majority_split = majority_splits.pop(0)
            combined_indices = majority_split
            # Sample minority data
            for other_class_id in range(num_classes):
                if other_class_id != class_id:
                    minority_splits = all_minority_splits[other_class_id]
                    minority_split = minority_splits.pop(0)

                    combined_indices = np.concatenate((combined_indices, minority_split))
                    
            # random shuffle
            np.random.shuffle(combined_indices)
            
            # Create client dataset
            client_dataset = torch.utils.data.Subset(dataset, combined_indices)
            client_datasets.append(client_dataset)        

    # Assertion to ensure all indices are unique across client datasets
    all_indices = np.concatenate([client.indices for client in client_datasets])
    assert len(all_indices) == len(set(all_indices)), "Indices in client_datasets are not unique."
    
    # Assertion to ensure all splits are used
    assert all(isinstance(value, list) and not value for value in all_majority_splits.values()), "Not all data is used"
    assert all(isinstance(value, list) and not value for value in all_minority_splits.values()), "Not all data is used"
        

    # for checking the variables
    if verbose:
        print(len(all_indices), len(set(all_indices)), len(client_datasets))
        print(all_majority_splits)
        print(all_minority_splits)


    return client_datasets      


#------------------------------------------------------------------------------
# get dataset
#------------------------------------------------------------------------------
def get_dataset(data, num_clients):
    # assertion
    assert data in ['MNIST', 'CIFAR10'], "dataset must be one of ['mnist', 'CIFAR10']"
    
    if data == 'MNIST':
        # Dataset preparation
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        client_datasets = create_noniid_datasets(train_data, num_clients)
        
        # Test dataset preparation
        test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
    elif data == 'CIFAR10':
        # Dataset preparation
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        client_datasets = create_noniid_datasets(train_data, num_clients)
        
        # Test dataset preparation
        test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)        
    
    return client_datasets, test_loader
    
    
    
    
