import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import random
import copy

# Global variables
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
ROUNDS = 150
GLOBAL_MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for random, NumPy, and PyTorch
seed = 49
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# If using CUDA, set the seed for all GPU operations as well
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

# Define the MobileNetV2 model
def create_mobilenetv2():
    model = models.mobilenet_v2(pretrained=True)
    
    # Modify the first convolutional layer to accept 1 input channel (for grayscale images)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    # Modify the classifier to have 10 output units (for 10 classes)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    
    # Add softmax activation to the classifier
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Softmax(dim=1)
    )
    
    #print(model)
    return model.to(DEVICE)

# Custom function to create non-IID dataset split
def create_noniid_datasets(dataset, num_clients, majority_ratio=0.8):
    # dataset
    targets = np.array(dataset.targets)
    indices = np.arange(len(targets))

    # specs
    num_classes = 10
    majority_split_size = num_clients // num_classes
    minority_split_size = num_clients - (num_clients // num_classes)
    print(minority_split_size, majority_split_size)

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
                    #print(len(combined_indices))
            np.random.shuffle(combined_indices)
            # Create client dataset
            client_dataset = torch.utils.data.Subset(dataset, combined_indices)
            client_datasets.append(client_dataset)        

    # Assertion to ensure all indices are unique across client datasets
    all_indices = np.concatenate([client.indices for client in client_datasets])
    assert len(all_indices) == len(set(all_indices)), "Indices in client_datasets are not unique."
    print(len(all_indices), len(set(all_indices)), len(client_datasets))



    print(all_majority_splits)
    print(all_minority_splits)


    return client_datasets            
            

# Dataset preparation
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
client_datasets = create_noniid_datasets(mnist_data, NUM_CLIENTS)

# Test dataset preparation
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Federated Learning Client
def client_train(client_id, model, data, epochs=5, reg_lambda=None): # 0.005
    """
    Trains the given model on the client data.
    
    Parameters:
        client_id (int): The ID of the client.
        model (torch.nn.Module): The model to train.
        data (torch.utils.data.Dataset): The dataset to train on.
        epochs (int, optional): Number of epochs to train. Default is 5.
    
    Returns:
        dict: The updated model state_dict.
    """
    # If L2 regularization is required, clone the global model to compare with
    global_model = None
    if reg_lambda is not None:
        global_model = copy.deepcopy(model)
        global_model.to(DEVICE)

    model = copy.deepcopy(model)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loader = DataLoader(data, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add L2 regularization if reg_lambda is provided
            if reg_lambda is not None:
                l2_reg = 0.0
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    l2_reg += torch.norm(param - global_param, p=2) ** 2
                l2_reg = reg_lambda * l2_reg
                loss += l2_reg

            loss.backward()
            optimizer.step()

    return model.state_dict()

# Aggregation function
def aggregate_models(global_model, client_updates):
    """
    Aggregates the global model using the average of client updates.

    Args:
        global_model (torch.nn.Module): The current global model that is being updated.
        client_updates (list): A list of state_dicts from client models.

    Returns:
        None: The function modifies the global model in place.
    """
    global_state = global_model.state_dict()
    for key in global_state.keys():
        # Ensure that the averaging is done on floating point tensors
        if global_state[key].dtype in [torch.float32, torch.float64, torch.float16]:
            global_state[key] = torch.stack([client_updates[i][key].float() for i in range(len(client_updates))], dim=0).mean(dim=0)
    global_model.load_state_dict(global_state)

# Evaluation function
def evaluate_global_model(model, test_loader):
    """
    Evaluates the global model's performance on a test dataset.

    Args:
        model (torch.nn.Module): The trained model that will be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader object containing the test dataset 
            with features and labels.

    Returns:
        None: This function prints out the accuracy of the global model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Global model accuracy: {accuracy:.2f}%")

# Main function
def main():
    global GLOBAL_MODEL
    GLOBAL_MODEL = create_mobilenetv2()

    for round_num in range(ROUNDS):
        # Select a subset of clients to participate in this round
        selected_clients = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        client_updates = []

        # Train each selected client and collect their updates
        for client_id in selected_clients:
            client_weights = client_train(client_id, GLOBAL_MODEL, client_datasets[client_id])
            client_updates.append(client_weights)

        # Aggregate client updates to update the global model
        aggregate_models(GLOBAL_MODEL, client_updates)

        # Output progress
        print(f"Round {round_num + 1}/{ROUNDS} completed.")

        # Evaluate global model every 10 rounds
        if (round_num + 1) % 10 == 0:
            evaluate_global_model(GLOBAL_MODEL, test_loader)

    # Final evaluation after all rounds
    print("Federated learning completed!")
    evaluate_global_model(GLOBAL_MODEL, test_loader)

if __name__ == "__main__":
    main()
