import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import time

from data import get_dataset
from model import create_mobilenetv2, client_train, aggregate_models, evaluate_global_model

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
# Global variables
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
ROUNDS = 15
GLOBAL_MODEL = None
DATA = 'CIFAR10'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for random, NumPy, and PyTorch
seed = 49
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# If using CUDA, set the seed for all GPU operations as well
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU setups


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
# get dataset
client_datasets, test_loader = get_dataset(data=DATA, num_clients=NUM_CLIENTS)



#------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------
# Main function
def main():
    # instantiate the global model
    global GLOBAL_MODEL
    GLOBAL_MODEL = create_mobilenetv2(data=DATA, device=DEVICE)

    # Assign latency characteristics to each client
    compute_latencies = [random.choice([0.25, 0.50, 0.75]) for _ in range(NUM_CLIENTS)]
    upload_latencies = [random.choice([1.0, 1.25, 1.75, 2.0]) for _ in range(NUM_CLIENTS)]

    total_time = 0
    all_clients = list(range(NUM_CLIENTS))

    for round_num in range(ROUNDS):
        # Select a subset of clients to participate in this round
        selected_clients = random.sample(all_clients, CLIENTS_PER_ROUND)
        client_updates = []
        client_times = []

        # Train each selected client and collect their updates
        for client_id in selected_clients:
            compute_latency = compute_latencies[client_id]
            upload_latency = upload_latencies[client_id]            
            _, client_weights, client_time = client_train(client_id, GLOBAL_MODEL, client_datasets[client_id], DEVICE, compute_latency, upload_latency, total_time)
            client_updates.append(client_weights)
            client_times.append(client_time)

        # Aggregate client updates to update the global model
        aggregate_models(GLOBAL_MODEL, client_updates)
        
        # get the maximum time and update the total_time
        longest_client_time = max(client_times)
        total_time = longest_client_time

         # Output progress
        print(f"Round {round_num + 1}/{ROUNDS} completed. Total time elapsed: {total_time / 3600:.2f} hours.")

        # Evaluate global model every 1 rounds
        if (round_num + 1) % 1 == 0:
            evaluate_global_model(GLOBAL_MODEL, test_loader, DEVICE)

    # Final evaluation after all rounds
    print("Federated learning completed!")
    evaluate_global_model(GLOBAL_MODEL, test_loader, DEVICE)

if __name__ == "__main__":
    main()
