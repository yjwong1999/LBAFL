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
# FL variables
#------------------------------------------------------------------------------
# Global variables
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
ROUNDS = 15 * CLIENTS_PER_ROUND
GLOBAL_MODEL = None
DATA = 'MNIST'
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
# Get dataset
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
    active_clients = random.sample(all_clients, CLIENTS_PER_ROUND)
    prev_active_clients = []
    remaining_clients = set(all_clients) - set(active_clients)

    client_updates = []
    client_last_round = {}
    round_num = 0
    while round_num != ROUNDS:      

        # Train each client and record the time taken
        for client_id in active_clients:
            if round_num == 0:
                compute_latency = compute_latencies[client_id]
                upload_latency = upload_latencies[client_id]
                _, client_weights, client_time = client_train(client_id, GLOBAL_MODEL, client_datasets[client_id], DEVICE, compute_latency, upload_latency, total_time)
                client_updates.append((client_id, client_weights, client_time))
                client_last_round[client_id] = 0

            else:
                if client_id in prev_active_clients:
                    continue
                compute_latency = compute_latencies[client_id]
                upload_latency = upload_latencies[client_id]
                _, client_weights, client_time = client_train(client_id, GLOBAL_MODEL, client_datasets[client_id], DEVICE, compute_latency, upload_latency, total_time)
                client_updates.append((client_id, client_weights, client_time))
                client_last_round[client_id] = round_num

        #print(active_clients)
        #print([x[0] for x in client_updates])
        #print([x[2] for x in client_updates])

        # Find the fastest client
        fastest_client = min(client_updates, key=lambda x: x[2])
        fastest_client_id, fastest_client_weights, fastest_client_time = fastest_client
        #print(fastest_client_id)

        # staleness + update last round
        #print(client_last_round)
        last_round = client_last_round[fastest_client_id]
        staleness = round_num - last_round
        #print(last_round, round_num, staleness)
        client_last_round[fastest_client_id] = round_num

        # remove the fastest client from client_updates
        client_updates.remove(fastest_client)

        # Replace the fastest client with a new one from the remaining pool
        new_client_id = random.choice(list(remaining_clients))
        remaining_clients.remove(new_client_id)
        remaining_clients.add(fastest_client_id)
        prev_active_clients = copy.deepcopy(active_clients)
        #active_clients = [c if c != fastest_client_id else new_client_id for c in active_clients]
        active_clients.remove(fastest_client_id)
        active_clients = active_clients + [new_client_id]

        if staleness > 5:
            print(f"Client {fastest_client_id} with staleness round {staleness}")
            continue

        # Update global time and perform aggregation
        total_time = fastest_client_time
        aggregate_models(GLOBAL_MODEL, fastest_client_weights, staleness)

        # Output progress
        print(f"Round {round_num + 1}/{ROUNDS} completed. Total time elapsed: {total_time / 3600:.2f} hours.")


        # Evaluate global model every 10 rounds
        if (round_num + 1) % 10 == 0:
            evaluate_global_model(GLOBAL_MODEL, test_loader, DEVICE)

        round_num += 1

    # Final evaluation after all rounds
    print("Federated learning completed!")
    evaluate_global_model(GLOBAL_MODEL, test_loader, DEVICE)

if __name__ == "__main__":
    main()
