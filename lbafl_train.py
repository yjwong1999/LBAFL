import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import time

from data import get_dataset, top_n_accuracy_changes
from model import create_mobilenetv2, client_train, aggregate_models, evaluate_global_model

#------------------------------------------------------------------------------
# FL variables
#------------------------------------------------------------------------------
# Global variables
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
ROUNDS = 15 * CLIENTS_PER_ROUND
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
# LBAFL variables
#------------------------------------------------------------------------------
import sys
sys.path.append('DDQN')
from DDQN.ddqn_agent import DDQNAgent

# sample random 5 clients
sample_dim = 5

# states dimension (client compute latency + upload_latency + top 2 accuracy improvement, staleness, current round, top 2 accuracy decrease)
states_dim = [int(sample_dim * 4) + 4, ]

# action is to choose from the 5 clients
action_dim = sample_dim

# init LBAFL agent
LBAFL = DDQNAgent(gamma=0.99, epsilon=0.9, lr=0.0001,
                  input_dims=states_dim, n_actions=action_dim,
                  mem_size=50000, eps_min=0.1,
                  batch_size=64, replace=25, eps_dec=5e-4,
                  chkpt_dir=f'LBAFL/', algo='DDQN',
                  env_name='LBAFL') 

# load weights for inference                    
if True:
    LBAFL.load_models()
    LBAFL.memory.mem_cntr = 0 # not learning
    LBAFL.epsilon = 0         # not learning

# training hyperparameters
DRL_ROUND = 10   


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

    #------------------------------------------------------------------------------
    # For LBAFL
    #------------------------------------------------------------------------------
    prev_acc = 0
    prev_cm = None
    top_N = 2 # top N accuracy improvement in confusion matrix
    MDP = None        
    traces_dict = {i:(0,0) for i in range(NUM_CLIENTS)}
    data_traces = {i:[-1] * top_N for i in range(NUM_CLIENTS)}    
    states_traces = {}
    action_traces = {}
    
    #------------------------------------------------------------------------------
    # For FL system
    #------------------------------------------------------------------------------
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
    
    
    # start FL
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
        
        # we do not skip for LBAFL
        # if staleness > 5:
        #     print(f"Client {fastest_client_id} with staleness round {staleness}")
        #     continue

        # Update global time and perform aggregation
        total_time = fastest_client_time
        aggregate_models(GLOBAL_MODEL, fastest_client_weights, staleness)

        # Output progress
        print(f"Round {round_num + 1}/{ROUNDS} completed. Total time elapsed: {total_time / 3600:.2f} hours.")   

        # Evaluate global model every round
        acc, cm = evaluate_global_model(GLOBAL_MODEL, test_loader, DEVICE, cm=True)     
        
        # find out top N accuracy changes
        if round_num == 0:
            # there are no prev_cm, fill with -1 first
            top_improvements = [-1] * top_N
            top_deteriorations = [-1] * top_N
        else:
            # find the top n accuracy changes                
            top_improvements, top_deteriorations = top_n_accuracy_changes(prev_cm, cm, N=top_N)   
            
        # overwrite prev_cm
        prev_cm = cm                
            
        '''
        Note that:
            top_improvements is used to describe the contribution made by client
            top_deteriorations is used to describe what the global model needs to improve            
        '''
        data_traces[fastest_client_id] = top_improvements

        
        #------------------------------------------------------------------------------
        # get LBAFL states for client selection
        #------------------------------------------------------------------------------
        # randomly select N = sample_dim clients
        sampled_ids = random.sample(list(remaining_clients), sample_dim)

        # init states        
        states = [round_num, staleness] + top_deteriorations
        for client_id in sampled_ids:                
            # We cannot directly use these values, cuz LBAFL might be first time sampling these clients
            '''
            compute_latency = compute_latencies[client_id]
            upload_latency = upload_latencies[client_id]  
            '''
            
            # get history of traces
            compute_latency, upload_latency = traces_dict[client_id]
            
            # record new trace
            traces_dict[client_id] = (compute_latencies[client_id], upload_latencies[client_id])                
                                
            # the partial observations
            partial_obs = [compute_latency, upload_latency] + data_traces[client_id]
            
            # add to states
            states += partial_obs                         
            
        
        #------------------------------------------------------------------------------
        # get LBAFL action (which is client selection)
        #------------------------------------------------------------------------------
        action = LBAFL.choose_action(states)    
        new_client_id = sampled_ids[action]      
        print('states: ', states)          
        print('action: ', action)
        print('epsilon: ', LBAFL.epsilon)
        

        # remove the new client, and add the fastest client back to remaining_clients
        remaining_clients.remove(new_client_id)
        remaining_clients.add(fastest_client_id)

        # keep track of previous active clients
        prev_active_clients = copy.deepcopy(active_clients)

        # replace the fastest client with new client
        active_clients.remove(fastest_client_id)
        active_clients = active_clients + [new_client_id]    
        
        
        # #------------------------------------------------------------------------------
        # # get LBAFL reward (to evaluate client selection)
        # #------------------------------------------------------------------------------
        # # Find the delta_acc            
        # delta_acc = 1.732 * (acc - prev_acc)
        
        # # this is previous reward, because this client is not an action of current state
        # prev_reward = delta_acc
        
        # # overwrite previous accuracy
        # prev_acc = acc            
        
        
        # #------------------------------------------------------------------------------
        # # get LBAFL MDP
        # #------------------------------------------------------------------------------
        # try:
        #     # get the prev_states and prev_action that we used to select this client back to few rounds ago
        #     prev_states = states_traces[fastest_client_id]
        #     prev_action = action_traces[fastest_client_id]
            
        #     # consturct MDP                
        #     MDP = [prev_states, prev_action, prev_reward, states]    

        #     # overwrite the states and action traces
        #     states_traces[fastest_client_id] = states
        #     action_traces[fastest_client_id] = action      
        # except:
        #     # there are not previous states and action, so we just record current states and action for this client selection
        #     states_traces[fastest_client_id] = states
        #     action_traces[fastest_client_id] = action                           
            
        
        # increment round_num
        round_num += 1

    # Final evaluation after all rounds
    print("Federated learning completed!")
    acc, cm = evaluate_global_model(GLOBAL_MODEL, test_loader, DEVICE, cm=True)                  
        
        

if __name__ == "__main__":
    main()
