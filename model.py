import torch
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader
import copy
from sklearn.metrics import confusion_matrix


#------------------------------------------------------------------------------
# Define the MobileNetV2 model
#------------------------------------------------------------------------------
def create_mobilenetv2(data, device):
    # parameters
    if data == 'MNIST':
        num_class = 10
        num_channel = 1
    elif data == 'CIFAR10':
        num_class = 10  
        num_channel = 3

    # init a pretrained mobilenetv2
    model = models.mobilenet_v2(pretrained=True)
    
    # Modify the first convolutional layer to accept 1 input channel (for grayscale images)
    if num_channel == 1:
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    # Modify the classifier to have N output units (for N classes)
    model.classifier[1] = nn.Linear(model.last_channel, num_class)
    
    # Add softmax activation to the classifier
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Softmax(dim=1)
    )
    
    #print(model)
    return model.to(device)


#------------------------------------------------------------------------------
# Federated Learning Client Local Training
#------------------------------------------------------------------------------
def client_train(client_id, model, data, device, compute_latency, upload_latency, start_time, epochs=5, reg_lambda=None):

    # If L2 regularization is required, clone the global model to compare with
    global_model = None
    if reg_lambda is not None:
        global_model = copy.deepcopy(model)
        global_model.to(device)
    
    model = copy.deepcopy(model)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loader = DataLoader(data, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
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

    # Calculate total time with latency considerations
    training_time = (compute_latency * len(data)) + upload_latency
    end_time = start_time + training_time

    return client_id, model.state_dict(), end_time


#------------------------------------------------------------------------------
# aggregate the client models
#------------------------------------------------------------------------------    
def aggregate_models(global_model, client_updates, staleness=None):
    # synchronous FL
    if staleness is None:
        # get all key/value (layer name, weights) of global model
        global_state = global_model.state_dict()
        
        # for each layer (key of the state dictionary)
        for key in global_state.keys():
            # Ensure that the averaging is done on floating point tensors
            if global_state[key].dtype in [torch.float32, torch.float64, torch.float16]:
                # averaged all the corresponding layers in the client_updates
                global_state[key] = torch.stack([client_updates[i][key].float() for i in range(len(client_updates))], dim=0).mean(dim=0)
                
        # load the new weights into global model
        global_model.load_state_dict(global_state) 
    
    # asynchronous FL
    else:
        # Weight is inverse relationship with staleness
        weight = 1.0 / (1.0 + staleness) 
        
        # init new state dict
        new_model_state = copy.deepcopy(global_model.state_dict())
        
        # for asynchronous FL, there is only one client update at a time
        update = client_updates
        
        # for each layer (key of the state dict)
        for key in new_model_state:
            # weighted average
            new_model_state[key] = (1 - weight) * new_model_state[key] + weight * update[key]
            
        # load the new weights into global model
        global_model.load_state_dict(new_model_state)        


#------------------------------------------------------------------------------
# Evaluation global model
#------------------------------------------------------------------------------
def evaluate_global_model(model, test_loader, device, cm=False):
    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store all labels and predictions for confusion matrix calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Global model accuracy: {accuracy:.2f}%")

    if not cm:
        return accuracy

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, conf_matrix
