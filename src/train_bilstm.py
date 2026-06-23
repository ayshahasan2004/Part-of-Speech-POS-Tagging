import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# import previously created modules and shared files
from bilstm_model import BiLSTMTagger
from train_utils import POSDataset, collate_fn

def train_model(model, train_loader, dev_loader, epochs, lr, device):
    
    #training loop with verification and validation accuracy tracking
    # define loss function and optimizer
    # ignore_index=0 to make sure that <PAD> tokens do not contribute to the loss or gradients

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting Training for {epochs} epochs on device: {device}\n")
    
    for epoch in range(epochs):
        # trainning 
        model.train()
        total_train_loss = 0
        for sentences, tags in train_loader:
            sentences, tags = sentences.to(device), tags.to(device)
            optimizer.zero_grad() #clear accumulated gradients
            outputs = model(sentences)   # compute predicted outputs
            
            #reshape outputs to (batch_size * seq_len, tagset_size) and tags to (batch_size * seq_len)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
            
            # Backward pass and optimization step
            loss.backward()  #compute gradients using backpropagation
            optimizer.step() # update model weights using the optimizer
            
            total_train_loss += loss.item() # add current batch loss to total training loss
            
        avg_train_loss = total_train_loss / len(train_loader)  # calculate the average training loss across all batches to measure the models overall performance during the epoch  
                                                                # loss decreases over time -> the model is learning
                                                                # loss stays high -> the training performance is poor
                                                                # loss becomes close to zero -> the model predictions are very accurate
        
        # validation phase 
        model.eval()
        correct_tokens = 0 #the number of correctly predicted tokens
        total_tokens = 0  #the total number of valid tokens
        
        with torch.no_grad():    # disable gradient calculation during validation , just to evaluate the performance
            for sentences, tags in dev_loader:      #each validation batches
                sentences, tags = sentences.to(device), tags.to(device)   # move input data and labels to the selected device to make sure input data and labels
                                                                          # are on the same device as the model (CPU/GPU) to avoid device mismatch errors
                
                outputs = model(sentences) # generate predictions from the model
                predictions = torch.argmax(outputs, dim=-1) #select the tag with the highest score for each token
                
                mask = (tags != 0)  # ignore padding tokens (tag index 0) so we dont count them in accuracy
                
                correct_tokens += ((predictions == tags) & mask).sum().item()  # count correct predictions only for valid tokens
                total_tokens += mask.sum().item() # count total valid tokens
                
        val_accuracy = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0    #calculate validation accuracy as a percentage
        
        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%") #training loss and validation accuracy for the current epoch

if __name__ == "__main__":
    print("--- starting training loop verification test ---")