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
    # setup small mock parameters
    vocab_size = 100
    tagset_size = 10  # number of possible POS tags
    word2idx = {"<PAD>": 0, "<UNK>": 1, "the": 2, "stock": 3, "fell": 4, "prices": 5, "rose": 6} #map each word to a unique index
    tag2idx = {"<PAD>": 0, "DET-DT": 1, "NOUN-NN": 2, "VERB-VBD": 3, "NOUN-NNS": 4}# map each POS tag to a unique index
    
    # create small mock training dataset used to test model quickly
    mock_train_data = [
        (["the", "stock", "fell"], ["DET-DT", "NOUN-NN", "VERB-VBD"]),
        (["prices", "rose"], ["NOUN-NNS", "VERB-VBD"])
    ]
    # create small mock validation dataset used to evaluate model
    mock_dev_data = [
        (["the", "prices", "fell"], ["DET-DT", "NOUN-NNS", "VERB-VBD"])
    ]
    # convert raw text data into PyTorch dataset format
    train_dataset = POSDataset(mock_train_data, word2idx, tag2idx)
    dev_dataset = POSDataset(mock_dev_data, word2idx, tag2idx)
    
    # generate data loaders used to load data in batches during training
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn) # shuffle training data for better learning
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)# no shuffling in validation
    print("mock dataLoaders generated successfully")
    
    # initialize Model and assign to device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create BiLSTM model for POS tagging and move it to selected device
    model = BiLSTMTagger(vocab_size, tagset_size, embedding_dim=16, hidden_dim=32).to(device)
    print("BiLSTMTagger model instantiated and moved to device")
    
    #run verification training for 3 sample epochs
    print("running short verification training pass...")
    train_model(model, train_loader, dev_loader, epochs=3, lr=0.01, device=device)
    
    print("\nTraining loop and validation tracking work perfectly")