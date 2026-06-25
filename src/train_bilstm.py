# src/train_bilstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

# import previously created modules and shared files
from data_loader import load_conll, get_vocab_and_tag_sets
from vocab import build_word2idx, build_tag2idx
from bilstm_model import BiLSTMTagger
from train_utils import POSDataset, collate_fn
from evaluation import evaluate, print_report # import shared evaluation metrics created by aysha

def train_model(model, train_loader, train_eval_loader, dev_loader, epochs, lr, device, idx2tag=None, train_sentences=None, dev_sentences=None):
    
    #training loop with verification and validation accuracy tracking
    # define loss function and optimizer
    # ignore_index=0 to make sure that <PAD> tokens do not contribute to the loss or gradients

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # parameters for step 6 early stopping
    best_val_accuracy = 0.0
    patience = 2
    patience_counter = 0
    
    print(f"Starting Training for {epochs} epochs on device: {device}\n")
    
    for epoch in range(epochs):
        # trainning 
        model.train() # this line automatically activates dropout layers during training phase
        total_train_loss = 0
        for sentences, tags in train_loader:
            sentences, tags = sentences.to(device), tags.to(device)
            optimizer.zero_grad() #clear accumulated gradients
            outputs = model(sentences)   # compute predicted outputs
            
            #reshape outputs to (batch_size * seq_len, tagset_size) and tags to (batch_size * seq_len)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
            
            # Backward pass and optimization step
            loss.backward()  #compute gradients using backpropagation
            
            # step 6: apply gradient clipping to prevent exploding gradients on full text datasets
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step() # update model weights using the optimizer
            
            total_train_loss += loss.item() # add current batch loss to total training loss
            
        avg_train_loss = total_train_loss / len(train_loader)  # calculate the average training loss across all batches to measure the models overall performance during the epoch  
                                                                # loss decreases over time -> the model is learning because weights are updating correctly to minimize error
                                                                # loss stays high -> the training performance is poor because the model is stuck or parameters are wrong
                                                                # loss becomes close to zero -> the model predictions are very accurate on the training samples
        
        # validation phase 
        model.eval() # this line automatically deactivates dropout layers for stable validation evaluation
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

        # step 6: early stopping logic implementation based on development set performance
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0 # reset counter because we found a better model performance
        else:
            patience_counter += 1 # no improvement detected
            
        if patience_counter >= patience:
            print(f"\n[Early Stopping] triggered at epoch {epoch+1}. validation accuracy did not improve for {patience} consecutive epochs.")
            break

    # step 7 & 8: running advanced unified evaluation reports if full datasets are active
    if idx2tag is not None and train_sentences is not None and dev_sentences is not None:
        model.eval()
        
        # 1. generate full training set evaluation predictions using non-shuffled eval loader
        print("\nrunning predictions on full training dataset for joint evaluation report...")
        train_predicted_sequences = []
        with torch.no_grad():
            for sentences, tags in train_eval_loader:
                sentences = sentences.to(device)
                outputs = model(sentences)
                predictions = torch.argmax(outputs, dim=-1).cpu().tolist()
                for i, pred_seq in enumerate(predictions):
                    actual_length = (tags[i] != 0).sum().item()
                    real_pred_tags = [idx2tag[idx] for idx in pred_seq[:actual_length]]
                    train_predicted_sequences.append(real_pred_tags)
        train_results = evaluate(train_sentences, train_predicted_sequences)
        print_report(train_results, title="BiLSTM Model — Training Set Full Metrics (Step 8)")

        # 2. generate full validation set evaluation predictions
        print("\nrunning predictions on full validation dataset for joint evaluation report...")
        dev_predicted_sequences = []
        with torch.no_grad():
            for sentences, tags in dev_loader:
                sentences = sentences.to(device)
                outputs = model(sentences)
                predictions = torch.argmax(outputs, dim=-1).cpu().tolist()
                for i, pred_seq in enumerate(predictions):
                    actual_length = (tags[i] != 0).sum().item()
                    real_pred_tags = [idx2tag[idx] for idx in pred_seq[:actual_length]]
                    dev_predicted_sequences.append(real_pred_tags)
        bilstm_results = evaluate(dev_sentences, dev_predicted_sequences)
        print_report(bilstm_results, title="BiLSTM Model — Dev Set Full Metrics (Step 8)")

if __name__ == "__main__":
    # choose pipeline: true for real full dataset training , false for quick mock verification test
    run_full_dataset = True
    
    if not run_full_dataset:
        print("--- starting training loop verification test ---")
        # [mock code remains identical for quick verification if switched]
        pass
    else:
        print("--- starting full dataset training and evaluation pipeline ---")
        # load full dataset tokens from corpus files
        print("loading full corpus dataset files...")
        train_sentences = load_conll("data/en-universal-train.conll")
        dev_sentences = load_conll("data/en-universal-dev.conll")
        
        # build shared vocabulary and tag maps from raw data
        vocab, tags = get_vocab_and_tag_sets(train_sentences)
        word2idx = build_word2idx(vocab)
        tag2idx = build_tag2idx(tags)
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}
        
        print(f"dataset loaded. train size: {len(train_sentences)} | dev size: {len(dev_sentences)}")
        
        # convert raw text datasets into custom PyTorch dataset format
        train_dataset = POSDataset(train_sentences, word2idx, tag2idx)
        dev_dataset = POSDataset(dev_sentences, word2idx, tag2idx)
        
        # generate data loaders for the full deep learning dataset
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        # critical fix: create a specific non-shuffled loader for training set evaluation to align with original sentences array
        train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        print("full dataset dataLoaders generated successfully")
        
        # initialize full capacity model on selected device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BiLSTMTagger(vocab_size=len(word2idx), tagset_size=len(tag2idx), embedding_dim=100, hidden_dim=128).to(device)
        print("BiLSTMTagger model instantiated with real project dimensions and moved to device")
        
        # run full training pipeline and generate the unified evaluation report
        train_model(
            model=model,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            dev_loader=dev_loader,
            epochs=5,
            lr=0.001,
            device=device,
            idx2tag=idx2tag,
            train_sentences=train_sentences,
            dev_sentences=dev_sentences
        )