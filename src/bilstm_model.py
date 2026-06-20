import torch
import torch.nn as nn

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):

        super(BiLSTMTagger, self).__init__()
        # create embedding layer to convert word IDs into vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # padding index = 0 so padding tokens are ignored during training
              
        # create bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,  # number of LSTM hidden units
            num_layers=1,   # use one LSTM layer
            bidirectional=True,    # process sequence in both forward and backward directions
            batch_first=True  # batch dimension comes first: (batch_size, seq_length, features)
        )
        
        # fully connected layer for tag prediction
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)   # hidden_dim * 2 because outputs come from both directions

    def forward(self, sentences):
        
        embeds = self.embedding(sentences) # pass the embedding sentences 
       
        lstm_out, _ = self.lstm(embeds) # pass embeddings through the BiLSTM layer
        
        tag_space = self.hidden2tag(lstm_out) #generate tag scores for every word in the sequence
        
        return tag_space  # return predicted scores for each token
    

#testging 
if __name__ == "__main__":
    
    #dummy dimensions for the test
    vocab_size = 5000   # 5000 unique words
    tagset_size = 17  #17 POS tags
    embedding_dim = 64
    hidden_dim = 256
    
    #creates the neural network object matching the new layer size
    model = BiLSTMTagger(vocab_size, tagset_size, embedding_dim, hidden_dim)
    
    # dummy input batch (1 sentence containing 5 words)
    dummy_input = torch.tensor([[10, 450, 3200, 88, 5]])
    print(f"sentence input created. Shape: {dummy_input.shape} (Batch Size=1, Sentence Length=5)")
    
    #pass the dummy input through the models forward pass
    # computes the network operations with the new vector dimensions
    predictions = model(dummy_input)
    print("forward pass executed through Embedding, BiLSTM, and Linear layers")
    
    #final output tensor shape
    print(f"output predictions tensor shape: {predictions.shape}") #must be (1, 5, 17) -> 1 sentence, 5 words, 17 possible scores per word
    
    #verification 
    if predictions.shape == torch.Size([1, 5, 17]):
        print("\nthe BiLSTM model is working correctly ")
    else:
        print("\nunexpected output shape")