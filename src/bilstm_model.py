import torch
import torch.nn as nn

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):

        super(BiLSTMTagger, self).__init__()
        # Padding index 0 keeps padded tokens separate from real words.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
              
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # The BiLSTM has forward and backward outputs, so the linear layer uses hidden_dim * 2.
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentences):
        
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        
        return tag_space
    

if __name__ == "__main__":
    
    vocab_size = 5000
    tagset_size = 17
    embedding_dim = 64
    hidden_dim = 256
    
    model = BiLSTMTagger(vocab_size, tagset_size, embedding_dim, hidden_dim)
    
    dummy_input = torch.tensor([[10, 450, 3200, 88, 5]])
    print(f"sentence input created. Shape: {dummy_input.shape} (Batch Size=1, Sentence Length=5)")
    
    predictions = model(dummy_input)
    print("forward pass executed through Embedding, BiLSTM, and Linear layers")
    
    print(f"output predictions tensor shape: {predictions.shape}")
    
    if predictions.shape == torch.Size([1, 5, 17]):
        print("\nthe BiLSTM model is working correctly ")
    else:
        print("\nunexpected output shape")
