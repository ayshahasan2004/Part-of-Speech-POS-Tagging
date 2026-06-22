import torch
import torch.nn as nn
from torch.utils.data import Dataset

class POSDataset(Dataset):
    # convert POS into numerical indices using shared vocabulary maps (word2idx and tag2idx) 
    def __init__(self, sentences, word2idx, tag2idx):
        self.sentences = sentences
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words, tags = self.sentences[idx]
        word_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]  # fallback to <UNK> if word is unseen
        tag_ids = [self.tag2idx[t] for t in tags]          # convert POS tags to indices
        
        return torch.tensor(word_ids), torch.tensor(tag_ids)


def collate_fn(batch):
    #handle padding and batching
    #pads variable-length sequences with 0 (<PAD>) to match the longest sentence in the batch
    sequences, tags = zip(*batch)
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=0)
    return padded_seqs, padded_tags

# 
if __name__ == "__main__":
    print("--- Starting Dataset and Padding Verification Test  ---")
    #add new words and specific joint POS tags 
    word2idx = {"<PAD>": 0, "<UNK>": 1, "the": 2, "stock": 3, "fell": 4, "prices": 5, "rose": 6, ".": 7}
    tag2idx = {"<PAD>": 0, "DET-DT": 1, "NOUN-NN": 2, "VERB-VBD": 3, "NOUN-NNS": 4, ".": 5}
    

    # create 2 new mock sentences with different lengths (4 tokens and 3 tokens)
    # sentence 1: "the stock fell ."
    # sentence 2: "prices rose ."
    mock_sentences = [
        (["the", "stock", "fell", "."], ["DET-DT", "NOUN-NN", "VERB-VBD", "."]),
        (["prices", "rose", "."], ["NOUN-NNS", "VERB-VBD", "."])
    ]
    print("mock dataset created with real corpus sentences (lengths 4 and 3)")
    
    #instantiate dataset and simulate the DataLoader collate process
    dataset = POSDataset(mock_sentences, word2idx, tag2idx)
    batch_samples = [dataset[0], dataset[1]]
    
    padded_words, padded_labels = collate_fn(batch_samples)
    print("Applied custom collate_fn padding logic")
    print(f"\nPadded Words Shape: {padded_words.shape} (Expected: torch.Size([2, 4]))")
    print(f"Padded Words Tensor:\n{padded_words}")
    
    print(f"\nPadded Tags Shape: {padded_labels.shape} (Expected: torch.Size([2, 4]))")
    print(f"Padded Tags Tensor:\n{padded_labels}")
    
    # verification check: max length is 4, sentence 2 (length 3) must end with a 0 <PAD> token
    if padded_words.shape == torch.Size([2, 4]) and padded_words[1, 3] == 0:
        print("\n[SUCCESS] Padding and Batching (Step 4) verified successfully with new cases!")
    else:
        print("\n[FAILED] Padding logic error encountered.")