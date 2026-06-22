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

