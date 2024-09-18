# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
import random
import math
import re
import nltk
nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
"""
## Train, Val, Test split
"""

# %%
corpus_path = '/kaggle/input/auguste-maquet/Auguste_Maquet.txt'

# Load the entire corpus
with open(corpus_path, "r") as f:
    corpus = f.read()

def clean(data):
    data = data.lower()
    data = re.sub(r'\n|\s+', ' ', data)  # replace newline and multiple spaces with single space
    data = re.sub(r"[^a-zA-Z0-9\s,.!?;:]", "", data) 
    data = re.sub(r'[’‘]', '\'', data)  # apostrophes
    data = re.sub(r'[“”`\' ]|[–—-]', ' ', data)  # quotes and dashes
    data = data.strip()
    return data

corpus = clean(corpus)
sentences = sent_tokenize(corpus)

def add_start_and_end_tokens(sentences):
    modified_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized_sentence = word_tokenize(sentence)
        new_sentence = ['<S>'] + tokenized_sentence + ['</S>']
        modified_sentences.append(new_sentence)
    return modified_sentences

sentences = add_start_and_end_tokens(sentences)

# Calculate the adjusted split sizes
train_size = int(0.7 * len(sentences))
val_size = int(0.2 * len(sentences))
test_size = int(0.1 * len(sentences))

train_sentences = sentences[:train_size]
validation_sentences = sentences[train_size:train_size + val_size]
test_sentences = sentences[train_size + val_size:train_size + val_size + test_size]

print(f"Number of tokenized train sentences: {len(train_sentences)}")
print(f"Number of tokenized validation sentences: {len(validation_sentences)}")
print(f"Number of tokenized test sentences: {len(test_sentences)}")

# %%
"""
## Building vocab
"""

# %%
from collections import Counter

def build_vocab(sentences):
    all_words = [word for sentence in sentences for word in sentence]
    word_counts = Counter(all_words)
    vocab = set(word_counts.keys())
    return vocab

all_sentences = train_sentences
vocab = build_vocab(all_sentences)

special_tokens = ['<UNK>', '<PAD>', '<S>', '</S>']
for token in special_tokens:
    vocab.add(token)

vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")

# %%
"""
## Load glove embeddings
"""

# %%
embedding_dim = 300

def load_glove_embeddings(glove_path):
    word_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = vector

    special_tokens = ['<UNK>', '<PAD>', '<S>', '</S>']
    for token in special_tokens:
        if token in ['<S>', '</S>']:
              # Random initialization for special tokens
            word_embeddings[token] = np.random.rand(embedding_dim).astype('float32')
        elif token == '<UNK>':
              # Mean of all embeddings in GloVe for <UNK>
            all_vectors = np.array(list(word_embeddings.values()))
            mean_vector = np.mean(all_vectors, axis=0)
            word_embeddings[token] = mean_vector
        elif token == '<PAD>':
              # Zero initialization for <PAD>
            word_embeddings[token] = np.zeros(embedding_dim, dtype='float32')

    return word_embeddings

glove_path = '/kaggle/input/glove-300/glove.6B.300d.txt'
glove_embeddings = load_glove_embeddings(glove_path)

embedding_dim = len(glove_embeddings[next(iter(glove_embeddings))])
print(f"Loaded GloVe embeddings with dimension: {embedding_dim}")

# %%
"""
## Create mapping
"""

# %%
def build_mappings(vocab, glove_embeddings):
    word2index = {word: idx for idx, word in enumerate(vocab)}
    index2word = {idx: word for word, idx in word2index.items()}

    return word2index, index2word

word2index, index2word = build_mappings(vocab, glove_embeddings)

print(f"Sample word2index mapping: {list(word2index.items())[:10]}")
print(f"Sample index2word mapping: {list(index2word.items())[:10]}")

# Check if all words have corresponding embeddings
count = 0
for word in word2index.keys():
    if word not in glove_embeddings:
        count = count + 1
        # print(f"Warning: GloVe embedding missing for {word}")

# %%
"""
## Embedding Matrix
"""

# %%
len(vocab), len(word2index), len(glove_embeddings), count

embedding_dim = 300
vocab_size = len(word2index)
embeddings_matrix = np.zeros((vocab_size, embedding_dim))

for idx, word in index2word.items():
    if word in glove_embeddings:
        embeddings_matrix[idx] = glove_embeddings[word]
    else:
        unknown_embeds = glove_embeddings['<UNK>']
        embeddings_matrix[idx] = unknown_embeds

# Check the shape of the embeddings matrix
print(f"Shape of embeddings matrix: {embeddings_matrix.shape}")
embeddings_matrix = torch.tensor(embeddings_matrix, dtype=torch.float32)

# %%
len(vocab), len(word2index), len(glove_embeddings), count


# %%
"""
## Dataset
"""

# %%
def get_index(word, word2index):
    """Returns the index for the word or a special token index if not found."""
    return word2index.get(word, word2index.get('<UNK>'))

def create_context_label_dataset(sentences, index2word):
    contexts = []
    labels = []

    for sentence in sentences:
        sentence_indices = [get_index(word, word2index) for word in sentence]

        # Create context (all words except the last) and label (all words except the first)
        context = sentence_indices[:-1]
        label = sentence_indices[1:]

        contexts.append(context)
        labels.append(label)

    return contexts, labels


train_contexts, train_labels = create_context_label_dataset(train_sentences, word2index)
val_contexts, val_labels = create_context_label_dataset(validation_sentences, word2index)
test_contexts, test_labels = create_context_label_dataset(test_sentences, word2index)

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, contexts, labels):
        self.contexts = contexts
        self.labels = labels

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return torch.tensor(self.contexts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

train_dataset = TextDataset(train_contexts, train_labels)
validation_dataset = TextDataset(val_contexts, val_labels)
test_dataset = TextDataset(test_contexts, test_labels)

# collate function for padding
def collate_fn(batch):
    contexts, labels = zip(*batch)
    # Pad sequences with '<PAD>' to make all sentences in the batch the same length
    contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=word2index['<PAD>'])
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=word2index['<PAD>'])
    return contexts_padded, labels_padded

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# %%
type(embeddings_matrix)

# %%
"""
## Positional Encoding
"""

# %%
def pos_encoding(num_tokens, n_dim):
    pos_enc = np.zeros((num_tokens, n_dim))
    positions = np.arange(num_tokens)[:, np.newaxis]
    div_term = np.exp(np.arange(0, n_dim, 2) * -(np.log(10000.0) / n_dim))
    pos_enc[:, 0::2] = np.sin(positions * div_term)
    pos_enc[:, 1::2] = np.cos(positions * div_term)
    return torch.tensor(pos_enc, dtype=torch.float)

# %%
"""
## Transformer Class
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderModel(nn.Module):
    def __init__(self, embedding, vocab_size, embedding_dim, hidden_dim, num_layers,  num_heads, dropout = 0.1):
        super(TransformerDecoderModel, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.positional_encoding = pos_encoding(1000, embedding_dim).to(device)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=num_layers)
        
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, tgt, tgt_mask=None):
        tgt = self.embedding(tgt) + self.positional_encoding[:tgt.size(1)]
        tgt = self.layer_norm(tgt)
        output = self.transformer_decoder(tgt, memory=tgt, tgt_mask=tgt_mask)  # Use tgt as memory
        output = self.fc_out(self.dropout(output))
        return output
    
# Model params
embedding_dim = 300
hidden_dim = 300
num_heads = 10
num_layers = 2
dropout = 0.1
pad_idx = word2index['<PAD>']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerDecoderModel(embeddings_matrix, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads,  dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, factor=0.1)

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Training loop
        for batch_idx, (contexts, labels) in enumerate(train_loader):
            contexts, labels = contexts.to(device), labels.to(device)

            optimizer.zero_grad()
            # Forward pass through the model
            output = model(contexts)

            # Reshape output and labels for CrossEntropyLoss
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}')

        # Validation step
        model.eval()
        total_val_loss = 0
        hidden = None

        with torch.no_grad():  # Disable gradient calculation for validation
            for batch_idx, (contexts, labels) in enumerate(val_loader):
                contexts, labels = contexts.to(device), labels.to(device)

                # Forward pass through the model
                output = model(contexts)

                # Compute the validation loss
                loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'Transformer_model.pth') 
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# %%
train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, device, num_epochs = 10, patience = 3)

# %%
def calculate_test_metrics(model, test_loader, device, criterion):
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, (contexts, labels) in enumerate(test_loader):
            contexts, labels = contexts.to(device), labels.to(device)

            batch_size = contexts.size(0)
            output = model(contexts)
            
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    perplexity = math.exp(avg_test_loss)

    return avg_test_loss, perplexity

# %%
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# %%
save_model(model,"2021101102-LM3-Transformer.pth")

# %%
train_loss, train_perplexity =  calculate_test_metrics(model, train_loader, device, criterion)
val_loss, val_perplexity = calculate_test_metrics(model, validation_loader, device, criterion)
test_loss, test_perplexity = calculate_test_metrics(model, test_loader, device, criterion)

# %%
print(f"Train Loss: {train_loss:.4f}")
print(f"Train Perplexity: {train_perplexity:.4f}")

# %%
print(f"Val Loss: {val_loss:.4f}")
print(f"Val Perplexity: {val_perplexity:.4f}")

# %%
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Perplexity: {test_perplexity:.4f}")

# %%
"""
## Saving perplexities to file
"""

# %%
def save_perplexities(model, test_loader, file_name):
    
    model.eval()  
    batch_perplexities = []
    total_test_loss = 0
    
    with torch.no_grad():  
        for batch_idx, (contexts, labels) in enumerate(test_loader):
            contexts, labels = contexts.to(device), labels.to(device)
            
            output = model(contexts)

            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            total_test_loss += loss.item()

            batch_perplexity = math.exp(loss.item())
            batch_perplexities.append(batch_perplexity)

            # Write to file (append mode)
            with open(file_name, 'a') as f:
                f.write(f'Batch {batch_idx + 1}: {batch_perplexity:.4f}\n')

    # Compute the average test loss and perplexity
    avg_test_loss = total_test_loss / len(test_loader)
    avg_perplexity = math.exp(avg_test_loss)
    
    with open(file_name, 'a') as f:
        f.write(f'Average Perplexity: {avg_perplexity:.4f}\n')

    # Print average perplexity
    print(f'Average Perplexity: {avg_perplexity:.4f}')

    return avg_perplexity

# %%
train_path = "2021101102-LM3-train-perplexity.txt"
val_path = "2021101102-LM3-val-perplexity.txt"
test_path = "2021101102_LM3-test-perplexity.txt"

# Createa loaders with size 1
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

save_perplexities(model, train_loader, train_path)
save_perplexities(model, validation_loader, val_path)
save_perplexities(model, test_loader, test_path)