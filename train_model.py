import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import os
from model.neural_network import ResumeClassifier
from utils.text_processor import clean_text, tokenize, text_to_sequence

def train():
    """
    Trains the resume screening model (Requirement 4)
    """
    # Load dataset
    if not os.path.exists('data/resume_data.csv'):
        print("Dataset not found. Please ensure data/resume_data.csv exists.")
        return

    df = pd.read_csv('data/resume_data.csv')
    
    # Preprocessing
    roles = df['Role'].unique().tolist()
    label_to_idx = {role: i for i, role in enumerate(roles)}
    
    # Save label mapping
    os.makedirs('model', exist_ok=True)
    with open('model/label_encoder.json', 'w') as f:
        json.dump(label_to_idx, f)
        
    print(f"Roles found: {roles}")
    
    # Build vocabulary
    all_tokens = []
    for resume in df['Resume']:
        all_tokens.extend(tokenize(clean_text(resume)))
        
    vocab = sorted(list(set(all_tokens)))
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    word_to_idx["<PAD>"] = 0
    word_to_idx["<UNK>"] = len(word_to_idx)
    
    # Save vocabulary
    with open('model/vocab.json', 'w') as f:
        json.dump(word_to_idx, f)
        
    # Prepare data for PyTorch
    X = []
    y = []
    max_len = 100
    
    for _, row in df.iterrows():
        tokens = tokenize(clean_text(row['Resume']))
        seq = text_to_sequence(tokens, word_to_idx, max_len=max_len)
        X.append(seq)
        y.append(label_to_idx[row['Role']])
        
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    # Initialize Model
    vocab_size = len(word_to_idx)
    model = ResumeClassifier(vocab_size=vocab_size, num_classes=len(roles))
    
    # Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training started...")
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
            
    # Save final model
    torch.save(model.state_dict(), 'model/resume_model.pth')
    print("Model saved to model/resume_model.pth")

if __name__ == "__main__":
    train()
