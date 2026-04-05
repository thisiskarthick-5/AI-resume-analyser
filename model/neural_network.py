import torch
import torch.nn as nn
import torch.nn.functional as F

class ResumeClassifier(nn.Module):
    """
    Neural network for predicting job roles from resume embeddings.
    (Requirement 4)
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, num_classes=4):
        super(ResumeClassifier, self).__init__()
        # PyTorch Embedding layer (Requirement 3)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Simple Linear Layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Input: Numerical word indices
        Output: Predicted job role scores
        """
        # x shape: (batch_size, max_seq_len)
        embedded = self.embedding(x)
        # pooled shape: (batch_size, embedding_dim)
        # Average pooling across sequence length (dimension 1)
        pooled = torch.mean(embedded, dim=1)
        
        # Pass through hidden layer
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        
        # Output layer
        logits = self.fc2(x)
        return logits
