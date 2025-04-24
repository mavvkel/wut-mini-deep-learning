import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class CNNBlock(nn.Module):
    def __init__(self, input_channels=41, output_channels=128):
        super(CNNBlock, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, output_channels, kernel_size=10, stride=2, padding=5)
        self.norm1d = nn.BatchNorm1d(output_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        x = self.conv1d(x)  
        x = self.norm1d(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, input_features=1, output_features=128):
        super(DenseBlock, self).__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.norm1 = nn.LayerNorm(output_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(output_features, output_features)
        self.norm2 = nn.LayerNorm(output_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, input_dim=128, nhead=8, hidden_dim=1024):
        super(TransformerBlock, self).__init__()
        
        self.encoder_layer = TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=3)
        
        self.decoder_layer = TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=3)
        
        self.layer_norm = nn.LayerNorm(input_dim)
        
        self.final_fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, memory, target):
        memory = self.transformer_encoder(memory)  
        output = self.transformer_decoder(target, memory)
        
        output = self.layer_norm(output)
        
        output = self.final_fc(output)
        
        return output

class ASRModel(nn.Module):
    def __init__(self, input_channels=1):
        super(ASRModel, self).__init__()

        self.cnn_block = CNNBlock(input_channels=41, output_channels=128)

        self.dense_block = DenseBlock(input_features=128, output_features=128)

        self.transformer_block = TransformerBlock(input_dim=128, nhead=8, hidden_dim=1024)
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)  
        x = self.cnn_block(x)  
        x = x.permute(2, 0, 1)  
        x = self.dense_block(x)
        output = self.transformer_block(x, x)
        return output[-1]
