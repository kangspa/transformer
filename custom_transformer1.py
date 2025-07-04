import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1, mask=None, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mask = mask
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, value)
        
    def forward(self, x, encoder_output=None):
        batch_size, seq_len, _ = x.size()
        
        if encoder_output is None:
            query = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
            key = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
            value = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        else:
            query = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
            key = self.key(encoder_output).view(batch_size, encoder_output.size(1), self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
            value = self.value(encoder_output).view(batch_size, encoder_output.size(1), self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        
        x = self.scaled_dot_product_attention(query, key, value, self.mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.fc_out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ffn=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, dropout=0.1, d_ffn=2048, device='cuda'):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.multi_head_attention = nn.ModuleList(
            MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, mask=None, device='cuda')
            for _ in range(num_layers)
        )
        
        self.layer_norm1 = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(num_layers)
        )
        
        self.feed_forward = nn.ModuleList(
            FeedForward(d_model=d_model, d_ffn=d_ffn)
            for _ in range(num_layers)
        )
        
        self.layer_norm2 = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(num_layers)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for i in range(self.num_layers):
            original_x = x
            
            output = self.multi_head_attention[i](x)
            output = self.dropout(output)
            output = original_x + output
            output = self.layer_norm1[i](output)
            
            x = self.feed_forward[i](output)
            x = self.dropout(x)
            x = output + x
            x = self.layer_norm2[i](x)
            
        return x

class Decoder(nn.Module):
    def __init__(self, seq_len=1000, d_model=512, num_heads=8, num_layers=6, dropout=0.1, d_ffn=2048, device='cuda'):
        super().__init__()
        
        self.num_layers = num_layers
        
        mask = self.generate_mask(seq_len).to(device)
        self.masked_multi_head_attention = nn.ModuleList(
            MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, mask=mask, device='cuda')
            for _ in range(num_layers)
        )
        
        self.layer_norm1 = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(num_layers)
        )
        
        self.multi_head_attention = nn.ModuleList(
            MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, mask=None, device='cuda')
            for _ in range(num_layers)
        )
        
        self.layer_norm2 = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(num_layers)
        )
        
        self.feed_forward = nn.ModuleList(
            FeedForward(d_model=d_model, d_ffn=d_ffn)
            for _ in range(num_layers)
        )
        
        self.layer_norm3 = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(num_layers)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, size):
        return torch.tril(torch.ones((1, 1, size, size)))

    def forward(self, x, encoder_output):
        for i in range(self.num_layers):
            original_x = x
            
            masked_output = self.masked_multi_head_attention[i](x)
            masked_output = self.dropout(masked_output)
            masked_output = original_x + masked_output
            masked_output = self.layer_norm1[i](masked_output)
            
            output = self.multi_head_attention[i](masked_output, encoder_output)
            output = self.dropout(output)
            output = masked_output + output
            output = self.layer_norm2[i](output)
            
            x = self.feed_forward[i](output)
            x = self.dropout(x)
            x = output + x
            x = self.layer_norm3[i](x)
            
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim=25000, seq_len=1000, d_model=512, num_heads=8, num_layers=6, dropout=0.1, d_ffn=2048, pad_idx=1, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.device = device
        
        self.input_embedding = nn.Embedding(input_dim, d_model, padding_idx=pad_idx)
        self.output_embedding = nn.Embedding(input_dim, d_model, padding_idx=pad_idx)
        
        self.positional_encoding = self.position_encoding_init(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layer = Encoder(d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout, d_ffn=d_ffn, device=device)
        self.decoder_layer = Decoder(seq_len=seq_len, d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout, d_ffn=d_ffn, device=device)
        
        self.fc_out = nn.Linear(d_model, input_dim)
    
    def position_encoding_init(self, d_model):
        position = torch.arange(0, self.seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(self.seq_len, d_model, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        encoder_x = self.input_embedding(x) + self.positional_encoding[:, :x.size(1), :]
        encoder_x = self.dropout(encoder_x)
        encoder_x = self.encoder_layer(encoder_x)
        
        decoder_x = self.output_embedding(x) + self.positional_encoding[:, :x.size(1), :]
        decoder_x = self.dropout(decoder_x)
        decoder_x = self.decoder_layer(decoder_x, encoder_x)
        
        x = self.fc_out(decoder_x)
        return x
        # return torch.softmax(x, dim=-1)