import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, query, key, value, padding_mask, look_ahead_mask=None):
        # query, key, value: (batch_size, num_heads, seq_len, d_k)
        # padding_mask shape: (batch_size, 1, 1, seq_len)
        
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(2, 3)) / (key.size(3) ** 0.5)
        # 패딩 마스크 길이를 현재 시퀀스에 맞게 슬라이딩 후 처리
        scores = scores.masked_fill(padding_mask[:, :, :, :scores.size(3)] == 0, float('-inf'))
        if look_ahead_mask is not None:
            # 현재 시퀀스 길이에 맞게 슬라이싱되도록 수정
            scores.masked_fill_(look_ahead_mask[:, :, :scores.size(2), :scores.size(3)] == 0, float('-inf'))
            
        # weights shape: (batch_size, num_heads, seq_len, seq_len)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # return shape: (batch_size, num_heads, seq_len, d_k)
        return torch.matmul(weights, value)
        
    def forward(self, Q, K, V, padding_mask, look_ahead_mask=None):
        # input shape: (batch_size, seq_len, d_model)
        
        # query, key, value shape: (batch_size, num_heads, seq_len, d_k)
        query = self.query(Q).view(Q.size(0), Q.size(1), self.num_heads, self.d_k).transpose(1, 2)
        key = self.key(K).view(K.size(0), K.size(1), self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(V).view(V.size(0), V.size(1), self.num_heads, self.d_k).transpose(1, 2)
        
        # x shape (after attention): (batch_size, num_heads, seq_len, d_k)
        x = self.scaled_dot_product_attention(query, key, value, padding_mask, look_ahead_mask)
        
        # x shape (after reshape): (batch_size, seq_len, d_model)
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # return shape: (batch_size, seq_len, d_model)
        return self.fc_out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ffn=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # x shape (after linear1): (batch_size, seq_len, d_ffn)
        x = self.linear1(x)
        x = torch.relu(x)
        # x shape (after linear2): (batch_size, seq_len, d_model)
        x = self.linear2(x)
        # return shape: (batch_size, seq_len, d_model)
        return x

class EncoderLayers(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, dropout=0.1, d_ffn=2048):
        super().__init__()
        self.num_layers = num_layers
        
        self.multi_head_attention = nn.ModuleList(
            MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.layer_norm1 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        
        self.feed_forward = nn.ModuleList(FeedForward(d_model=d_model, d_ffn=d_ffn) for _ in range(num_layers))
        self.layer_norm2 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        # x shape: (batch_size, seq_len, d_model)
        for i in range(self.num_layers):
            # 1. Multi-Head Attention
            attn_output = self.multi_head_attention[i](x, x, x, padding_mask)
            # 2. Add & Norm
            attn_output = self.layer_norm1[i](x + self.dropout(attn_output))
            
            # 3. Feed Forward
            ff_output = self.feed_forward[i](attn_output)
            # 4. Add & Norm
            x = self.layer_norm2[i](attn_output + self.dropout(ff_output))
            
        # return shape: (batch_size, seq_len, d_model)
        return x

class DecoderLayers(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, dropout=0.1, d_ffn=2048):
        super().__init__()
        self.num_layers = num_layers
        
        self.masked_multi_head_attention = nn.ModuleList(
            MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.layer_norm1 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        
        self.multi_head_attention = nn.ModuleList(
            MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.layer_norm2 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        
        self.feed_forward = nn.ModuleList(FeedForward(d_model=d_model, d_ffn=d_ffn) for _ in range(num_layers))
        self.layer_norm3 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, trg, src_padding_mask, trg_padding_mask, look_ahead_mask):
        # src shape: (batch_size, src_seq_len, d_model)
        # trg shape: (batch_size, trg_seq_len, d_model)
        
        for i in range(self.num_layers):
            # 1. Masked Multi-Head Attention (Self-Attention)
            attn_output = self.masked_multi_head_attention[i](trg, trg, trg, trg_padding_mask, look_ahead_mask) # (batch_size, trg_seq_len, d_model)
            # 2. Add & Norm
            trg_output = self.layer_norm1[i](trg + self.dropout(attn_output)) # (batch_size, trg_seq_len, d_model)
            
            # 3. Multi-Head Attention (Encoder-Decoder Attention)
            attn_output = self.multi_head_attention[i](trg_output, src, src, src_padding_mask) # (batch_size, trg_seq_len, d_model)
            # 4. Add & Norm
            output = self.layer_norm2[i](trg_output + self.dropout(attn_output)) # (batch_size, trg_seq_len, d_model)
            
            # 5. Feed Forward
            ff_output = self.feed_forward[i](output) # (batch_size, trg_seq_len, d_model)
            # 6. Add & Norm
            trg = self.layer_norm3[i](output + self.dropout(ff_output)) # (batch_size, trg_seq_len, d_model)
            
        # return shape: (batch_size, trg_seq_len, d_model)
        return trg

class Transformer(nn.Module):
    def __init__(self, input_dim=25000, seq_len=1000, d_model=512, num_heads=8, num_layers=6, dropout=0.1, d_ffn=2048, pad_idx=1, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.pad_idx = pad_idx
        self.device = device
        
        self.register_buffer('look_ahead_mask', self.generate_look_ahead_mask(seq_len))
        self.register_buffer('positional_encoding', self.position_encoding_init(d_model, seq_len))
        
        self.input_embedding = nn.Embedding(input_dim, d_model, padding_idx=pad_idx)
        self.output_embedding = nn.Embedding(input_dim, d_model, padding_idx=pad_idx)
        
        self.encoder_layer = EncoderLayers(d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout, d_ffn=d_ffn)
        self.decoder_layer = DecoderLayers(d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout, d_ffn=d_ffn)
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, input_dim)
    
    def position_encoding_init(self, d_model, seq_len):
        position = torch.arange(0, seq_len, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        # pe shape: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # return shape: (1, seq_len, d_model) - 배치 차원 추가
        return pe.unsqueeze(0)
    
    def generate_look_ahead_mask(self, size):
        # shape: (1, 1, size, size)
        return torch.tril(torch.ones((1, 1, size, size), device=self.device))
    
    def generate_padding_mask(self, seq):
        mask = (seq != self.pad_idx)
        return mask.unsqueeze(1).unsqueeze(2)

    def forward(self, src, trg):
        # src shape: (batch_size, src_seq_len)
        # trg shape: (batch_size, trg_seq_len)
        
        # 0. Generate padding mask
        src_padding_mask = self.generate_padding_mask(src)
        trg_padding_mask = self.generate_padding_mask(trg)
        
        # --- 인코더 ---
        # 1. 입력 임베딩 + 포지셔널 인코딩
        # input_embedding(src) shape: (batch_size, src_seq_len, d_model)
        # positional_encoding 슬라이싱 shape: (1, src_seq_len, d_model)
        # encoder_x shape: (batch_size, src_seq_len, d_model)
        encoder_x = self.input_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        encoder_x = self.dropout(encoder_x)
        
        # 2. 인코더 레이어 통과
        # encoder_x shape: (batch_size, src_seq_len, d_model)
        encoder_x = self.encoder_layer(encoder_x, src_padding_mask)
        
        # --- 디코더 ---
        # 1. 출력 임베딩 + 포지셔널 인코딩
        # output_embedding(trg) shape: (batch_size, trg_seq_len, d_model)
        # positional_encoding 슬라이싱 shape: (1, trg_seq_len, d_model)
        # decoder_x shape: (batch_size, trg_seq_len, d_model)
        decoder_x = self.output_embedding(trg) + self.positional_encoding[:, :trg.size(1), :]
        decoder_x = self.dropout(decoder_x)
        
        # 2. 디코더 레이어 통과
        # decoder_x shape: (batch_size, trg_seq_len, d_model)
        decoder_x = self.decoder_layer(encoder_x, decoder_x, src_padding_mask, trg_padding_mask, self.look_ahead_mask)
        
        # --- 최종 출력 ---
        # 3. Linear 레이어를 통한 단어 예측
        # x shape: (batch_size, trg_seq_len, input_dim)
        x = self.fc_out(decoder_x)
        
        return x