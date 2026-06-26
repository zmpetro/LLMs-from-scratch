import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)

        # Apply causal attention mask
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)
d_in = 3
d_out = 2

# Define sample inputs
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # --> [2, 6, 3]

context_length = batch.shape[1]

num_heads = 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, num_heads)
context_vecs_mha = mha(batch)
print("context_vecs_mha.shape:", context_vecs_mha.shape)
print(context_vecs_mha)

assert context_vecs_mha.shape[-1] == 4

"""
Exercise: Change the input args so that the output context vecs are 2D instead
of 4D while keeping num_heads=2
"""
d_out = 1  # Set d_out to 1 instead of 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, num_heads)
context_vecs_mha = mha(batch)
print("context_vecs_mha.shape:", context_vecs_mha.shape)
print(context_vecs_mha)

assert context_vecs_mha.shape[-1] == 2
