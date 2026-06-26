import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


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

# Instantiate and use SelfAttention v1
sa_v1 = SelfAttention_v1(d_in, d_out)
sa_v1_out = sa_v1(inputs)
print(sa_v1(inputs))

# Instantiate and use SelfAttention v2
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v2_out = sa_v2(inputs)
print(sa_v2(inputs))

assert not torch.equal(sa_v1_out, sa_v2_out)

"""
At this point, the outputs between sa_v1 and sa_v2 were different because
nn.Linear uses a different weight initialization scheme than
nn.Parameter(torch.rand()), even with the same seed.  The exercise is to
correctly assign the weights from an instance of SelfAttention_v2 to an instance
of SelfAttention_v1, and prove they produce the same output afterwards, to
demonstrate the relationship between the weights in both versions.
"""

# Instantiate both versions of SelfAttention
sa_v1_new = SelfAttention_v1(d_in, d_out)
sa_v2_new = SelfAttention_v2(d_in, d_out)

# Assign the weights from sa_v2_new to sa_v1_new
# The solution is that nn.Linear stores the weight matrix in a transposed form
sa_v1_new.W_query = nn.Parameter(sa_v2_new.W_query.weight.T)
sa_v1_new.W_key = nn.Parameter(sa_v2_new.W_key.weight.T)
sa_v1_new.W_value = nn.Parameter(sa_v2_new.W_value.weight.T)

# Demostrate they produce the same output now
sa_v1_new_out = sa_v1_new(inputs)
sa_v2_new_out = sa_v2_new(inputs)
print(sa_v1_new_out)
print(sa_v2_new_out)

assert torch.equal(sa_v1_new_out, sa_v2_new_out)
