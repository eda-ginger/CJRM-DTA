import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d1_len, d2_len, embed_dim, out_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.d1_len = d1_len
        self.d2_len = d2_len
        self.embed_dim = embed_dim

        self.fc = nn.Linear(self.embed_dim, out_dim)
        self.attention = nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads)

    def forward(self, input1, input2): # input1 -> graph (after GNN), input2 -> protein sequence (after Conv1d)
        # change input shape for nn.MultiheadAttention (seq_len, batch_size, embed_dim)
        # input1 = (batch, 290==node, 121)
        # input2 = (batch, 32==channel, 121)

        input1_t = self.fc(input1.transpose(0, 1))  # (290, b, 121) -> (290, b, 256)
        input2_t = self.fc(input2.transpose(0, 1))  # (32, b, 121) -> (32, b, 256)

        # drug representation (drug -> query, prot -> key, value)
        attn_output, _ = self.attention(input1_t, input2_t, input2_t)  # (290, b, 256)

        # change to original shape
        attn_output = attn_output.transpose(0, 1)  # (b, 290, 256)

        # pooling
        output = torch.sum(attn_output, dim=1) # (b, 256)
        return output


class CoAttention(nn.Module):
    def __init__(self, d1_len, d2_len, embed_dim, out_dim, num_heads):
        super(CoAttention, self).__init__()
        self.d1_len = d1_len
        self.d2_len = d2_len
        self.embed_dim = embed_dim
        self.co_dim = out_dim // 2

        self.fc = nn.Linear(self.embed_dim, self.co_dim)
        self.attention1 = nn.MultiheadAttention(embed_dim=self.co_dim, num_heads=num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim=self.co_dim, num_heads=num_heads)

    def forward(self, input1, input2): # input1 -> graph (after GNN), input2 -> protein sequence (after Conv1d)
        # change input shape for nn.MultiheadAttention (seq_len, batch_size, embed_dim)
        # input1 = (batch, 290==node, 121)
        # input2 = (batch, 32==channel, 121)

        input1_t = self.fc(input1.transpose(0, 1))  # (290, b, 121) -> (290, b, 128)
        input2_t = self.fc(input2.transpose(0, 1))  # (32, b, 121) -> (32, b, 128)

        # drug representation (drug -> query, prot -> key, value)
        attn_output1, _ = self.attention1(input1_t, input2_t, input2_t)  # (290, b, 128)

        # protein representation (protein -> query, input1 -> key, value)
        attn_output2, _ = self.attention2(input2_t, input1_t, input1_t)  # (32, b, 128)

        # change to original shape
        attn_output1 = attn_output1.transpose(0, 1)  # (b, 290, 128)
        attn_output2 = attn_output2.transpose(0, 1)  # (b, 32, 128)

        # pooling
        attn_output1 = torch.sum(attn_output1, dim=1) # (b, 128)
        attn_output2 = torch.sum(attn_output2, dim=1) # (b, 128)

        # Concatenate along the last dimension (features)
        output = torch.cat((attn_output1, attn_output2), dim=1)  # (b, 256)
        return output


if __name__ == '__main__':
    # Parameters
    d1_len = 290
    d2_len = 32
    input1 = torch.randn(36, 290, 121)  # (batch_size, seq_len1, embed_dim)
    input2 = torch.randn(36, 32, 121)    # (batch_size, seq_len2, embed_dim)

    embed_dim = 121
    out_dim = 256
    num_heads = 8

    # Initialize the model
    model = CrossAttention(d1_len, d2_len, embed_dim, out_dim, num_heads)

    # Forward pass
    output = model(input1, input2)
    print('Cross')
    print(output.shape)  # Should output: (36,256)
    print(sum(p.numel() for p in model.parameters()))
    print()

    # Initialize the model
    model = CoAttention(d1_len, d2_len, embed_dim, out_dim, num_heads)

    # Forward pass
    output = model(input1, input2)
    print('Co')
    print(output.shape)  # Should output: (36,256)
    print(sum(p.numel() for p in model.parameters()))
    print()

    d1_len = 96
    d2_len = 96
    input1 = torch.randn(36, 96, 121)  # (batch_size, seq_len1, embed_dim)
    input2 = torch.randn(36, 96, 121)    # (batch_size, seq_len2, embed_dim)

    # Initialize the model
    model = CrossAttention(d1_len, d2_len, embed_dim, out_dim, num_heads)

    # Forward pass
    output = model(input1, input2)
    print('Cross')
    print(output.shape)  # Should output: (36,256)
    print(sum(p.numel() for p in model.parameters()))
    print()

    # Initialize the model
    model = CoAttention(d1_len, d2_len, embed_dim, out_dim, num_heads)

    # Forward pass
    output = model(input1, input2)
    print('Co')
    print(output.shape)  # Should output: (36, 256)
    print(sum(p.numel() for p in model.parameters()))
    print()
