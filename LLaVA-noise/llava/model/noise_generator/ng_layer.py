import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Cross_MultiAttention(nn.Module):
    def __init__(self, token_num, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.token_num = token_num
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads


        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.out_proj = nn.Linear(emb_dim, emb_dim)


    def forward(self, V_token, L_token, pad_mask=None):
        '''
        :param V_token: [batch_size, c, h, w]
        :param L_token: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, n, d = V_token.shape

        Q = self.Wq(V_token) 
        K = self.Wk(L_token) 
        V = self.Wv(L_token)
        

        Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2) 
        V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        
        with torch.no_grad():
            row_sums = self.Wq.weight.abs().sum(dim=1)
            assert torch.all(row_sums > 0), "Found a row in Wq that is all zeros."
            assert not torch.any(torch.isnan(self.Wq.weight)), "Found NaN."
            assert not torch.any(torch.isinf(self.Wq.weight)), "Found Inf."
        
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K) 
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out_1 = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out_1.transpose(1, 2).contiguous().view(b, -1, self.emb_dim) 
        out = self.out_proj(out) 
        return out_1, out, att_weights


class NG_VLToken_CA(torch.nn.Module):
    """
    input:vision feature(B*token_num*emb_dim) and text feature(B*K*emb_dim) from clip
    output:mu and variance in image form
    """

    def __init__(self, V_input_dim, L_input_dim, V_output_dim, num_heads, ):
        super(NG_VLToken_CA, self).__init__()
        self.V_input_dim = V_input_dim
        self.L_input_dim = L_input_dim
        self.output_dim = V_output_dim
        self.num_heads = num_heads
        
        self.crossAttention = Cross_MultiAttention(
            token_num = self.V_input_dim,
            emb_dim = self.L_input_dim,
            num_heads = self.num_heads,
            )

        self.fc_variance = torch.nn.Linear(self.output_dim, self.output_dim)
        self.fc_mean = torch.nn.Linear(self.output_dim, self.output_dim)

    def forward(self, V_token, L_token):
        
        out_1, CA_feat,att_weights = self.crossAttention(V_token, L_token)
        mu = self.fc_mean(CA_feat)
        variance = self.fc_variance(CA_feat).abs()
       
        return mu, variance

    def sample(self, mu, variance, num=1):
        """
        Args:
            mu:
            variance:
            num:

        Returns:
            Tensor: batch_size * num * image.shape
        """
        # noise = noise.reshape(batch_size, num, dim)
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var*noise + m
        return noise