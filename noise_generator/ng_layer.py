import torch
import torch.nn as nn
import torch.nn.functional as F

class Masked_NG_VLToken_CA(nn.Module):
    def __init__(self, feat_dim: int, n_heads: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_heads = n_heads
        self.head_dim = feat_dim // n_heads
        
        assert self.head_dim * n_heads == feat_dim, "feat_dim must be divisible by n_heads"
        
        self.Wq = nn.Linear(feat_dim, feat_dim)
        self.Wk = nn.Linear(feat_dim, feat_dim)
        self.Wv = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

        self.fc_mean = nn.Linear(feat_dim, feat_dim)
        self.fc_variance = nn.Linear(feat_dim, feat_dim)

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_variance.weight)

    def _create_mask(self, image_split_list, text_split_list, device):
        
        v_indices = torch.cat([torch.full((s,), i, device=device) for i, s in enumerate(image_split_list)])
        l_indices = torch.cat([torch.full((s,), i, device=device) for i, s in enumerate(text_split_list)])
        mask = (v_indices.unsqueeze(1) != l_indices.unsqueeze(0))
        return mask  # [sum_p, bt]

    def forward(self, V_token, L_token, image_split_list, text_split_list):
        # V_token: [sum_p, d], L_token: [bt, d]
        device = V_token.device
        bt, _ = L_token.shape
        sum_p = V_token.size(0)
        
        mask = self._create_mask(image_split_list, text_split_list, device)  # [sum_p, bt]
                 
        Q = self.Wq(V_token)  # [sum_p, d]
        K = self.Wk(L_token)  # [bt, d]
        V_lang = self.Wv(L_token)  # [bt, d]
        
        Q = Q.view(sum_p, self.n_heads, self.head_dim).transpose(0, 1)  # [h, sum_p, d/h]
        K = K.view(bt, self.n_heads, self.head_dim).transpose(0, 1)     # [h, bt, d/h]
        V_lang = V_lang.view(bt, self.n_heads, self.head_dim).transpose(0, 1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [h, sum_p, bt]
        
        mask = mask.unsqueeze(0).expand(self.n_heads, -1, -1)  # [h, sum_p, bt]
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # [h, sum_p, bt]
        
        output = torch.matmul(attn_weights, V_lang)  # [h, sum_p, d/h]
        output = output.transpose(0, 1).contiguous().view(sum_p, -1)  # [sum_p, d]
        
        output = self.out_proj(output)

        mu = self.fc_mean(output)
        log_var = self.fc_variance(output)

        return mu, log_var, attn_weights
    

    def sample(self, mu, variance, num=1):
        # noise = noise.reshape(batch_size, num, dim)
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var*noise + m
        return noise

