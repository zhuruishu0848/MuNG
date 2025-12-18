import torch
import torch.nn as nn
import torch.nn.functional as F

# def initialize_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             init.constant_(m.bias, 0)

class ModelTest(torch.nn.Module):
    def __init__(self, input_dim, scale=4):
        super(ModelTest, self).__init__()
        self.input_dim = input_dim
        self.scale = scale
        self._build_up()

    # def _build_up(self):
    #     self.fc1 = torch.nn.Linear(self.input_dim, self.output_dim*4)
    #     self.relu =nn.ReLU()
    #     self.fc2 = torch.nn.Linear(self.input_dim*4, self.output_dim)
    #     self.relu2 = nn.ReLU()
    #     self.fc_mean = nn.Linear(self.output_dim, self.output_dim)
    #     self.fc_variance = nn.Linear(self.output_dim, self.output_dim)

    def _build_up(self):
        self.fc1 = nn.Linear(self.input_dim, self.input_dim * self.scale)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.input_dim*self.scale, self.input_dim)
        self.relu2 = nn.ReLU()
        self.fc_variance = torch.nn.Linear(self.input_dim, self.input_dim)
        self.fc_mean = torch.nn.Linear(self.input_dim, self.input_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)    
        x = self.fc2(x)
        x = self.relu2(x)


        mu = self.fc_mean(x)
        variance = self.fc_variance(x).abs()

        return mu, variance
    
class ModelTestImage(torch.nn.Module):
    def __init__(self, image_size):
        super(ModelTestImage, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu =nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x 

class VPNGeneratorImage_VisionFeatureCNN(nn.Module):
    def __init__(self, grid_size=24, image_size=336, output_dim=1024):
        super(VPNGeneratorImage_VisionFeatureCNN, self).__init__()
        self.grid_size = grid_size
        self.image_size = image_size
        self.output_dim = output_dim
        self.tmp = int(image_size /2)
        self._build_up()

    def _build_up(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=self.grid_size*self.grid_size, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.tmp*self.tmp, self.output_dim*4)
        self.relu2 = nn.GELU()
        self.fc2 = nn.Linear(self.output_dim*4, self.output_dim)
        self.relu3 = nn.GELU()

        self.fc_variance = nn.Linear(self.output_dim, self.output_dim)
        self.fc_mean = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, image):
        x = image
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.conv2(x)  # n x (grid x grid) x image_size/2 x image_size/2
        x = x.view(-1, self.grid_size*self.grid_size, self.tmp*self.tmp)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)

        
        mu = self.fc_mean(x)
        variance = self.fc_variance(x)

        return mu, variance
    

    def sample(self, mu, variance, num=1):
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var*noise + m
        return noise

class VPNGeneratorVisionFeatureDNN(torch.nn.Module):
    """
    input:spatial feature(N*HW*D) and text feature(K*D) from clip
    output:mu and variance in image form
    """

    def __init__(self, input_dim, scale=4):
        super(VPNGeneratorVisionFeatureDNN, self).__init__()
        self.input_dim = input_dim
        self.scale = scale
        self._build_up()

    def _build_up(self):
        self.fc1 = nn.Linear(self.input_dim, self.input_dim * self.scale)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.input_dim*self.scale, self.input_dim*self.scale)
        self.relu2 = nn.ReLU()
        self.fc_variance = torch.nn.Linear(self.input_dim * self.scale, self.input_dim)
        self.fc_mean = torch.nn.Linear(self.input_dim * self.scale, self.input_dim)
        

    def forward(self, vision_feat):
        x = vision_feat
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)

        mu = self.fc_mean(x)
        variance = self.fc_variance(x).abs()
        return mu, variance

    def sample(self, mu, variance, num=1):
        '''
        encoder input: batch x (grid x grid) x input_dim

        '''
        # noise = noise.reshape(batch_size, num, dim)
        # batch x num x (grid x grid) x input_dim
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var*noise + m
        return noise

class VPNGenerator_VisionFeatureDNN(torch.nn.Module):
    """
    input:spatial feature(N*HW*D) and text feature(K*D) from clip
    output:mu and variance in image form
    """

    def __init__(self, input_dim, n_channel, image_size, patch_size):
        super(VPNGenerator_VisionFeatureDNN, self).__init__()
        self.input_dim = input_dim
        self.n_channel = n_channel
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.output_dim = n_channel * patch_size * patch_size
        self.scale = 4
        self._build_up()

    def _build_up(self):
        self.fc1 = torch.nn.Linear(self.input_dim, self.input_dim * self.scale)
        self.fc2 = torch.nn.Linear(self.input_dim * self.scale, self.input_dim * self.scale)
        self.fc_variance = torch.nn.Linear(self.input_dim * self.scale, self.output_dim)
        self.fc_mean = torch.nn.Linear(self.input_dim * self.scale, self.output_dim)

    def forward(self, spatial_feat, text_feat):
        # batch dim -> batch (grid*grid) dim
        #text_feat = text_feat.unsqueeze(1).expand_as(spatial_feat)
        # batch (grid*grid) (2*dim)
        #x = torch.cat((spatial_feat, text_feat), dim=-1)
        x = spatial_feat
        x = self.fc1(x)
        x = self.fc2(x)
        
        # batch * (grid*grid) * (channel*patch*patch)
        variance = self.fc_variance(x).abs()
        # bggcpp
        variance = variance.reshape(
            -1, self.grid_size, self.grid_size, self.n_channel, self.patch_size,
            self.patch_size
        )
        # bcgpgp
        variance = variance.permute(0, 3, 1, 4, 2, 5)
        # bchw
        variance = variance.reshape(
            variance.shape[0], variance.shape[1], self.image_size,
            self.image_size
        )

        # mu = torch.zeros(variance.shape).to(input.device)
        # batch * (grid*grid) * (channel*patch*patch)
        mu = self.fc_mean(x)
        # bggcpp
        mu = mu.reshape(
            -1, self.grid_size, self.grid_size, self.n_channel, self.patch_size,
            self.patch_size
        )
        # bcgpgp
        mu = mu.permute(0, 3, 1, 4, 2, 5)
        # bchw
        mu = mu.reshape(
            mu.shape[0], mu.shape[1], self.image_size, self.image_size
        )
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
    

class NG_VFeat_DNN(torch.nn.Module):
    """
    input:spatial feature(N*HW*D) and text feature(K*D) from clip
    output:mu and variance in image form
    """

    def __init__(self, input_dim, scale=4):
        super(NG_VFeat_DNN, self).__init__()
        self.input_dim = input_dim
        self.scale = scale
        self._build_up()

    def _build_up(self):
        self.fc1 = torch.nn.Linear(self.input_dim, self.input_dim * self.scale)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc_variance = torch.nn.Linear(self.input_dim * self.scale, self.input_dim)
        self.fc_mean = torch.nn.Linear(self.input_dim * self.scale, self.input_dim)


        # print("self.fc1",self.fc1.weight)
        # print("self.fc_variance",self.fc_variance.weight)

        # exit()

        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        # nn.init.uniform_(self.fc_variance.weight,a=0.01, b=0.05)
        # nn.init.normal_(self.fc_mean.weight, mean=0.0, std=0.02)
        

    def forward(self, visual_feat):
        x = visual_feat
        
        x = self.fc1(x)
        x = self.relu(x)

        mu = self.fc_mean(x)
        variance = self.fc_variance(x).abs()
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


# class Cross_MultiAttention(nn.Module):
#     def __init__(self, token_num, emb_dim, out_dim, num_heads, att_dropout=0.0, aropout=0.0):
#         super(Cross_MultiAttention, self).__init__()
#         self.token_num = token_num
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.num_heads = num_heads
#         self.scale = out_dim ** -0.5

#         assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
#         self.depth = out_dim // num_heads


#         self.Wq = nn.Linear(emb_dim, out_dim)
#         self.Wk = nn.Linear(emb_dim, out_dim)
#         self.Wv = nn.Linear(emb_dim, out_dim)

#         self.out_proj = nn.Linear(out_dim, out_dim)


#     def forward(self, V_token, L_token, pad_mask=None):
#         '''

#         :param V_token: [batch_size, c, h, w]
#         :param L_token: [batch_szie, seq_len, emb_dim]
#         :param pad_mask: [batch_size, seq_len, seq_len]
#         :return:
#         '''
#         b, n, d = V_token.shape

#         # V_token = self.proj_in(V_token)   # [batch_size, c, h, w] = [3, 512, 512, 512]
#         # V_token = rearrange(V_token, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

#         Q = self.Wq(V_token)  # [batch_size, token_num, emb_dim] = [3, 262144, 512]
#         K = self.Wk(L_token)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
#         V = self.Wv(L_token)
        

#         Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, token_num, depth]
#         K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
#         V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        
#         with torch.no_grad():
#             row_sums = self.Wq.weight.abs().sum(dim=1)
#             if torch.all(row_sums == 0):
#                 print("Found a row in Wq that is all zeros.")
#             if torch.any(torch.isnan(self.Wq.weight)):
#                 print("Found NaN.")
#             if torch.any(torch.isinf(self.Wq.weight)):
#                 print("Found Inf.")
#             # assert torch.all(row_sums > 0), "Found a row in Wq that is all zeros."
#             # assert not torch.any(torch.isnan(self.Wq.weight)), "Found NaN."
#             # assert not torch.any(torch.isinf(self.Wq.weight)), "Found Inf."
        
#         att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K) # [batch_size, num_heads, token_num, seq_len]
#         att_weights = att_weights * self.scale

#         if pad_mask is not None:
#             # [batch_size, token_num, seq_len] -> [batch_size, nums_head, token_num, seq_len]
#             pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
#             att_weights = att_weights.masked_fill(pad_mask, -1e9)

#         # assert not torch.any(torch.isnan(att_weights)), "Found NaN."
#         # assert not torch.any(torch.isinf(att_weights)), "Found Inf."
#         att_weights = F.softmax(att_weights, dim=-1)
#         # if att_weights.device=="cuda:1":
#             # print("att_weights",att_weights)
#         out_1 = torch.einsum('bnij, bnjd -> bnid', att_weights, V)  # [batch_size, num_heads, token_num, depth]  bfloat16
#         out = out_1.transpose(1, 2).contiguous().view(b, -1, self.out_dim)   # [batch_size, token_num, emb_dim]

#         out = self.out_proj(out)   # [batch_size, c, h, w]

#         return out_1, out, att_weights

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

        # V_token = self.proj_in(V_token)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # V_token = rearrange(V_token, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(V_token)  # [batch_size, token_num, emb_dim] = [3, 262144, 512]
        K = self.Wk(L_token)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(L_token)
        

        Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, token_num, depth]
        K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        
        with torch.no_grad():
            row_sums = self.Wq.weight.abs().sum(dim=1)
            assert torch.all(row_sums > 0), "Found a row in Wq that is all zeros."
            assert not torch.any(torch.isnan(self.Wq.weight)), "Found NaN."
            assert not torch.any(torch.isinf(self.Wq.weight)), "Found Inf."
        
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K) # [batch_size, num_heads, token_num, seq_len]
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # [batch_size, token_num, seq_len] -> [batch_size, nums_head, token_num, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        # assert not torch.any(torch.isnan(att_weights)), "Found NaN."
        # assert not torch.any(torch.isinf(att_weights)), "Found Inf."
        att_weights = F.softmax(att_weights, dim=-1)
        # if att_weights.device=="cuda:1":
            # print("att_weights",att_weights)
        out_1 = torch.einsum('bnij, bnjd -> bnid', att_weights, V)  # [batch_size, num_heads, token_num, depth]  bfloat16
        out = out_1.transpose(1, 2).contiguous().view(b, -1, self.emb_dim)   # [batch_size, token_num, emb_dim]

        out = self.out_proj(out)   # [batch_size, c, h, w]

        return out_1, out, att_weights

class Cross_MultiAttention_qa(nn.Module):
    def __init__(self, token_num, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention_qa, self).__init__()
        self.token_num = token_num
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads


        self.Wq = nn.Linear(self.emb_dim, self.emb_dim)
        self.Wk = nn.Linear(self.emb_dim, self.emb_dim)
        self.Wv = nn.Linear(self.emb_dim, self.emb_dim)

        self.proj_in = nn.Linear(self.token_num, self.emb_dim)
        self.proj_out = nn.Linear(self.emb_dim, self.token_num)


    def forward(self, V_token, L_token, pad_mask=None):
        '''

        :param V_token: [batch_size, c, h, w]
        :param L_token: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        # V_token = V_token.squeeze()
        # n, d = V_token.shape

        V_token = self.proj_in(V_token)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # V_token = rearrange(V_token, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(V_token)  # [batch_size, token_num, emb_dim] = [3, 262144, 512]
        K = self.Wk(L_token)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(L_token)
        

        # Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, token_num, depth]
        # K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        # V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)

        Q = Q.view(-1, self.num_heads, self.depth).transpose(0, 1)  # [batch_size, num_heads, token_num, depth]
        K = K.view(-1, self.num_heads, self.depth).transpose(0, 1)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(-1, self.num_heads, self.depth).transpose(0, 1)
        
        with torch.no_grad():
            row_sums = self.Wq.weight.abs().sum(dim=1)
            assert torch.all(row_sums > 0), "Found a row in Wq that is all zeros."
            assert not torch.any(torch.isnan(self.Wq.weight)), "Found NaN."
            assert not torch.any(torch.isinf(self.Wq.weight)), "Found Inf."
        
        att_weights = torch.einsum('nid,njd -> nij', Q, K) * self.scale # [batch_size, num_heads, token_num, seq_len]

        if pad_mask is not None:
            # [batch_size, token_num, seq_len] -> [batch_size, nums_head, token_num, seq_len]
            pad_mask = pad_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        # assert not torch.any(torch.isnan(att_weights)), "Found NaN."
        # assert not torch.any(torch.isinf(att_weights)), "Found Inf."
        att_weights = F.softmax(att_weights, dim=-1)
        # if att_weights.device=="cuda:1":
            # print("att_weights",att_weights)
        out_1 = torch.einsum('nij, njd -> nid', att_weights, V)  # [batch_size, num_heads, token_num, depth]  bfloat16
        out = out_1.transpose(0, 1).contiguous().view(-1, self.emb_dim)   # [batch_size, token_num, emb_dim]
        out = self.proj_out(out)   # [batch_size, c, h, w]

        return out, att_weights
    
class Cross_MultiAttention_VVL(nn.Module):
    def __init__(self, token_num, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention_VVL, self).__init__()
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

        # V_token = self.proj_in(V_token)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # V_token = rearrange(V_token, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(V_token)  # [batch_size, token_num, emb_dim] = [3, 262144, 512]
        K = self.Wk(L_token)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(V_token)
        

        Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, token_num, depth]
        K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        
        with torch.no_grad():
            row_sums = self.Wq.weight.abs().sum(dim=1)
            if torch.all(row_sums == 0):
                print("Found a row in Wq that is all zeros.")
            if torch.any(torch.isnan(self.Wq.weight)):
                print("Found NaN.")
            if torch.any(torch.isinf(self.Wq.weight)):
                print("Found Inf.")
            # assert torch.all(row_sums > 0), "Found a row in Wq that is all zeros."
            # assert not torch.any(torch.isnan(self.Wq.weight)), "Found NaN."
            # assert not torch.any(torch.isinf(self.Wq.weight)), "Found Inf."
        
        # att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K) # [batch_size, num_heads, token_num, seq_len]
        att_weights = torch.einsum('bnjd,bnid -> bnji', K, Q) # [batch_size, num_heads, seq_len, token_num]
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # [batch_size, token_num, seq_len] -> [batch_size, nums_head, token_num, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        # assert not torch.any(torch.isnan(att_weights)), "Found NaN."
        # assert not torch.any(torch.isinf(att_weights)), "Found Inf."
        att_weights = F.softmax(att_weights, dim=-1)
        # if att_weights.device=="cuda:1":
            # print("att_weights",att_weights)
        # out_1 = torch.einsum('bnij, bnjd -> bnid', att_weights, V)  # [batch_size, num_heads, token_num, depth]  bfloat16
        out_1 = torch.einsum('bnji, bnid -> bnjd', att_weights, V)  # [batch_size, num_heads, seq_len, depth]  bfloat16
        out = out_1.transpose(1, 2).contiguous().view(b, -1, self.emb_dim)   # [batch_size, seq_len, emb_dim]

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
        
        # variance = self.fc_variance(CA_feat).abs()
        # # bggcpp
        # variance = variance.reshape(
        #     -1, self.grid_size, self.grid_size, self.n_channel, self.patch_size,
        #     self.patch_size
        # )
        # # bcgpgp
        # variance = variance.permute(0, 3, 1, 4, 2, 5)
        # # bchw
        # variance = variance.reshape(
        #     variance.shape[0], variance.shape[1], self.image_size,
        #     self.image_size
        # )

        # # mu = torch.zeros(variance.shape).to(input.device)
        # # batch * (grid*grid) * (channel*patch*patch)
        # mu = self.fc_mean(attn_feat)
        # # bggcpp
        # mu = mu.reshape(
        #     -1, self.grid_size, self.grid_size, self.n_channel, self.patch_size,
        #     self.patch_size
        # )
        # # bcgpgp
        # mu = mu.permute(0, 3, 1, 4, 2, 5)
        # # bchw
        # mu = mu.reshape(
        #     mu.shape[0], mu.shape[1], self.image_size, self.image_size
        # )
        return out_1, CA_feat, att_weights, mu, variance

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
    
class Masked_NG_VLToken_MLP(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = None):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim or feat_dim * 2

        # 用于融合图像和语言 token 的 MLP
        # self.mlp = nn.Sequential(
        #     nn.Linear(feat_dim * 2, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, feat_dim),
        # )

        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, feat_dim)
        )    

        self.fc_mean = nn.Linear(feat_dim, feat_dim)
        self.fc_variance = nn.Linear(feat_dim, feat_dim)

        nn.init.xavier_uniform_(self.mlp[1].weight)
        nn.init.xavier_uniform_(self.mlp[4].weight)
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_variance.weight)

    def forward(self, V_token, L_token, image_split_list, text_split_list):
        """
        Args:
            V_token: [sum_p, d] - 图像 tokens
            L_token: [sum_t, d] - 文本 tokens
            image_split_list: List[int] - 每个样本图像 token 数
            text_split_list: List[int] - 每个样本文本 token 数
        """

        device = V_token.device
        output_list = []

        # 逐 sample 拼接 text 和 image tokens
        img_offset = 0
        txt_offset = 0

        for img_len, txt_len in zip(image_split_list, text_split_list):
            V_sample = V_token[img_offset:img_offset + img_len]  # [img_len, d]
            L_sample = L_token[txt_offset:txt_offset + txt_len]  # [txt_len, d]

            L_avg = L_sample.mean(dim=0, keepdim=True).expand(img_len, -1)  # [img_len, d]
            fused = torch.cat([V_sample, L_avg], dim=-1)  # [img_len, 2d]

            fused_out = self.mlp(fused)  # [img_len, d]
            output_list.append(fused_out)

            img_offset += img_len
            txt_offset += txt_len

        output = torch.cat(output_list, dim=0)  # [sum_p, d]

        mu = self.fc_mean(output)
        log_var = self.fc_variance(output)
        log_var = torch.clamp(log_var, min=-10, max=10)
        attn_weights = None

        return mu, log_var, attn_weights

    def sample(self, mu, variance, num=1):
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var * noise + m
        return noise
    


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
        # l_indices = torch.repeat_interleave(torch.arange(b, device=device), t)
        mask = (v_indices.unsqueeze(1) != l_indices.unsqueeze(0))
        return mask  # [sum_p, bt]

    def forward(self, V_token, L_token, image_split_list, text_split_list):
        # V_token: [sum_p, d], L_token: [bt, d]
        device = V_token.device
        bt, _ = L_token.shape
        sum_p = V_token.size(0)
        
        mask = self._create_mask(image_split_list, text_split_list, device)  # [sum_p, bt]
         
        # L_flat = L_token.view(-1, self.feat_dim)  # [bt, d]
        
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
        epsilon = noise
        noise = var*noise + m
        return noise
        # return noise, epsilon


class Masked_NG_VLToken_CA_wo_noise(nn.Module):
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

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _create_mask(self, image_split_list, text_split_list, device):
        
        v_indices = torch.cat([torch.full((s,), i, device=device) for i, s in enumerate(image_split_list)])
        l_indices = torch.cat([torch.full((s,), i, device=device) for i, s in enumerate(text_split_list)])
        # l_indices = torch.repeat_interleave(torch.arange(b, device=device), t)
        mask = (v_indices.unsqueeze(1) != l_indices.unsqueeze(0))
        return mask  # [sum_p, bt]

    def forward(self, V_token, L_token, image_split_list, text_split_list):
        # V_token: [sum_p, d], L_token: [bt, d]
        device = V_token.device
        bt, _ = L_token.shape
        sum_p = V_token.size(0)
        
        mask = self._create_mask(image_split_list, text_split_list, device)  # [sum_p, bt]
         
        # L_flat = L_token.view(-1, self.feat_dim)  # [bt, d]
        
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

        return output
    

class NG_vlt_CA_cat(torch.nn.Module):
    """
    input:vision feature(B*token_num*emb_dim) and text feature(B*K*emb_dim) from clip
    output:mu and variance in image form
    """

    def __init__(self, V_input_dim, L_input_dim, V_output_dim, num_heads, ):
        super(NG_vlt_CA_cat, self).__init__()
        self.V_input_dim = V_input_dim
        self.L_input_dim = L_input_dim
        self.output_dim = V_output_dim
        self.num_heads = num_heads
        
        self.crossAttention = Cross_MultiAttention(
            token_num = self.V_input_dim,
            emb_dim = self.L_input_dim,
            out_dim = self.output_dim,
            num_heads = self.num_heads,
            )

        self.fc_variance = torch.nn.Linear(self.output_dim, self.output_dim)
        self.fc_mean = torch.nn.Linear(self.output_dim, self.output_dim)
        self.dim_ch = torch.nn.Linear(576, 8)

    def forward(self, V_token, L_token):
        # out_1, CA_feat,att_weights = self.crossAttention(V_token, L_token)
        # # CA_feat_t = CA_feat.transpose(1, 2)  # [b, d, t]
        # # output_tensor_transposed = self.dim_ch(CA_feat_t)  # [b, d, n]
        # # new_CA_feat = output_tensor_transposed.transpose(1, 2)  #  [b, n, d]


        # mu = self.fc_mean(CA_feat)
        # variance = self.fc_variance(CA_feat)
        
        # return out_1, CA_feat, att_weights, mu, variance
        
        out_1, CA_feat,att_weights = self.crossAttention(V_token, L_token)
        CA_feat_t = CA_feat.transpose(1, 2)  # [b, d, t]
        output_tensor_transposed = self.dim_ch(CA_feat_t)  # [b, d, n]
        new_CA_feat = output_tensor_transposed.transpose(1, 2)  #  [b, n, d]


        mu = self.fc_mean(new_CA_feat)
        variance = self.fc_variance(new_CA_feat)
        
        return CA_feat, new_CA_feat, att_weights, mu, variance

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
    
class NG_vlt_CA_qa(torch.nn.Module):
    """
    input:vision feature(B*token_num*emb_dim) and text feature(B*K*emb_dim) from clip
    output:mu and variance in image form
    """

    def __init__(self, V_input_dim, L_input_dim, V_output_dim, num_heads, ):
        super(NG_vlt_CA_qa, self).__init__()
        self.V_input_dim = V_input_dim
        self.L_input_dim = L_input_dim
        self.output_dim = V_output_dim
        self.num_heads = num_heads
        
        self.crossAttention = Cross_MultiAttention_qa(
            token_num = self.V_input_dim,
            emb_dim = self.L_input_dim,
            num_heads = self.num_heads,
            )

        self.fc_variance = torch.nn.Linear(self.V_input_dim, self.output_dim)
        self.fc_mean = torch.nn.Linear(self.V_input_dim, self.output_dim)

    def forward(self, V_token, L_token):
        
        CA_feat, att_weights = self.crossAttention(V_token, L_token)

        mu = self.fc_mean(CA_feat)
        variance = self.fc_variance(CA_feat)
        
        return CA_feat, att_weights, mu, variance

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
    
class NG_VFeatLToken_CA(torch.nn.Module):
    """
    input:vision feature(B*token_num*emb_dim) and text feature(B*K*emb_dim) from clip
    output:mu and variance in image form
    """

    def __init__(self, V_input_dim, L_input_dim, V_output_dim, num_heads, ):
        super(NG_VFeatLToken_CA, self).__init__()
        self.V_input_dim = V_input_dim
        self.L_input_dim = L_input_dim
        self.output_dim = V_output_dim
        self.num_heads = num_heads
        
        self.crossAttention = Cross_MultiAttention(
            token_num = self.V_input_dim,
            emb_dim = self.L_input_dim,
            num_heads = self.num_heads,
            )

        self.trans = torch.nn.Linear(self.V_input_dim, self.L_input_dim)
        self.fc_variance = torch.nn.Linear(self.L_input_dim, self.output_dim)
        self.fc_mean = torch.nn.Linear(self.L_input_dim, self.output_dim)

    def forward(self, V_feat, L_token):
        V_feat = self.trans(V_feat)
        
        out_1, CA_feat,att_weights = self.crossAttention(V_feat, L_token)
        mu = self.fc_mean(CA_feat)
        variance = self.fc_variance(CA_feat).abs()

        # variance = self.fc_variance(CA_feat).abs()
        # # bggcpp
        # variance = variance.reshape(
        #     -1, self.grid_size, self.grid_size, self.n_channel, self.patch_size,
        #     self.patch_size
        # )
        # # bcgpgp
        # variance = variance.permute(0, 3, 1, 4, 2, 5)
        # # bchw
        # variance = variance.reshape(
        #     variance.shape[0], variance.shape[1], self.image_size,
        #     self.image_size
        # )

        # # mu = torch.zeros(variance.shape).to(input.device)
        # # batch * (grid*grid) * (channel*patch*patch)
        # mu = self.fc_mean(attn_feat)
        # # bggcpp
        # mu = mu.reshape(
        #     -1, self.grid_size, self.grid_size, self.n_channel, self.patch_size,
        #     self.patch_size
        # )
        # # bcgpgp
        # mu = mu.permute(0, 3, 1, 4, 2, 5)
        # # bchw
        # mu = mu.reshape(
        #     mu.shape[0], mu.shape[1], self.image_size, self.image_size
        # )
        return out_1, CA_feat, att_weights, mu, variance

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
        sigma = torch.randn_like(var).to(var.device)
        noise = var*sigma + m
        return noise

    
class VPNGeneratorImageDNN(torch.nn.Module):

    def __init__(self, input_dim, n_channel, image_size, patch_size):
        super(VPNGeneratorImageDNN, self).__init__()
        self.input_dim = input_dim
        self.n_channel = n_channel
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.output_dim = n_channel * patch_size * patch_size
        self.scale = 4
        self._build_up()

    def _build_up(self):
        self.fc1 = torch.nn.Linear(self.input_dim, self.input_dim * self.scale)
        self.gelu = nn.GELU()
        self.fc2 = torch.nn.Linear(self.input_dim * self.scale, self.input_dim * self.scale)
        self.gelu2 = nn.GELU()
        self.fc_variance = torch.nn.Linear(self.input_dim * self.scale, self.output_dim)
        self.fc_mean = torch.nn.Linear(self.input_dim * self.scale, self.output_dim)

    def forward(self, spatial_feat):

        x = spatial_feat
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu2(x)
        
        # batch * (grid*grid) * (channel*patch*patch)
        variance = self.fc_variance(x).abs()
        # bggcpp
        variance = variance.reshape(
            -1, self.grid_size, self.grid_size, self.n_channel, self.patch_size,
            self.patch_size
        )
        # bcgpgp
        variance = variance.permute(0, 3, 1, 4, 2, 5)
        # bchw
        variance = variance.reshape(
            variance.shape[0], variance.shape[1], self.image_size,
            self.image_size
        )

        # mu = torch.zeros(variance.shape).to(input.device)
        # batch * (grid*grid) * (channel*patch*patch)
        mu = self.fc_mean(x)
        # bggcpp
        mu = mu.reshape(
            -1, self.grid_size, self.grid_size, self.n_channel, self.patch_size,
            self.patch_size
        )
        # bcgpgp
        mu = mu.permute(0, 3, 1, 4, 2, 5)
        # bchw
        mu = mu.reshape(
            mu.shape[0], mu.shape[1], self.image_size, self.image_size
        )
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
    

class VPNGeneratorVisionTokenDNN(torch.nn.Module):
    """
    input:spatial feature(N*HW*D) and text feature(K*D) from clip
    output:mu and variance in image form
    """

    def __init__(self, input_dim, scale=4):
        super(VPNGeneratorVisionTokenDNN, self).__init__()
        self.input_dim = input_dim
        self.scale = scale
        self._build_up()

    def _build_up(self):
        self.fc1 = nn.Linear(self.input_dim, self.input_dim * self.scale)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(self.input_dim*self.scale, self.input_dim*self.scale)
        self.relu2 = nn.GELU()
        self.fc_variance = torch.nn.Linear(self.input_dim*self.scale, self.input_dim)
        self.fc_mean = torch.nn.Linear(self.input_dim*self.scale, self.input_dim)
        

    def forward(self, vision_feat):
        x = vision_feat
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        # x = self.fc3(x)
        mu = self.fc_mean(x)
        variance = self.fc_variance(x).abs()
        return mu, variance

    def sample(self, mu, variance, num=1):
        '''
        encoder input: batch x (grid x grid) x input_dim

        '''
        # noise = noise.reshape(batch_size, num, dim)
        # batch x num x (grid x grid) x input_dim
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var*noise + m
        return noise


# if __name__ == "__main__":   
    # def remove_prefix_from_state_dict(state_dict, prefix):
    #     """
    #     移除 state_dict 中所有键的指定前缀。
    #     """
    #     new_state_dict = {}
    #     for key in state_dict.keys():
    #         if key.startswith(prefix):
    #             # 去掉前缀并重新命名键
    #             new_key = key[len(prefix) + 1:]  # +1 是为了去掉下划线 _
    #             new_state_dict[new_key] = state_dict[key]
    #         else:
    #             new_state_dict[key] = state_dict[key]
    #     return new_state_dict
    
    # # model = VPNGeneratorImageDNN(1024,3,336,14)
    # model = NG_VFeat_DNN(1024)



    # # 加载已保存的权重
    # # checkpoint_path = '~/wzr_workspace/LLaVA/checkpoints/llava-v1.5-7b-noise-test/noise_generator.bin'
    # checkpoint_path = 'checkpoints/NG/llava-v1.5-7b-pretrain/mm_projector.bin'  # 你的权重文件路径
    # state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 载入权重
    # state_dict = remove_prefix_from_state_dict(state_dict, 'noise_generator')

    # print("model",model)

    # # 将权重加载到模型中
    # model.load_state_dict(state_dict)


    # for n, p in model.named_parameters():
    #     print(f"name {n}")
    #     print(f"param {p}")



