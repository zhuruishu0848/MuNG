# # # # # import torch

# # # # # cur_input_embeds = torch.randn(5,7)
# # # # # print(cur_input_embeds.shape)
# # # # # print(cur_input_embeds)
# # # # # target_length = 5

# # # # # padding_size = target_length - cur_input_embeds.size(0)
# # # # # assert padding_size >= 0 ,"over target_length" 
# # # # # cur_input_embeds = torch.nn.functional.pad(cur_input_embeds, (0,0,0,padding_size), mode='constant', value=0)
# # # # # print(cur_input_embeds.shape)
# # # # # print(cur_input_embeds)


# # # # # from torch.nn.utils.rnn import pad_sequence
# # # # # import torch
# # # # # a=torch.randn(3)
# # # # # b=torch.randn(5)
# # # # # c=torch.randn(7)
# # # # # print(a)
# # # # # # tensor([ 0.7160,  1.2006, -1.8447])
# # # # # print(b)
# # # # # # tensor([ 0.3941,  0.3839,  0.1166, -0.7221,  1.8661])
# # # # # print(c)
# # # # # # tensor([-0.6521,  0.0681,  0.6626, -0.3679, -0.6042,  1.6951,  0.4937])
# # # # # print(pad_sequence([a,b,c]).shape)
# # # # # # tensor([[ 0.7160,  0.3941, -0.6521],
# # # # # #         [ 1.2006,  0.3839,  0.0681],
# # # # # #         [-1.8447,  0.1166,  0.6626],
# # # # # #         [ 0.0000, -0.7221, -0.3679],
# # # # # #         [ 0.0000,  1.8661, -0.6042],
# # # # # #         [ 0.0000,  0.0000,  1.6951],
# # # # # #         [ 0.0000,  0.0000,  0.4937]])
# # # # # print(pad_sequence([a,b,c],batch_first=True).shape)
# # # # # # tensor([[ 0.7160,  1.2006, -1.8447,  0.0000,  0.0000,  0.0000,  0.0000],
# # # # # #         [ 0.3941,  0.3839,  0.1166, -0.7221,  1.8661,  0.0000,  0.0000],
# # # # # #         [-0.6521,  0.0681,  0.6626, -0.3679, -0.6042,  1.6951,  0.4937]])

# # # # # print(pad_sequence([a,b,c],batch_first=True,padding_value=1).shape)
# # # # # # tensor([[ 0.7160,  1.2006, -1.8447,  1.0000,  1.0000,  1.0000,  1.0000],
# # # # # #         [ 0.3941,  0.3839,  0.1166, -0.7221,  1.8661,  1.0000,  1.0000],
# # # # # #         [-0.6521,  0.0681,  0.6626, -0.3679, -0.6042,  1.6951,  0.4937]])

# # # # import torch
# # # # import torch.nn as nn

# # # # embedding = nn.Embedding(32000, 4096,padding_idx=0) # an Embedding module containing 10 tensors of size 3
# # # # _input_ids = torch.randint(-50, 100, (16,25), dtype=torch.int64)
# # # # _input_ids[0][10:] = 0
# # # # print(_input_ids[0])
# # # # print("max",_input_ids.max())
# # # # negative_rows = (_input_ids < 0).any(dim=1)
# # # # for i, row in enumerate(_input_ids[negative_rows]):
# # # #     print(f"Row {i}: {row}")
# # # # print(_input_ids.dtype)
# # # # e = embedding(_input_ids)
# # # # # print(e)
# # # # print(e.shape)

# # # import torch
# # # from diffusers import DiffusionPipeline

# # # pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True)
# # # pipe = pipe.to("cuda")

# # # prompt = "a photo of an astronaut riding a horse on mars"
# # # image = pipe(prompt).images[0]

# # # print("image",image)

# # # import torch
# # # import matplotlib.pyplot as plt
# # # import os

# # # def visualize_and_save_tensor(tensor, save_dir):
# # #     """
# # #     将 CUDA Tensor 移动到 CPU，
# # #     并保存形状为 B*C*H*W 的图片到指定目录。

# # #     参数：
# # #         tensor (torch.Tensor): 输入的 Tensor，形状为 B*C*H*W。
# # #         save_dir (str): 保存图片的目录路径。
# # #     """
# # #     # 确保 tensor 在 CPU 上
# # #     if tensor.is_cuda:
# # #         tensor = tensor.cpu()

# # #     # 检查 tensor 形状
# # #     if tensor.dim() != 4 or tensor.size(1) not in [1, 3]:
# # #         raise ValueError("输入 Tensor 的形状必须为 B*C*H*W，且 C 必须为 1 或 3。")

# # #     # 创建保存图片的文件夹
# # #     os.makedirs(save_dir, exist_ok=True)

# # #     # 遍历 batch
# # #     for i in range(tensor.size(0)):
# # #         # 提取单张图片 (C, H, W)
# # #         image = tensor[i]

# # #         # 若为单通道 (灰度图)，扩展为三通道以便显示
# # #         if image.size(0) == 1:
# # #             image = image.repeat(3, 1, 1)

# # #         # 调整通道顺序为 (H, W, C)
# # #         image_np = image.permute(1, 2, 0).numpy()

# # #         # 绘制并保存图片
# # #         plt.figure(figsize=(4, 4))
# # #         plt.imshow(image_np)
# # #         plt.axis('off')

# # #         # 保存文件
# # #         save_path = os.path.join(save_dir, f"image_{i}.png")
# # #         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
# # #         plt.close()

# # #     print(f"所有图片已保存到文件夹: {save_dir}")

# # # # 示例用法
# # # if __name__ == "__main__":
# # #     # 创建一个示例 Tensor，形状为 B*C*H*W
# # #     example_tensor = torch.rand(4, 3, 128, 128).cuda()  # 在 CUDA 上生成数据
# # #     save_dir = "visualize/"

# # #     # 调用函数，可视化并保存图片
# # #     visualize_and_save_tensor(example_tensor, save_dir+"visualized_images")


# # # from sentence_transformers import SentenceTransformer, util
# # # from PIL import Image

# # # #Load CLIP model
# # # model = SentenceTransformer('clip-ViT-L-14')
# # # print("model",model)
# # # print("model",model.weight.dtype)


# # import torch

# # def kl_divergence_loss(mu_q, var_q, mu_p, var_p, reduction='mean'):
# #     """
# #     计算 q ~ N(mu_q, var_q) 与 p ~ N(mu_p, var_p) 之间的 KL 散度。
    
# #     参数：
# #         mu_q: Tensor，形状为 [b, t, e]，q 分布的均值
# #         var_q: Tensor，形状为 [b, t, e]，q 分布的方差
# #         mu_p: Tensor，形状为 [b, t, e]，p 分布的均值
# #         var_p: Tensor，形状为 [b, t, e]，p 分布的方差
# #         reduction: str，'mean' 表示取均值，'sum' 表示求和，或者 'none' 返回各元素的 KL 值
        
# #     返回：
# #         KL 散度损失，根据 reduction 参数返回标量或与输入同形状的 Tensor。
# #     """
# #     # 为防止除零，增加一个很小的数值 epsilon
# #     eps = 1e-8
    
# #     # 计算每个元素的 KL 散度
# #     kl_element = 0.5 * (
# #         torch.log((var_p + eps) / (var_q + eps)) + 
# #         (var_q + (mu_q - mu_p)**2) / (var_p + eps) - 
# #         1
# #     )
    
# #     # 对特征维度求和（假设 e 维度上独立）
# #     kl_sum = kl_element.sum(dim=-1)  # 结果 shape 为 [b, t]
    
# #     # 根据 reduction 参数进行归约
# #     if reduction == 'mean':
# #         return kl_sum.mean()
# #     elif reduction == 'sum':
# #         return kl_sum.sum()
# #     else:
# #         return kl_sum

# # # 示例
# # b, t, e = 4, 10, 32
# # mu_q = torch.randn(b, t, e)
# # var_q = torch.rand(b, t, e) + 1.0  # 保证方差大于0
# # mu_p = torch.randn(b, t, e)
# # var_p = torch.rand(b, t, e) + 1.0

# # kl_loss = kl_divergence_loss(mu_q, var_q, mu_p, var_p, reduction='mean')
# # print("KL Loss:", kl_loss.item())


# import torch

# # 假设 label_inputs 和 qa_inputs_embeds 已经在 GPU 上
# device = torch.device("cuda")
# label_inputs = torch.tensor([
#     -100, -100, -100, -100, -100, -100, -100, -100,  # Q1
#     2023, 2003, 1037, 3437, 102,                     # A1
#     -100, -100, -100, -100, -100, -100, -100, -100,  # Q2
#     2643, 2339, 2196, 1037, 2812, 102               # A2
# ], device=device)

# # 假设 qa_inputs_embeds 是 (t, d) 维度的张量
# t, d = 27, 5  # 以示例为准
# qa_inputs_embeds = torch.rand(t, d, device=device)
# print("qa_inputs_embeds",qa_inputs_embeds)

# # 找到所有-100的索引
# mask = (label_inputs == -100).int()

# # 创建切割点：-100的变化位置
# neg_start_indices = (mask.diff(prepend=torch.tensor([0], device=device)) == 1).nonzero().squeeze()
# pos_start_indices = (mask.diff(append=torch.tensor([0], device=device)) == -1).nonzero().squeeze() + 1

# # 将切割点添加为列表
# sub_tensors = []

# # 根据切割点切分 qa_inputs_embeds
# sub_tensors = [qa_inputs_embeds[neg_start_indices[i].item():neg_start_indices[i + 1].item()]
#                if i + 1 < len(neg_start_indices)
#                else qa_inputs_embeds[neg_start_indices[i].item():] 
#                for i in range(len(neg_start_indices))]

# # sub_tensors 现在是一个列表，包含了按每组分割的子张量
# print(f"Number of sub-tensors: {len(sub_tensors)}")
# for i, sub_tensor in enumerate(sub_tensors):
#     print(f"Sub-tensor {i} shape: {sub_tensor.shape}")


import torch 

# 假设 label_inputs 和 qa_inputs_embeds 已经在 GPU 上
device = torch.device("cuda")
label_inputs = torch.tensor([
    -100, -100, -100, -100, -100, -100, -100, -100,  # Q1
    -100, -100, -100, -100, -100,                     # A1
    -100, -100, -100, -100, -100, -100, -100, -100,  # Q2
    2643, 2339, 2196, 1037, 2812, 102               # A2
], device=device)

# 假设 qa_inputs_embeds 是 (t, d) 维度的张量
t, d = 27, 5  # 以示例为准
qa_inputs_embeds = torch.rand(t, d, device=device)
cur_image_features = torch.rand(576, 4096, device=device)

# 获取 -100 的起始索引，并在最后添加总长度
mask = (label_inputs == -100).int()
neg_start_indices = (mask.diff(prepend=torch.tensor([0], device=device)) == 1).nonzero().view(-1)
neg_start_indices = torch.cat([neg_start_indices, torch.tensor([len(label_inputs)], device=device)])

# 切分 qa_inputs_embeds
qa_inputs_embeds_split = [qa_inputs_embeds[neg_start_indices[i]:neg_start_indices[i + 1]] for i in range(len(neg_start_indices) - 1)]

qa_max_len = max([qa.size(0) for qa in qa_inputs_embeds_split])
batch_qa = len(qa_inputs_embeds_split)

batch_image_features = cur_image_features.unsqueeze(0).expand(batch_qa, -1, -1)
print("batch_qa",batch_qa)
print("batch_image_features",batch_image_features.shape)


padded_qa_list = []
for qa in qa_inputs_embeds_split:
    n = qa.size(0)
    if n < qa_max_len:
        padding = torch.cat([qa, qa.mean(dim=0, keepdim=True).repeat(qa_max_len - n, 1)], dim=0)
    else:
        padding = qa[:qa_max_len]
    assert not torch.any(torch.isnan(padding)), "Found NaN."
    padded_qa_list.append(padding)
batch_qa_inputs_embeds = torch.stack(padded_qa_list)
print("batch_qa_inputs_embeds",batch_qa_inputs_embeds.shape)

# # 输出切分结果
# print(f"Number of sub-tensors: {len(qa_inputs_embeds_split)}")
# for i, sub_tensor in enumerate(qa_inputs_embeds_split):
#     print(f"Sub-tensor {i} shape: {sub_tensor.shape}")
#     print(f"Sub-tensor {i} shape: {sub_tensor}")
