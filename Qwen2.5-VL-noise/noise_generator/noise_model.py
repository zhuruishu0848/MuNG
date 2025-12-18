import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from ng_layer import *

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

class Qwen2_5_VL_Noise_Config(Qwen2_5_VLConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None



_CONFIG_FOR_DOC = "Qwen2_5_VL_Noise_Config"

QWEN2_5_VL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing images.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
"""

class Qwen2_5_VL_Noise(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, n_heads, noise_generator_type, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.hidden_size = config.hidden_size
        self.noise_generator_type = noise_generator_type
        
        if noise_generator_type == "NG_vlt_CA":
            self.noise_generator = Masked_NG_VLToken_CA(
                                    feat_dim=self.hidden_size,
                                    n_heads=n_heads)
        elif noise_generator_type == "NG_vlt_MLP":
            self.noise_generator = Masked_NG_VLToken_MLP(
                                    feat_dim=self.hidden_size)
        elif noise_generator_type == "NG_vlt_CA_wo_noise":
            self.noise_generator = Masked_NG_VLToken_CA_wo_noise(
                                    feat_dim=self.hidden_size,
                                    n_heads=n_heads)
        else:
            raise ValueError(f"Unknown noise generator type: {noise_generator_type}")
    
    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print("inputs_embeds",inputs_embeds is None)
        # print("input_ids",input_ids is None)
        # print("pixel_values",pixel_values is None)
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

                # compute noise
                if self.noise_generator_type == "NG_vlt_CA_wo_noise":
                    image_tokens_list = [(i == self.config.image_token_id).sum().item() for i in input_ids]
                    
                    mask = input_ids != self.config.image_token_id
                    mask = mask.to(inputs_embeds.device)  # [b, t]
                    tokens_list = mask.sum(dim=1).tolist()  # List of token counts per batch
                    filtered_inputs_embeds = inputs_embeds[mask] # [sum(tokens_list), d]

                    output_ca = self.noise_generator(V_token=image_embeds, L_token=filtered_inputs_embeds, image_split_list=image_tokens_list, text_split_list=tokens_list)
                    image_embeds = image_embeds + output_ca
                else:
                    image_tokens_list = [(i == self.config.image_token_id).sum().item() for i in input_ids]
                    
                    mask = input_ids != self.config.image_token_id
                    mask = mask.to(inputs_embeds.device)  # [b, t]
                    tokens_list = mask.sum(dim=1).tolist()  # List of token counts per batch
                    filtered_inputs_embeds = inputs_embeds[mask] # [sum(tokens_list), d]

                    # print("input_ids",input_ids.shape)
                    # print("image_embeds",image_embeds.shape)
                    # print("inputs_embeds",inputs_embeds.shape)
                    # print("filtered_inputs_embeds",filtered_inputs_embeds.shape)
                    # exit()

                    mu, log_var, att_weights = self.noise_generator(V_token=image_embeds, L_token=filtered_inputs_embeds, image_split_list=image_tokens_list, text_split_list=tokens_list)
                    noises = self.noise_generator.sample(mu, torch.exp(log_var), 1).squeeze().to(image_embeds.device)
                    image_embeds = image_embeds + noises
                    # image_embeds = image_embeds * noises


                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image
class Qwen2_5_VL_Noise_vis(Qwen2_5_VL_Noise):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

    def vis_am(self, save_path, original_image, attention_map):

        os.makedirs(save_path, exist_ok=True)

        # original_image: [C, H, W]
        # attention_map: [1, 4, P, T]

        # attention_map_head = attention_map[0, 5, :, :]  #[P, T]

        # image_embeds torch.Size([216, 2048])
        # inputs_embeds torch.Size([1, 307, 2048])
        # attention_map torch.Size([1, 4, 216, 307])
        # attention_map_head torch.Size([216, 307])
        # original_image torch.Size([3, 342, 512])


        # print("\n attention_map",attention_map.shape) #([1, 4, 216, 307])
        attention_map_head = torch.mean(attention_map, dim=1).squeeze() # [P, T]
        # attention_map_head = attention_map[0, 6, :, :]
        P = attention_map_head.shape[0] 

        # print("\n attention_map_head",attention_map_head.shape) # ([P, T])
        # print("\n original_image",original_image.shape) # ([3, 336, 504])

        original_image_np  = original_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        # print("\n original_image_np",original_image_np.shape) 

        patch_size =28
        grid_size_h = original_image.shape[1] // patch_size
        grid_size_w = original_image.shape[2] // patch_size

        assert grid_size_h * grid_size_w == P, "注意：Patch 数量与图像尺寸不匹配"

        # 构建一个空白图像，准备叠加 attention map
        attn_overlay = np.zeros_like(original_image_np)

        for token_idx in range(attention_map_head.shape[1]):  # 遍历每个文本 token
            token_attention_map = attention_map_head[:, token_idx].to(torch.float32).view(grid_size_h, grid_size_w).cpu().numpy()  # [grid_size_h, grid_size_w]

            # 归一化 attention map
            token_attention_map = (token_attention_map - token_attention_map.min()) / (token_attention_map.max() - token_attention_map.min() + 1e-6)

            # 使用插值将 attention map 放大到原始图像大小
            token_attention_map_resized = torch.nn.functional.interpolate(
                torch.tensor(token_attention_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0),  # [1, 1, grid_size_h, grid_size_w]
                size=(original_image.shape[1], original_image.shape[2]),  # (H, W)
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()  # [H, W]

            # 让 `original_image` 变为 float 以正确显示
            original_image_float = original_image_np.astype(np.float32)

            # 可视化
            plt.figure(figsize=(8, 8))
            plt.imshow(original_image_float) 
            plt.imshow(token_attention_map_resized, alpha=0.6, cmap='rainbow')  # 叠加 attention map
            plt.colorbar(label='Attention Weight')
            plt.axis('off')

            output_file = os.path.join(save_path, f"attention_token_{token_idx+1}.png")
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"Token {token_idx+1} attention map saved to {output_file}")

    def vis_noise(self, save_path, original_image, noises):

        os.makedirs(save_path, exist_ok=True)

        mean_values = noises.mean(dim=1)
        P = mean_values.shape[0]
        original_image_np = original_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

        patch_size = 28
        grid_size_h = original_image.shape[1] // patch_size
        grid_size_w = original_image.shape[2] // patch_size

        assert grid_size_h * grid_size_w == P, "注意：Patch 数量与图像尺寸不匹配"

        token_attention_map = mean_values.to(torch.float32).view(grid_size_h, grid_size_w).cpu().numpy()  # [grid_size_h, grid_size_w]

        # 归一化 attention map
        token_attention_map = (token_attention_map - token_attention_map.min()) / (token_attention_map.max() - token_attention_map.min() + 1e-6)

        # 使用插值将 attention map 放大到原始图像大小
        token_attention_map_resized = torch.nn.functional.interpolate(
            torch.tensor(token_attention_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=(original_image.shape[1], original_image.shape[2]),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()  # [H, W]

        # 保存单独的 attention map（不叠加原图）
        plt.figure(figsize=(8, 8))
        plt.imshow(token_attention_map_resized, cmap='rainbow')
        plt.colorbar(label='Attention Weight')
        plt.axis('off')
        separate_output_file = os.path.join(save_path, "attention_map_only.png")
        plt.savefig(separate_output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Separate attention map saved to {separate_output_file}")

        # 保存叠加图像
        original_image_float = original_image_np.astype(np.float32)
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image_float)
        plt.imshow(token_attention_map_resized, alpha=0.6, cmap='rainbow')  # 叠加 attention map
        plt.colorbar(label='Attention Weight')
        plt.axis('off')
        output_file = os.path.join(save_path, "noise.png")
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Noise saved to {output_file}")


    # def vis_noise_real(self, save_path, original_image, noises):
    #     import os
    #     from sklearn.decomposition import PCA
    #     import torch
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     os.makedirs(save_path, exist_ok=True)

    #     if noises.is_cuda:
    #         noises = noises.cpu()
    #     noises = noises.to(torch.float32)  # 🔥 转成 float32

    #     # PCA降到3维
    #     pca = PCA(n_components=3)
    #     noise_pca = pca.fit_transform(noises.numpy())  # (234, 3)
    #     noise_pca = torch.tensor(noise_pca)  # (234, 3)

    #     P = noise_pca.shape[0]
    #     original_image_np = original_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

    #     patch_size = 28
    #     grid_size_h = original_image.shape[1] // patch_size
    #     grid_size_w = original_image.shape[2] // patch_size

    #     assert grid_size_h * grid_size_w == P, f"Patch数不匹配: {grid_size_h}x{grid_size_w} != {P}"

    #     # reshape成 (grid_h, grid_w, 3)
    #     token_noise_map = noise_pca.view(grid_size_h, grid_size_w, 3).cpu().numpy()

    #     # 归一化到0-1
    #     token_noise_map = (token_noise_map - token_noise_map.min()) / (token_noise_map.max() - token_noise_map.min() + 1e-6)

    #     # resize到原图大小
    #     token_noise_map_resized = torch.nn.functional.interpolate(
    #         torch.tensor(token_noise_map, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0),  # (1, 3, h, w)
    #         size=(original_image.shape[1], original_image.shape[2]),  # (H, W)
    #         mode='bilinear',
    #         align_corners=False
    #     ).squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]

    #     # 保存1：叠加到原图上
    #     original_image_float = original_image_np.astype(np.float32)

    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(original_image_float)
    #     plt.imshow(token_noise_map_resized, alpha=0.5)  # 🔥 叠加噪声
    #     plt.axis('off')

    #     output_file = os.path.join(save_path, f"noise_real_pca3_overlay.png")
    #     plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     print(f"叠加噪声图已保存到 {output_file}")

    #     # 保存2：单独的噪声图
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(token_noise_map_resized)
    #     plt.axis('off')

    #     output_noise_only = os.path.join(save_path, f"noise_real_pca3_only.png")
    #     plt.savefig(output_noise_only, bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     print(f"单独噪声图已保存到 {output_noise_only}")

    def vis_noise_real_pca(self, save_path, original_image, noises):
        import os
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        os.makedirs(save_path, exist_ok=True)

        if noises.is_cuda:
            noises = noises.cpu()
        noises = noises.to(torch.float32)

        noises_np = noises.numpy()  # [num_patches, feature_dim]
        pca = PCA(n_components=3)
        noise_pca = pca.fit_transform(noises_np)  # [num_patches, 3]

        # Normalize each channel to 0-1 range
        noise_pca -= noise_pca.min(axis=0, keepdims=True)
        noise_pca /= noise_pca.max(axis=0, keepdims=True) + 1e-6  # avoid division by zero

        P = noise_pca.shape[0]
        original_image_np = original_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

        patch_size = 28
        grid_size_h = original_image.shape[1] // patch_size
        grid_size_w = original_image.shape[2] // patch_size

        assert grid_size_h * grid_size_w == P, f"Patch数不匹配: {grid_size_h}x{grid_size_w} != {P}"

        # Reshape to grid
        token_noise_map_rgb = noise_pca.reshape(grid_size_h, grid_size_w, 3)  # [h, w, 3]

        # Resize to original image size
        token_noise_map_rgb_resized = torch.nn.functional.interpolate(
            torch.tensor(token_noise_map_rgb).permute(2, 0, 1).unsqueeze(0),  # [1, 3, h, w]
            size=(original_image.shape[1], original_image.shape[2]),  # (H, W)
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]

        # Overlay on original image
        original_image_float = original_image_np.astype(np.float32)
        overlay_image = 0.6 * original_image_float + 0.4 * token_noise_map_rgb_resized
        overlay_image = np.clip(overlay_image, 0, 1)

        # Save overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay_image)
        plt.axis('off')
        output_file = os.path.join(save_path, f"noise_rgb_overlay.png")
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"叠加RGB彩色噪声图已保存到 {output_file}")

        # Save noise-only RGB map
        plt.figure(figsize=(8, 8))
        plt.imshow(token_noise_map_rgb_resized)
        plt.axis('off')
        output_noise_only = os.path.join(save_path, f"noise_rgb_only.png")
        plt.savefig(output_noise_only, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"RGB彩色噪声图已保存到 {output_noise_only}")

    def vis_noise_real_tsne(self, save_path, original_image, noises):
        import os
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        os.makedirs(save_path, exist_ok=True)

        if noises.is_cuda:
            noises = noises.cpu()
        noises = noises.to(torch.float32)

        noises_np = noises.numpy()  # [num_patches, feature_dim]
        tsne = TSNE(n_components=3, perplexity=30, random_state=42, n_iter=500)
        noise_tsne = tsne.fit_transform(noises_np)  # [num_patches, 3]

        # Normalize each channel to 0-1 range
        noise_tsne -= noise_tsne.min(axis=0, keepdims=True)
        noise_tsne /= noise_tsne.max(axis=0, keepdims=True) + 1e-6  # avoid division by zero

        P = noise_tsne.shape[0]
        original_image_np = original_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

        patch_size = 28
        grid_size_h = original_image.shape[1] // patch_size
        grid_size_w = original_image.shape[2] // patch_size

        assert grid_size_h * grid_size_w == P, f"Patch数不匹配: {grid_size_h}x{grid_size_w} != {P}"

        # Reshape to grid
        token_noise_map_rgb = noise_tsne.reshape(grid_size_h, grid_size_w, 3)  # [h, w, 3]

        # Resize to original image size
        token_noise_map_rgb_resized = torch.nn.functional.interpolate(
            torch.tensor(token_noise_map_rgb).permute(2, 0, 1).unsqueeze(0),  # [1, 3, h, w]
            size=(original_image.shape[1], original_image.shape[2]),  # (H, W)
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]

        # Overlay on original image
        original_image_float = original_image_np.astype(np.float32)
        overlay_image = 0.6 * original_image_float + 0.4 * token_noise_map_rgb_resized
        overlay_image = np.clip(overlay_image, 0, 1)

        # Save overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay_image)
        plt.axis('off')
        output_file = os.path.join(save_path, f"noise_rgb_overlay_TSNE.png")
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"叠加RGB彩色噪声图已保存到 {output_file}")

        # Save noise-only RGB map
        plt.figure(figsize=(8, 8))
        plt.imshow(token_noise_map_rgb_resized)
        plt.axis('off')
        output_noise_only = os.path.join(save_path, f"noise_rgb_only_TSNE.png")
        plt.savefig(output_noise_only, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"RGB彩色噪声图已保存到 {output_noise_only}")

    def visualize_attention_map(self, attention_map, save_dir, grid=False):
        """
        可视化注意力图，并保存到指定路径。

        参数：
        - attention_map: numpy 数组，形状为 (1, num_heads, 576, 43)
        - save_dir: 保存图片的路径
        - grid: 是否显示网格线，默认为 False
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import torch

        os.makedirs(save_dir, exist_ok=True)
        if attention_map.is_cuda:
            attention_map = attention_map.cpu()
        attention_map = attention_map.to(torch.float32).numpy()

        num_heads = attention_map.shape[1]
        x = attention_map.shape[-1]

        # 先设定好横轴tick
        xticks = np.arange(0, x, 1)  # index 0,1,...,x-1
        xtick_labels = np.arange(1, x + 1)  # label 1,2,...,x

        # 保存所有head的attention
        all_heads_attention = []

        for head_index in range(num_heads):
            attention_head = attention_map[0, head_index, :, :]  # (576, 43)
            all_heads_attention.append(attention_head)

            plt.figure(figsize=(10, 6))
            plt.imshow(attention_head, cmap='jet', aspect='auto')
            plt.colorbar(label='Attention Weight')
            plt.title(f"Attention Map for Head {head_index}")

            plt.xticks(xticks, xtick_labels, rotation=45)
            plt.xlim(-0.5, x - 0.5)

            if grid:
                plt.grid(axis="x")

            plt.xlabel("Language token")
            plt.ylabel("Vision token")

            save_path = os.path.join(save_dir, f"attention_head_{head_index}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

        # === 额外加：所有head求平均 ===
        all_heads_attention = np.stack(all_heads_attention, axis=0)  # (num_heads, 576, 43)
        avg_attention = np.mean(all_heads_attention, axis=0)         # (576, 43)

        plt.figure(figsize=(10, 6))
        plt.imshow(avg_attention, cmap='jet', aspect='auto')
        plt.colorbar(label='Average Attention Weight')
        plt.title(f"Average Attention Map Across All Heads")

        plt.xticks(xticks, xtick_labels, rotation=45)
        plt.xlim(-0.5, x - 0.5)

        if grid:
            plt.grid(axis="x")

        plt.xlabel("Language token")
        plt.ylabel("Vision token")

        save_path = os.path.join(save_dir, "attention_average_heads.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


    def visualize_and_save_tensor(self, tensor, save_dir):
        """
        将 CUDA Tensor 移动到 CPU，
        并保存形状为 B*C*H*W 的图片到指定目录。

        参数：
            tensor (torch.Tensor): 输入的 Tensor，形状为 B*C*H*W。
            save_dir (str): 保存图片的目录路径。
        """
        # 确保 tensor 在 CPU 上
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # 检查 tensor 形状
        if tensor.size(1)==3:
            # 遍历 batch
            for i in range(tensor.size(0)):
                # 提取单张图片 (C, H, W)
                image = tensor[i]

                # 若为单通道 (灰度图)，扩展为三通道以便显示
                if image.size(0) == 1:
                    image = image.repeat(3, 1, 1)

                # # 对图像数据进行逐通道检查和映射
                # for c in range(image.size(0)):
                #     channel_min = image[c].min()
                #     channel_max = image[c].max()
                #     # 仅当数据范围超出 [0, 255] 时进行映射
                #     if channel_min < 0 or channel_max > 1:
                #         image[c] = (image[c] - channel_min) / (channel_max - channel_min + 1e-5) * 1

                # 调整通道顺序为 (H, W, C)
                image_np = image.permute(1, 2, 0).contiguous().to(torch.float32).numpy()

                # 绘制并保存图片
                plt.figure(figsize=(4, 4))
                plt.imshow(image_np)
                plt.axis('off')

                # 保存文件
                # save_path = os.path.join(save_dir, f"image_{i}.png")
                plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"所有图片已保存到文件夹: {save_dir}")
        else:
            tensor = tensor.squeeze()
            tensor = tensor.to(torch.float32).numpy()

            plt.figure(figsize=(10, 6))
            # plt.imshow(attention_head, cmap='viridis', aspect='auto')
            plt.imshow(tensor, cmap='jet', aspect='auto')
            plt.colorbar()
            
            plt.savefig(save_dir)
            plt.close()
            print(f"图片已保存到文件夹: {save_dir}")


    def visualize_heatmap_save(self, tensor, save_dir):
        """
        将 CUDA Tensor 移动到 CPU，
        并保存形状为 B*C*H*W 的图片到指定目录。

        参数：
            tensor (torch.Tensor): 输入的 Tensor，形状为 B*C*H*W。
            save_dir (str): 保存图片的目录路径。
        """
        # 确保 tensor 在 CPU 上
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # 检查 tensor 形状
        if tensor.size(1)==3:
            
            # 遍历 batch
            for i in range(tensor.size(0)):
                # 提取单张图片 (C, H, W)
                image = tensor[i]

                # 若为单通道 (灰度图)，扩展为三通道以便显示
                if image.size(0) == 1:
                    image = image.repeat(3, 1, 1)

                image = torch.sum(image,dim=0)

                channel_min = image.min()
                channel_max = image.max()
                # 仅当数据超出范围时进行映射
                if channel_min < 0 or channel_max > 1:
                    image = (image - channel_min) / (channel_max - channel_min + 1e-5) * 1
                    

                # 调整通道顺序为 (H, W, C)
                # image_np = image.permute(1, 2, 0).contiguous().to(torch.float32).numpy()
                image_np = image.to(torch.float32).numpy()

                ax = sns.heatmap(image_np, vmin=0,vmax=1,cmap="brg", square=True)

                figure = ax.get_figure()
                save_dir = save_dir+"_hm"
                figure.savefig(save_dir, bbox_inches='tight', pad_inches=0)  # 保存图片

            print(f"所有图片已保存到文件夹: {save_dir}")
        else:
            tensor = tensor.squeeze()
            tensor = tensor.to(torch.float32).numpy()

            # ax = sns.heatmap(tensor, cmap="brg", square=True)
            ax = sns.heatmap(tensor, cmap="jet", square=False)

            figure = ax.get_figure()
            figure.savefig(save_dir, bbox_inches='tight', pad_inches=0)  # 保存图片
                
    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

               # compute noise
                image_tokens_list = [(i == self.config.image_token_id).sum().item() for i in input_ids]
                
                mask = input_ids != self.config.image_token_id  # [b, t]
                inputs_embeds = inputs_embeds.cuda()

                # tokens_list = (mask.sum(dim=1)-21).tolist()  # List of token counts per batch
                # filtered_inputs_embeds = inputs_embeds[mask]  # [sum(tokens_list), d]

                # filtered_inputs_embeds = filtered_inputs_embeds[17:-4].cuda()

                tokens_list = mask.sum(dim=1).tolist()  # List of token counts per batch
                filtered_inputs_embeds = inputs_embeds[mask]  # [sum(tokens_list), d]

                filtered_inputs_embeds = filtered_inputs_embeds.cuda()
                
                mu, log_var, att_weights = self.noise_generator(V_token=image_embeds, L_token=filtered_inputs_embeds, image_split_list=image_tokens_list, text_split_list=tokens_list)

                noises = self.noise_generator.sample(mu, torch.exp(log_var), 1)
                noises = noises.squeeze().to(image_embeds.device)
                # epsilon = epsilon.squeeze().to(image_embeds.device)
                image_embeds = image_embeds + noises
                # exit()

                # #  ----------------------image vis----------------------
                import seaborn as sns
                import matplotlib.pyplot as plt
                import numpy as np
                import os
                from PIL import Image
                model_path = "MMPR-shuff-3B/noise-5e4-n8-b32/checkpoint-400/img_diagram"
                save_dir = "./noise_generator/visualize/" + model_path
                img_path = "./noise_generator/visualize/images/img_diagram.jpg"
                image = Image.open(img_path).convert("RGB")  
                width, height = image.size
                from qwen_vl_utils import smart_resize
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=28,
                    min_pixels=4 * 28 * 28,
                    max_pixels=16384 * 28 * 28,
                )
                image = image.resize((resized_width, resized_height))
                import torchvision.transforms as transforms
                transform = transforms.ToTensor()  # (C, H, W) 格式，值范围 [0, 1]
                images = transform(image)

                self.vis_noise(save_dir+"/noise", images, noises.detach())
                self.vis_noise(save_dir+"/mu", images, mu.detach())
                # self.vis_noise(save_dir+"/epsilon", images, epsilon.detach())
                self.vis_noise(save_dir+"/var", images, torch.exp(log_var).detach())

                self.vis_noise_real_pca(save_dir+"/noise_real", images, noises.detach())
                self.vis_noise_real_pca(save_dir+"/mu", images, mu.detach())
                # self.vis_noise_real_pca(save_dir+"/epsilon", images, epsilon.detach())
                self.vis_noise_real_pca(save_dir+"/var", images, torch.exp(log_var).detach())

                self.vis_noise_real_tsne(save_dir+"/noise_real", images, noises.detach())
                self.vis_noise_real_tsne(save_dir+"/mu", images, mu.detach())
                # self.vis_noise_real_tsne(save_dir+"/epsilon", images, epsilon.detach())
                self.vis_noise_real_tsne(save_dir+"/var", images, torch.exp(log_var).detach())

                self.vis_am(save_dir+"/tokens", images, att_weights.detach().unsqueeze(0))

                sum_tensor = att_weights.detach().sum(dim=-1)  # shape: [1, b, n]
                sum_tensor = sum_tensor.sum(dim=1)  # shape: [1, b, n]
                softmax_tensor = F.softmax(sum_tensor, dim=-1)  # shape: [1, n]
                softmax_np = softmax_tensor.cpu().to(torch.float32).numpy()  # [b, n]

                self.visualize_attention_map(att_weights.detach().unsqueeze(0), save_dir+"/att_weights/",grid=True)




                # self.visualize_and_save_tensor(images, save_dir+"/images")
                # visualize_heatmap_save(image_clipfeatures.detach(), save_dir+"image_clipfeatures")

                # self.visualize_heatmap_save(cur_image_features.detach(), save_dir+"image_token")
                # self.visualize_heatmap_save(qa_inputs_embeds.detach(), save_dir+"text_features")
                # visualize_heatmap_save(visual_feat_new.detach(), save_dir+"visual_feat_new")
                # self.visualize_heatmap_save(mu.detach(), save_dir+"mu")
                # self.visualize_heatmap_save(log_var.detach(), save_dir+"log_var")
                # self.visualize_heatmap_save(noises.detach(), save_dir+"noises")
                # self.visualize_heatmap_save(( image_features[cur_image_idx] + noises).detach(), save_dir+"noise_image_features")
                # visualize_heatmap_save(image_features_add.detach(), save_dir+"noise_image_features_add")
                # visualize_heatmap_save(image_features_mul.detach(), save_dir+"noise_image_features_mul")
                


                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )