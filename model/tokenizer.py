##config: use_vision_zoe(bool)

from transformers import ZoeDepthForDepthEstimation
import torch
from torch import nn
from PIL import Image
from transformers.processing_utils import Unpack, _validate_images_text_input_order, ProcessorMixin

import os
import torch
import torch.utils.checkpoint
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    ZoeDepthConfig,
    ZoeDepthForDepthEstimation,
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainingArguments,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from typing import List, Optional, Union, Dict
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import Unpack, _validate_images_text_input_order, ProcessorMixin
from transformers.tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from transformers.models.paligemma.processing_paligemma import (
    make_batched_images, 
    build_string_from_input, 
    _is_str_or_image, 
    PaliGemmaProcessorKwargs,
    IMAGE_TOKEN,
    EXTRA_TOKENS
)

SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
ZOE_MEAN, ZOE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


class Ego3DPositionEmbeddingMLP(nn.Module):
    """Absolute pos embedding, learned.
    https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/models/nerf.py#L4
    """

    def __init__(self, in_channels=3, num_pos_feats=768, n_freqs=8, logscale=True):
        super(Ego3DPositionEmbeddingMLP, self).__init__()
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)
        
        center = torch.tensor([0., 0., 2.]).repeat(in_channels // 3)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """init with small weights to maintain stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def frequency_encoding(self, xyz):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        x \in [-2, 2]
        y \in [-2, 2]
        z \in [0., 4]
        Inputs:
            x: (b n m)
        Outputs:
            out: (b n o)
        """
        xyz_n = ((xyz - self.center) / 2.0).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)  # (b n m nf)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        freq_encoding = self.frequency_encoding(xyz)
        position_embedding = self.position_embedding_head(freq_encoding)
        return position_embedding

def process_zoe(pixel_values, pad_mode="reflect", output_size=(384, 512)):
    """https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/zoedepth/image_processing_zoedepth.py"""
    # h, w = images.shape[-2:]
    # pad
    ph, pw = 31, 31  # int((h / 2)**0.5 * 3), int((w / 2)**0.5 * 3) # 32, 31
    images = F.pad(pixel_values, (pw, pw, ph, ph), mode=pad_mode)
    # resize
    size = (384, 384)  # get_resize_output_image_size
    images = F.interpolate(images, size=size, mode="bicubic", align_corners=True)
    # zoe: padding -> resize -> nomalize. we follow `nomalize -> padding -> resize` from siglip
    images = TF.normalize(images, mean=ZOE_MEAN, std=ZOE_STD)
    return images, ph, pw


class Imagetokenizer():
    def __init__(self, config, vision_zoe_model=None):
        super().__init__(config)
        
        self.config=config
        
        if config.use_vision_zoe:
            self.vision_zoe_model = vision_zoe_model or ZoeDepthForDepthEstimation(config.vision_zoe_config)
            self.position_embedding_3d = Ego3DPositionEmbeddingMLP(
                config.ego3d_patch_reso**2 * 3, num_pos_feats=config.vision_config.hidden_size, n_freqs=config.n_freqs
            )
            # register buffer
            patch_size, reso, image_size = config.vision_config.patch_size, config.ego3d_patch_reso, config.vision_config.image_size
            y, x = torch.meshgrid(torch.arange(0, image_size, patch_size // reso), torch.arange(0, image_size, patch_size // reso), indexing="ij")  # (h//sp w//sp)
            y, x = y + patch_size / reso / 2, x + patch_size / reso / 2
            uv_h = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)  # (3 hw)
            self.register_buffer("uv_h", uv_h, persistent=False)
    
    
    def get_image_features(self, pixel_values: torch.FloatTensor, intrinsic: torch.FloatTensor):
        siglip_pixel_values = TF.normalize(pixel_values, mean=SIGLIP_MEAN, std=SIGLIP_STD)
        image_outputs = self.vision_tower(siglip_pixel_values)

        # ego3d position encoding
        if self.config.use_vision_zoe:
            zoe_pixel_values, ph, pw = process_zoe(pixel_values, pad_mode="reflect")
            with torch.no_grad():
                pvh, pvw = pixel_values.shape[-2:]
                depth = self.vision_zoe_model(pixel_values=zoe_pixel_values).predicted_depth
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(pvh+2*ph, pvw+2*pw),
                    mode="bicubic",
                    align_corners=True,
                )[..., ph:-ph, pw:-pw]
                xyz = self.backproject_patch(
                    intrinsic, depth, patch_size=self.config.vision_config.patch_size, reso=self.config.ego3d_patch_reso
                )  # (b, n, 3*4)
            pos_embed_3d = self.position_embedding_3d(xyz)
            selected_image_feature = image_outputs.last_hidden_state + pos_embed_3d
        else:
            selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.text_config.hidden_size**0.5)
        return image_features
    
       
        # vision_zoe_config = ZoeDepthConfig.from_pretrained(
        #     model_args.vision_zoe_path, torch_dtype=torch_dtype, local_files_only=True
        # )
        # vision_zoe_model = ZoeDepthForDepthEstimation.from_pretrained(  # zoe does not support Flash Attention 2.0 yet.
        #     model_args.vision_zoe_path,
        #     config=vision_zoe_config,
        #     torch_dtype=torch_dtype,
        #     local_files_only=True,
        # )
        
class ForceProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        statistics: Optional[dict] = None,
        bin_policy=None,
        intrinsic_config=None,
        action_config=None,
        num_obs_steps=1,
        obs_delta=1,
        action_chunk_size=1,
        min_sigma=0.0,
        **kwargs,
    ):    
        self.image_processor=image_processor
        
        for k, v in intrinsic_config.items():
            K = torch.tensor(v["intrinsic"]).float()
            K[:2] *= torch.tensor([width / v["width"], height / v["height"]])[:, None]
            self.dataset_intrinsics[k] = K
        
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        unnorm_key: Optional[str] = None,
        suffix_actions: Optional[np.array] = None, # (t e)
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature: 
        images, text = _validate_images_text_input_order(images, text)
        if text is not None and images is not None:
            if not any(IMAGE_TOKEN in sample for sample in text):
                if isinstance(text, List) and isinstance(images, List):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image or list of images."
                        )
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, list) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                    raise ValueError("images must be an image, list of images or list of list of images")
                if suffix is not None and _is_str_or_image(suffix): suffix = [suffix]
                if suffix is not None: suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]
                input_strings = [
                    build_string_from_input(
                        prompt=prompt,
                        bos_token=self.tokenizer.bos_token,
                        image_seq_len=self.image_seq_length,
                        image_token=IMAGE_TOKEN,
                        num_images=len(image_list) if isinstance(image_list, list) else 1,
                    )
                    for prompt, image_list in zip(text, images)
                ]
                images = make_batched_images(images)
            else:
                expanded_samples = []
                for sample in text:
                    expanded_sample = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                    bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                    bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
                    expanded_sample = (
                        expanded_sample[:bos_index] + self.tokenizer.bos_token + expanded_sample[bos_index:]
                    )
                    expanded_samples.append(expanded_sample)
                input_strings = [f"{sample}\n" for sample in expanded_samples]
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        
        
if __name__ == '__main__':
    image = Image.open("right3.png").convert("RGB")
    prompt = "What action should the robot take to pick the bottle?"
    paligemma_processor = PaliGemmaProcessor.from_pretrained(model_args.vlm_path, local_files_only=True)
    paligemma_processor.image_processor.do_normalize = False 
    processor=
    
    inputs = processor(images=[image], text=prompt, return_tensors="pt")
    vision_zoe_path = "../pretrained/zoedepth-nyu-kitti"
    torch_dtype = torch.bfloat16
    vision_zoe_config = ZoeDepthConfig.from_pretrained(
        vision_zoe_path, torch_dtype=torch_dtype, local_files_only=True
    )
    I=Imagetokenizer()