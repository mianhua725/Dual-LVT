from collections.abc import Sequence
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm
import numpy as np

from monai.networks.blocks import (PatchEmbed,
                                   UnetOutBlock,
                                   UnetrBasicBlock,
                                   UnetrUpBlock)
from monai.utils import ensure_tuple_rep, optional_import
from dependency.swin_unetr import (SwinTransformerBlock,
                                          PatchMerging,
                                          get_window_size,
                                          compute_mask)
from adahu import GateClip

rearrange, _ = optional_import("einops", name="rearrange")


def get_num_params(model: torch.nn.Module):
    print(sum([p.numel() for p in model.parameters()]))


class DualAttnUNet(nn.Module):

    def __init__(self,
                 img_in_chans: int,
                 lang_in_chans: int,
                 out_channels: int,
                 seq_length: int, 
                 img_size: Sequence[int] | int,
                 patch_size: Sequence[int] | int,
                 window_size: Sequence[int] | int,
                 use_gateclip: bool = True,
                 depths: Sequence[int] = (1, 1, 1, 1),
                 num_heads: Sequence[int] = (4, 2, 1, 1),
                 feature_size: int = 48,
                 norm_name: tuple | str = "instance",
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 dropout_path_rate: float = 0.0,
                 normalize: bool = True,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3,) -> None:


        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.window_size = ensure_tuple_rep(7, spatial_dims)
        self.use_gateclip = use_gateclip

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")
    
        # self.normalize = normalize

        if use_gateclip:
            self.gate_clip = GateClip(in_channel=img_in_chans)

        self.dualattn_transformer = DualAttnFormer(img_in_chans=img_in_chans,
                                                   lang_in_chans=lang_in_chans,
                                                   dim=feature_size,
                                                   seq_length=seq_length,
                                                   image_size=img_size,
                                                   window_size=self.window_size,
                                                   patch_size=self.patch_size,
                                                   num_heads=num_heads,
                                                   depths=depths)

        self.input_encoder = UnetrBasicBlock(spatial_dims=spatial_dims,
                                             in_channels=img_in_chans,
                                             out_channels=feature_size,
                                             kernel_size=3,
                                             stride=1,
                                             norm_name=norm_name,
                                             res_block=True,)

        self.encoder_list = nn.ModuleList()

        for stage in range(len(depths)):
            channels = feature_size if stage == 0 else feature_size*2**(2**(stage-1))
            encoder = UnetrBasicBlock(spatial_dims=spatial_dims,
                                      in_channels=channels,
                                      out_channels=channels,
                                      kernel_size=3,
                                      stride=1,
                                      norm_name=norm_name,
                                      res_block=True,)
            self.encoder_list.append(encoder)
        
        self.decoder_list = nn.ModuleList()

        for stage in range(len(depths)+1):
            decoder = UnetrUpBlock(spatial_dims=spatial_dims,
                                   in_channels=feature_size*2**(len(depths)-stage),
                                   out_channels=feature_size*2**max(len(depths)-stage-1, 0),
                                   kernel_size=3,
                                   upsample_kernel_size=2,
                                   norm_name=norm_name,
                                   res_block=True)
            self.decoder_list.append(decoder)

        self.out = UnetOutBlock(spatial_dims=spatial_dims,
                                in_channels=feature_size,
                                out_channels=out_channels)

    def forward(self, img_in: torch.FloatTensor, lang_in: torch.FloatTensor):
        b, c, d, h, w = img_in.shape
        if self.use_gateclip:
            img_in = self.gate_clip(img_in)
            img_min = img_in.view(b, -1).min(dim=-1, keepdim=True)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3).detach()
            img_max = img_in.view(b, -1).max(dim=-1, keepdim=True)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3).detach()
            img_in = (img_in - img_min) / (img_max - img_min + 1e-6)

        hidden_states_out = self.dualattn_transformer(img_in, lang_in)
        hidden_states_out[-1], hidden_states_out[-2] = hidden_states_out[-2], hidden_states_out[-1]
        enc = self.input_encoder(img_in)
        enc_list = [enc]
        
        for i, encoder in enumerate(self.encoder_list):
            enc = encoder(hidden_states_out[i])
            enc_list.append(enc)
        dec = self.decoder_list[0](enc_list.pop(), hidden_states_out[-1])
        for i, decoder in enumerate(self.decoder_list[1:]):
            dec = decoder (dec, enc_list.pop())
        logits = self.out (dec)
        return logits


    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5) > 0)
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                             f" must be divisible by {self.patch_size}**5.")


class DualAttnFormer(torch.nn.Module):

    def __init__(self, 
                 img_in_chans: int,
                 lang_in_chans: int,
                 dim: int,
                 seq_length: int,
                 image_size: Sequence[int],
                 patch_size: Sequence[int],
                 window_size: Sequence[int],
                 num_heads: Sequence[int],
                 depths: Sequence[int],
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path_rate: float = 0.0,
                 act_layer: str = "GELU",
                 spatial_dims: int = 3,
                 downsample="merging",
                 norm_layer: type[LayerNorm] = nn.LayerNorm,
                 use_checkpoint: bool = False,) -> None:

        super().__init__()

        assert all([not (img_sz % w_sz) for img_sz, w_sz in zip(image_size, patch_size)]),\
            "Image size should be divisible by patch size."

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.act_layer = act_layer
        self.patch_size = patch_size
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(patch_size=patch_size,
                                      in_chans=img_in_chans,
                                      embed_dim=dim,
                                      norm_layer=norm_layer,
                                      spatial_dims=spatial_dims)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_list = nn.ModuleList()

        for stage in range(4):
            fmap_size=[sz // w_sz // 2**stage for sz, w_sz in zip(image_size, patch_size)]
            blk_list = self._create_block(c_i=int(dim * 2**stage),
                                          c_t=lang_in_chans,
                                          depth=depths[stage],
                                          num_heads=num_heads[stage],
                                          image_size=fmap_size,
                                          seq_length=seq_length,
                                          drop_paths=dpr[sum(depths[:stage]) : sum(depths[: stage + 1])])
            self.layers_list.append(blk_list)

    def forward(self, img_in: torch.Tensor, lang_in: torch.Tensor):
        x = self.patch_embed(img_in)
        x = self.pos_drop(x)
        out = [self._proj_out(x, normalize=True)]
        for i, blk in enumerate(self.layers_list):
            for depth_idx in range(len(blk)-1):
                layer = blk[f"depth_{depth_idx}"]
                shortcut = x
                attn_mask = self._get_attn_mask(x)
                x = rearrange(x, "b c d h w -> b d h w c")
                x = layer[f"swin_transformer_{depth_idx}"](x, attn_mask)
                x = rearrange(x, "b d h w c -> b c d h w")
                x_img = layer[f"image_query_branch_{depth_idx}"](x, lang_in)
                x_lang = layer[f"language_query_branch_{depth_idx}"](x, lang_in)
                x = x + 0.1*x_img + 0.1*x_lang + shortcut

            x = rearrange(x, "b c d h w -> b d h w c")
            x = blk[f"downsample"](x)
            x = rearrange(x, "b d h w c -> b c d h w")
            x = self._proj_out(x, normalize=True)
            out.append(x)
        return out

    def _create_block(self,
                      c_i: int,
                      c_t: int,
                      depth: int,
                      num_heads: int,
                      image_size: Sequence[int],
                      seq_length: int,
                      drop_paths: Sequence[float]) -> nn.Module:

        self.shift_size = tuple(i // 2 for i in self.window_size)
        self.no_shift = tuple(0 for i in self.window_size)

        blk_dict = nn.ModuleDict()
        for i in range(depth):
            swin_transformer = SwinTransformerBlock(dim=c_i,
                                                    num_heads=num_heads,
                                                    window_size=self.window_size,
                                                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=self.qkv_bias,
                                                    drop=0.0,
                                                    attn_drop=0.0,
                                                    drop_path=drop_paths[i],
                                                    norm_layer=nn.LayerNorm,)
            
            image_query_branch = ImageQueryBranch(c_i=c_i,
                                                  c_t=c_t,
                                                  num_heads=num_heads)

            language_query_branch = LanguageQueryBranch(c_i=c_i,
                                                        c_t=c_t,
                                                        image_size=image_size,
                                                        seq_length=seq_length,
                                                        num_heads=num_heads)
            
            layer_dict = torch.nn.ModuleDict({f"swin_transformer_{i}": swin_transformer,
                                              f"image_query_branch_{i}": image_query_branch, 
                                              f"language_query_branch_{i}": language_query_branch})
            blk_dict[f'depth_{i}'] = layer_dict
        
        blk_dict[f"downsample"] = PatchMerging(dim=c_i, norm_layer=nn.LayerNorm, spatial_dims=3)

        return blk_dict

    def _proj_out(self, x: torch.FloatTensor, normalize=False):
        if normalize:
            x_shape = x.shape
            # Force trace() to generate a constant by casting to int
            ch = int(x_shape[1])
            if len(x_shape) == 5:
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def _get_attn_mask(self, x: torch.FloatTensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        return attn_mask


class ImageQueryBranch(torch.nn.Module):

    def __init__(self, c_i, c_t, num_heads=1) -> None:
        super().__init__()
        self.c_i = c_i
        self.c_t = c_t
        self.num_heads = num_heads

        # Projections for visual features as query
        self.f_query = nn.Sequential(nn.Conv3d(c_i, c_i, kernel_size=1),
                                     nn.InstanceNorm3d(c_i))

        # Projection for language feature as key and value
        self.f_key = nn.Conv1d(c_t, c_i, kernel_size=1)
        self.f_value = nn.Conv1d(c_t, c_i, kernel_size=1)

        # Fusion weight projection
        self.proj_w = nn.Sequential(nn.Conv3d(c_i, c_i, kernel_size=1),
                                    nn.InstanceNorm3d(c_i))

        # skit projection
        self.proj_img = nn.Sequential(nn.Conv3d(c_i, c_i, kernel_size=1),
                                      nn.InstanceNorm3d(c_i))

        # output projection
        self.proj_out = nn.Sequential(nn.Conv3d(c_i, c_i, kernel_size=1),
                                      nn.ReLU())

        # gating to balance the language info
        self.gate = nn.Sequential(nn.Conv3d(c_i, c_i, kernel_size=1),
                                  nn.ReLU(),
                                  nn.Conv3d(c_i, c_i, kernel_size=1),
                                  nn.Tanh())

    def forward(self,
                img_in: torch.Tensor,
                lang_in: torch.Tensor,
                lang_mask: torch.Tensor = None):

        # img_in shape:  (B, C_i, H*W*(D))
        # lang_in shape: (B, C_t, SeqLen)
        # lang_mask shape: (B, SeqLen)

        assert len(img_in.shape ) == 5, "img_in shape should be (B, C_i, H, W) or (B, C_i, D, H, W)"
        assert len(lang_in.shape) == 3, "lang_in shape should be (B, C_t, SeqLen)"

        if lang_mask:
            assert len(lang_mask.shape) == 2, "lang_mask shape should be (B, SeqLen)"
            assert lang_in.shape[-1] == lang_mask.shape[-1], "lang_in and lang_mask should have the same sequence length."

        else:
            lang_mask = torch.ones([lang_in.shape[0], lang_in.shape[-1]], requires_grad=False)

        lang_mask = lang_mask.to(img_in.device)

        b = img_in.shape[0]
        d = torch.prod(torch.tensor(img_in.shape[-3:]))
        l = lang_in.shape[-1]
        lang_mask = lang_mask.unsqueeze(1)

        # cross attention
        query = self.f_query(img_in).reshape(b, self.c_i, -1)
        query = rearrange(query, "b c d -> b d c")
        key = self.f_key(lang_in) * lang_mask
        value = self.f_value(lang_in) * lang_mask

        query = query.reshape(b, d, self.num_heads, self.c_i//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, D*H*W, self.c_i//self.num_heads)
        key = key.reshape(b, self.num_heads, self.c_i//self.num_heads, l)
        # (b, num_heads, self.c_i//self.num_heads, seq_len)
        value = value.reshape(b, self.num_heads, self.c_i//self.num_heads, l)
        # (b, num_heads, self.c_i//self.num_heads, seq_len)
        lang_mask = lang_mask.unsqueeze(1)  # (b, 1, 1, seq_len)

        sim_map = torch.matmul(query, key)  # (b, num_heads, H*W*(D), l)
        sim_map = ((self.c_i / self.num_heads) ** -.5) * sim_map  # scaled dot product
        sim_map = sim_map + (1e4*lang_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (b, num_heads, H*W*(D), l)
        x = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (b, num_heads, H*W*(D), c_i//num_heads)
        x = x.permute(0, 2, 1, 3).contiguous().reshape(b, d, self.c_i)  # (B, H*W*(D), c_i)
        x = x.permute(0, 2, 1)  # (b, c_i, d)

        # fusion
        skip = self.proj_img(img_in)
        weight = self.proj_w(x.permute(0, 2, 1).view_as(img_in))
        out = self.proj_out(skip*weight)  # (b, c_i, d)

        # gating
        gate = self.gate(out)
        out = out * gate

        return out
 

class LanguageQueryBranch(torch.nn.Module):

    def __init__(self,
                 c_i: int,
                 c_t: int,
                 image_size: Sequence[int],
                 seq_length: int,
                 num_heads: int=1) -> None:

        super().__init__()
        self.c_i = c_i
        self.c_t = c_t
        self.num_heads = num_heads
        self.image_size = image_size
        self.seq_length = seq_length
        self.scale = self.c_i ** -0.5

        # project in (optional maybe let's try)
        self.img_proj_in = nn.Sequential(nn.Conv3d(c_i, c_i, kernel_size=3, stride=2, padding=1),
                                         nn.InstanceNorm3d(c_i),
                                         nn.ReLU(),
                                         nn.Conv3d(c_i, c_i, kernel_size=3, stride=2, padding=1),
                                         nn.InstanceNorm3d(c_i),
                                         nn.ReLU(),)

        # projection for language features as query
        self.d_fmap = int(np.prod(image_size) // 2**6) if len(image_size) == 3 else int(np.prod(image_size) // 2**4)
        self.f_query = nn.Conv1d(c_t, self.d_fmap, kernel_size=1)

        # Projections for visual features as key and value
        self.f_key = nn.Sequential(nn.Conv3d(c_i, c_i, kernel_size=1),
                                   nn.InstanceNorm3d(c_i))

        self.f_value = nn.Sequential(nn.Conv3d(c_i, c_i, kernel_size=1),
                                     nn.InstanceNorm3d(c_i))

        # weight projection
        self.proj_w = nn.Conv1d(self.d_fmap, self.d_fmap, kernel_size=1)

        # skip projection of language features
        self.proj_lang = nn.Conv1d(c_t, self.d_fmap, kernel_size=1)
        self.relu = nn.ReLU()

        # gating to balance the image info
        self.gate = nn.Conv1d(self.d_fmap, self.d_fmap, kernel_size=1)
        self.tanh = nn.Tanh()

        # project out to summarize sequence infon
        self.proj_out = nn.Conv1d(self.seq_length, c_i, kernel_size=1)

    def forward(self,
                img_in: torch.Tensor,
                lang_in: torch.Tensor,
                lang_mask: torch.Tensor = None):

        # img_in shape:  (B, C_i, H*W*(D))
        # lang_in shape: (B, C_t, SeqLen)
        # lang_mask shape: (B, SeqLen)

        assert len(img_in.shape) == 5, "img_in shape should be (B, C_i, H, W) or (B, C_i, D, H, W)"
        assert len(lang_in.shape) == 3, "lang_in shape should be (B, C_t, SeqLen)"

        if lang_mask:
            assert len(lang_mask.shape) == 2, "lang_mask shape should be (B, SeqLen)"
            assert lang_in.shape[-1] == lang_mask.shape[-1], "lang_in and lang_mask should have the same sequence length."

        else:
            lang_mask = torch.ones([lang_in.shape[0], lang_in.shape[-1]], requires_grad=False)
        lang_mask = lang_mask.to(img_in.device)

        b, _, l = lang_in.size()

        # img project in
        img_in = self.img_proj_in(img_in) # (b, c_i, H//2, W//2, D//2)

        lang_mask = lang_mask.unsqueeze(1)  # (b, 1, seq_len)

        # cross attention
        query = self.f_query(lang_in) * lang_mask  # (b, d_fmap, l)
        key = self.f_key(img_in)  # (b, c_i, fmap_d, fmap_h, fmap_w)
        value = self.f_value(img_in)  # (b, c_i, fmap_d, fmap_h, fmap_w)
        fd, fw, fh = key.size()[-3:]

        query = query.reshape(b, self.num_heads, self.d_fmap // self.num_heads, l) # (b, num_heads, d_fmap//num_heads, l)
        query = rearrange(query, "b h d l -> b h l d")
        key = key.reshape(b, self.c_i, self.num_heads, self.d_fmap//self.num_heads) # (b, c_i, num_heads, d_fmap//num_heads)
        key = rearrange(key, "b c h d -> b h d c")
        value = value.reshape(b, self.c_i, self.num_heads, self.d_fmap//self.num_heads) # same as key
        value = rearrange(value, "b c h d -> b h d c")

        lang_mask = lang_mask.unsqueeze(-1)  # (b, 1, seq_len)

        sim_map = torch.matmul(query, key)  # (b, num_heads, l, c_i)
        sim_map = (self.d_fmap ** -.5) * sim_map  # scaled dot product
        sim_map = sim_map + (1e4*lang_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (b, num_heads, l, c_i)
        x = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (b, num_heads, l, d_fmap//num_heads)
        x = x.permute(0, 2, 1, 3).contiguous().reshape(b, l, self.d_fmap).transpose(-2, -1)  # (B, d_fmap, l)

        # fusion
        weight = self.proj_w(x) # (B, d_fmap, l)
        skip = self.proj_lang(lang_in) # (B, d_fmap, l)
        out = self.relu(skip * weight)

        # gating
        gate = self.gate(out)  # (B, d_fmap, l)
        out = out * self.tanh(gate)
        out = rearrange(out, "b d l -> b l d")
        out = self.proj_out(out)
        out = out.reshape(b, self.c_i, fd, fw, fh)

        # if project in then interpolate back to original size. From coarse fmap to original size
        out = F.interpolate(out, size=self.image_size, mode="trilinear", align_corners=False)

        return out
    

