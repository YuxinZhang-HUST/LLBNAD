
import torch
import torch.nn as nn
from functools import partial
from .vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from .._base_model import BaseModel
from . import vit_encoder

from .uad import ViTill, ViTillv2

VALID_ENCODER_LIST = [
	'dinov2reg_vit_small_14',
	'dinov2reg_vit_base_14',
	'dinov2reg_vit_large_14',
	'dinov2_vit_base_14'
	'dino_vit_base_16'
	'ibot_vit_base_16'
	'mae_vit_base_16'
	'beitv2_vit_base_16'
	'beit_vit_base_16'
	'digpt_vit_base_16'
	'deit_vit_base_16'
]
class Dinomaly(BaseModel):
	def __init__(self, encoder_name='dinov2reg_vit_base_14'):
		super(Dinomaly, self).__init__()

		assert encoder_name in VALID_ENCODER_LIST, f'Only support {VALID_ENCODER_LIST} for now but got {encoder_name}'

		target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
		fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
		fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

		# target_layers = list(range(4, 19))

		encoder = vit_encoder.load(encoder_name)

		if 'small' in encoder_name:
			embed_dim, num_heads = 384, 6
		elif 'base' in encoder_name:
			embed_dim, num_heads = 768, 12
		elif 'large' in encoder_name:
			embed_dim, num_heads = 1024, 16
			target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
		else:
			raise "Architecture not in small, base, large."

		bottleneck = []
		decoder = []

		bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))#原来这里是drop=0.2，但是Dinoamly文章对realiad数据集取0.4##
		# bottleneck.append(nn.Sequential(FeatureJitter(scale=40),
		#                                 bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)))

		bottleneck = nn.ModuleList(bottleneck)

		for i in range(8):
			blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
						   qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
						   attn=LinearAttention2)
			# blk = ConvBlock(dim=embed_dim, kernel_size=7, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-8))
			decoder.append(blk)
		decoder = nn.ModuleList(decoder)

		self.model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
					   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder,
					   fuse_layer_decoder=fuse_layer_decoder)

		self.set_frozen_layers(['encoder'])


	def forward(self, imgs):
		en, de = self.model(imgs)
		# en=torch.stack(en, dim=0)
		# de=torch.stack(de, dim=0)
		# print("Shape of en:", en.shape)
		# print("Shape of de:", de.shape)
		# en = [en[i] for i in range(en.size(0))]
		return en, de