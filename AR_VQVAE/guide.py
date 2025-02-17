import fairseq

import torch
import torchaudio
import torch as th
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.distributions import Categorical
from transformers import BertModel

from typing import Callable, List

from model.modules.rotary_embedding_torch import RotaryEmbedding
from model.modules.transformer_modules import DecoderLayerStack, FiLMTransformerDecoderLayer

def get_TextEncoder():
    Text_encoder = BertModel.from_pretrained('bert-base-chinese')
    # freeze
    for param in Text_encoder.parameters():
        param.requires_grad = False
    return Text_encoder

def get_AudioEncoder() -> ("Audio2LipRegressionTransformer", torchaudio.transforms.Resample):
    checkpoint_path = "./vq-wav2vec.pt"
    audio_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    audio_model = audio_model[0]
    # freeze
    for param in audio_model.parameters():
        param.requires_grad = False
    audio_model.eval()
    audio_resampler = torchaudio.transforms.Resample(48000, 16000) # 采样
    return audio_model, audio_resampler

# dropout mask
def prob_mask_like(shape, prob, device):
    if prob == 1: 
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else: 
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

class GuideTransformer(nn.Module):

    def __init__(self, tokens: int, num_heads: int = 4, num_layers: int = 4, dim: int = 512, ff_size: int = 1024, dropout: float = 0.1, 
            activation: Callable = F.gelu, use_rotary: bool = True, audio_feature_dim: int = 512, emb_len: int = 798, num_audio_layers: int = 2):
        super(GuideTransformer, self).__init__()
        self.tokens = tokens
        self.token_embedding = nn.Embedding(
            num_embeddings = tokens + 1,  # account for sequence start and end tokens
            embedding_dim = dim)
        
        
        self.Text_encoder = get_TextEncoder() # get and freeze Text Encoder
        self.audio_model, self.audio_resampler = get_AudioEncoder() 

        
        self.pre_audio = self.get_process_audio_models(audio_feature_dim, num_audio_layers) 

        
        self.audio_projection = nn.Linear(audio_feature_dim, dim)
        self.text_projection = nn.Linear(768, dim)

        
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim))
        
        
        self.rand_null_cond_embed = nn.Parameter(torch.randn(1, emb_len, dim))
        self.rand_null_cond_hidden = nn.Parameter(torch.randn(1, dim))

        # layerNorm audio
        self.norm_cond = nn.LayerNorm(dim)

        
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=dim)

        # autoregressive
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(dim, num_heads, dim_feedforward = ff_size, dropout = dropout,
                    activation = activation, batch_first = True, rotary = self.rotary)
            )
        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.final_layer = nn.Linear(dim, tokens)

    
    def get_process_audio_models(self, cond_feature_dim: int, num_audio_layers: int):
        pre_layers = []
        for _ in range(num_audio_layers):
            pre_layers += self._build_single_audio_conv(cond_feature_dim)
        pre_layers += [torch.nn.Conv1d(cond_feature_dim, cond_feature_dim, kernel_size=1)]
        pre_layers = torch.nn.ModuleList(pre_layers)
        process_audio = nn.Sequential(*pre_layers)
        return process_audio
    
    
    def _build_single_audio_conv(self, c: int) -> List[nn.Module]:
        return [torch.nn.Conv1d(c, max(256, c), kernel_size=3, dilation=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(256, c), max(256, c), kernel_size=3, dilation=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(128, c), max(128, c), kernel_size=3, dilation=3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(128, c), c, kernel_size=3, dilation=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(c, c, kernel_size=3, dilation=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(c, c, kernel_size=3, dilation=3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2)]

    # get autoregressive mask
    def get_tgt_mask(self, size: int, device: str) -> torch.tensor:
        
        mask = torch.tril(torch.ones((size, size), device = device) == 1)
        mask = mask.float() # True -> 1, False -> 0
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask

   
    def encode_audio(self, raw_audio: torch.Tensor) -> torch.Tensor:
        a0 = self.audio_resampler(raw_audio[:, :, 0])  # B x T # 采样
        with torch.no_grad():
            z0 = self.audio_model.feature_extractor(a0)
        return z0    
    
    def forward(self, tokens: th.Tensor, text_ids: th.Tensor, text_masks: th.Tensor, Text_tokenType: th.Tensor, 
            audio_condition: th.Tensor, cond_drop_prob: float = 0.0) -> torch.Tensor:
        ''' 
        param: tokens: [B T*residual_depth]
        param: text_ids: [B, max_len]
        param: text_masks: [B, max_len]
        param: Text_tokenType: [B, max_len]
        param: audio_condition: [B, 480000, 2]
        param: cond_drop_prob: the rate of ramdom select
        '''
        
        # process motion
        batch_size, device = tokens.shape[0], tokens.device
        target = self.token_embedding(tokens) 
        tgt_mask = self.get_tgt_mask(target.shape[1], target.device) 

        # process audio
        audio_embed = self.encode_audio(audio_condition) 
        audio_tokens = self.pre_audio(audio_embed).permute(0, 2, 1) 
        audio_tokens = self.audio_projection(audio_tokens) 

        
        keep_mask = prob_mask_like((batch_size), 1 - cond_drop_prob, device = device) 

        # random sample audio_token
        rand_null_audio_embed = self.rand_null_cond_embed.to(audio_tokens.dtype)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        audio_tokens = torch.where(keep_mask_embed, audio_tokens, rand_null_audio_embed[:, :audio_tokens.shape[1], :]) # cond_tokens: [B 1950 dim]
        mean_pooled_cond_tokens = audio_tokens.mean(dim=-2) 
        audio_tokens = self.norm_cond(audio_tokens) # nn.LayerNorm

        # random sample audio_hiden
        audio_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens) 
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")
        rand_null_cond_hidden = self.rand_null_cond_hidden.to(audio_tokens.dtype)
        audio_hidden = torch.where(keep_mask_hidden, audio_hidden, rand_null_cond_hidden) 

        # process text
        text_embed = self.Text_encoder(input_ids = text_ids, attention_mask = text_masks, token_type_ids = Text_tokenType).last_hidden_state
        # text_embed = self.Text_encoder(input_ids = text_ids, attention_mask = text_masks, token_type_ids = Text_tokenType).last_hidden_state[:, 0] # if ues [cls] token [B 768] should unsqueeze(1)
        text_tokens = self.text_projection(text_embed) # [B, dim]

        # autoregressive
        output = self.seqTransDecoder(target, tgt_mask = tgt_mask, audio_token = audio_tokens, audio_hidden = audio_hidden, text_token = text_tokens)
        output = self.final_layer(output) # linear
        return output
    
    def generate(self, text_ids: th.Tensor, text_masks: th.Tensor, Text_tokenType: th.Tensor, 
            audio_condition: th.Tensor, sequence_length: int, residual_depth: int = 4, batchsize: int = 1, top_p: float = 0.94, pre_tokens: th.Tensor = None) -> th.Tensor:
        with torch.no_grad():
            input_tokens = torch.zeros(batchsize, 1, dtype=th.int64).to(audio_condition.device) + self.tokens
            
            if pre_tokens != None:
                input_tokens = torch.cat([input_tokens, pre_tokens], dim=-1)

            
            for _ in range(sequence_length*residual_depth):
                curr_input_tokens = input_tokens
                logits = self.forward(tokens = curr_input_tokens, text_ids = text_ids, text_masks = text_masks, Text_tokenType = Text_tokenType, audio_condition=audio_condition)
                logits = logits[:, -1, :]  

                
                one_hot = torch.nn.functional.softmax(logits, dim=-1)
                sorted_probs, indices = torch.sort(one_hot, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumulative_probs < top_p
                nucleus = torch.cat(
                    [
                        nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                        nucleus[..., :-1],
                    ],
                    dim = -1,
                )
                sorted_probs[~nucleus] = 0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                dist = Categorical(sorted_probs)
                idx = dist.sample()
                tokens = indices.gather(-1, idx.unsqueeze(-1))
                
                input_tokens = torch.cat([input_tokens, tokens], dim=-1)
            
            
            remove_num = 1+pre_tokens.shape[0] if pre_tokens != None else 1
            tokens = input_tokens[:, remove_num:].contiguous()
            return tokens