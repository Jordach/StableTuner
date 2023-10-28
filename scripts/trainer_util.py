
from typing import Iterable
import torch
import torch.utils.checkpoint
from diffusers import DiffusionPipeline
from torchvision import transforms
from PIL import Image, ImageFilter
from pathlib import Path
from einops import rearrange
from torch import einsum
import math
import diffusers
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
# FlashAttention based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main
# /memory_efficient_attention_pytorch/flash_attention.py LICENSE MIT
# https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE constants
EPSILON = 1e-6

# Add noise to latents
@torch.jit.script
def get_noisy_latents(batch, noise_scheduler, with_pertubation_noise, perturbation_weight, model_variant):
    with torch.no_grad():
        latent_dist = batch[0][0]
        latents = latent_dist.sample() * 0.18215
        #if args.model_variant == 'inpainting':
            #conditioning_latent_dist = batch[0][2]
            #mask = batch[0][3]
            #conditioning_latents = conditioning_latent_dist.sample() * 0.18215
        #if args.model_variant == 'depth2img':
            #depth = batch[0][3]

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    if with_pertubation_noise:
        # https://arxiv.org/pdf/2301.11706.pdf
        noisy_latents = noise_scheduler.add_noise(latents, noise + perturbation_weight * torch.randn_like(latents), timesteps)
    else:
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps, device=latents.device)
    return timesteps, latents, noisy_latents, noise, bsz

# Get the loss for this batch
@torch.jit.script
def get_batch_loss(noise_scheduler, latents, noise, timesteps, model_pred, use_msnr, msnr_gamma, force_v_pred, accelerator):
    if noise_scheduler.config.prediction_type == "epsilon" and not force_v_pred:
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction" or force_v_pred:
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    #if args.with_prior_preservation:
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        """
        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
        noise, noise_prior = torch.chunk(noise, 2, dim=0)

        # Compute instance loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

        # Compute prior loss
        prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

        # Add the prior loss to the instance loss.
        loss = loss + args.prior_loss_weight * prior_loss
        """
        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        #model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        #target, target_prior = torch.chunk(target, 2, dim=0)
        #loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
        #prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
    #else:
    if use_msnr:
        if msnr_gamma == None:
            msnr_gamma = 5

        are_we_v_pred = False
        if force_v_pred:
            are_we_v_pred = True
        elif noise_scheduler.config.prediction_type == "v_prediction":
            are_we_v_pred = True

        loss = (target.float() - model_pred.float()) ** 2
        loss = loss.mean([1, 2, 3])
        loss = apply_snr_weight_neo(are_we_v_pred, loss.float(), timesteps, noise_scheduler, msnr_gamma, accelerator)
        loss = loss.mean()
        return loss
    else:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

# Text encoder inference without gradients:
def text_encoder_inference(text_enc_context, batch, text_encoder, tokenizer, clip_skip, accelerator, max_standard_tokens, token_chunks_limit):
    with torch.no_grad():
        tru_len = max(len(x) for x in batch[0][1])
        max_chunks = np.ceil(tru_len / max_standard_tokens).astype(int)
        max_len = max_chunks.item() * max_standard_tokens
        clamp_event = False
        #print(f"\n\n\nC:{max_chunks}, L:{max_len}, T:{tru_len}")
        # If we're a properly padded bunch of tokens that have come from the tokeniser padder, train normally;
        # otherwise we're handling a dropout batch, and thusly need to handle it the normal way. 
        if tru_len == max_len and max_chunks > 1:
            # Duplicate batch tensor to prevent irreparably damaging it's token data
            # Recommended over torch.tensor()
            n_batch = batch[0][1].clone().detach()
            z = None
            for i, x in enumerate(n_batch):
                if len(x) < max_len:
                    n_batch[i] = [*x, *np.full((max_len - len(x)), tokenizer.eos_token_id)]
                del i
                del x

            chunks = [n_batch[:, i:i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)]
            clamp_chunk = 0
            for chunk in chunks:
                # Hard limit the tokens to fit in memory for the rare event that latent caches that somehow exceed the limit.
                if clamp_chunk > (token_chunks_limit):
                    del chunk
                    break

                chunk = chunk.to(accelerator.device)
                chunk = torch.cat((torch.full((chunk.shape[0], 1), tokenizer.bos_token_id).to(accelerator.device), chunk, torch.full((chunk.shape[0], 1), tokenizer.eos_token_id).to(accelerator.device)), 1)
                if z is None:
                    if clip_skip:
                        encode = text_encoder(chunk, output_hidden_states=True)
                        z = text_encoder.text_model.final_layer_norm(encode['hidden_states'][-2])
                        del encode
                    else:
                        encode = text_encoder(chunk, output_hidden_states=True)
                        z = text_encoder.text_model.final_layer_norm(encode['hidden_states'][-1])
                        del encode
                else:
                    if clip_skip:
                        encode = text_encoder(chunk, output_hidden_states=True)
                        z = torch.cat((z, text_encoder.text_model.final_layer_norm(encode['hidden_states'][-2])), dim=-2)
                        del encode
                    else:
                        encode = text_encoder(chunk, output_hidden_states=True)
                        z = torch.cat((z, text_encoder.text_model.final_layer_norm(encode['hidden_states'][-1])), dim=-2)
                        del encode

                clamp_chunk += 1
                del chunk
            encoder_hidden_states = torch.stack(tuple(z))
            del n_batch
            del tru_len
            del max_chunks
            del max_len
            del z
            del chunks
            del clamp_chunk
        else:
            if clip_skip == True:
                encoder_hidden_states = text_encoder(batch[0][1],output_hidden_states=True)
                encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
            else:
                encoder_hidden_states = text_encoder(batch[0][1])[0]
        del clamp_event
        return encoder_hidden_states

# Text encoder training:
def text_encoder_training(text_enc_context, batch, text_encoder, tokenizer, clip_skip, accelerator, max_standard_tokens, token_chunks_limit):
    with text_enc_context:
        tru_len = max(len(x) for x in batch[0][1])
        max_chunks = np.ceil(tru_len / max_standard_tokens).astype(int)
        max_len = max_chunks.item() * max_standard_tokens
        clamp_event = False
        #print(f"\n\n\nC:{max_chunks}, L:{max_len}, T:{tru_len}")
        # If we're a properly padded bunch of tokens that have come from the tokeniser padder, train normally;
        # otherwise we're handling a dropout batch, and thusly need to handle it the normal way. 
        if tru_len == max_len and max_chunks > 1:
            # Duplicate batch tensor to prevent irreparably damaging it's token data
            # Recommended over torch.tensor()
            n_batch = batch[0][1].clone().detach()
            z = None
            for i, x in enumerate(n_batch):
                if len(x) < max_len:
                    n_batch[i] = [*x, *np.full((max_len - len(x)), tokenizer.eos_token_id)]
                del i
                del x

            chunks = [n_batch[:, i:i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)]
            clamp_chunk = 0
            for chunk in chunks:
                # Hard limit the tokens to fit in memory for the rare event that latent caches that somehow exceed the limit.
                if clamp_chunk > (token_chunks_limit):
                    del chunk
                    break

                chunk = chunk.to(accelerator.device)
                chunk = torch.cat((torch.full((chunk.shape[0], 1), tokenizer.bos_token_id).to(accelerator.device), chunk, torch.full((chunk.shape[0], 1), tokenizer.eos_token_id).to(accelerator.device)), 1)
                if z is None:
                    if clip_skip:
                        encode = text_encoder(chunk, output_hidden_states=True)
                        z = text_encoder.text_model.final_layer_norm(encode['hidden_states'][-2])
                        del encode
                    else:
                        encode = text_encoder(chunk, output_hidden_states=True)
                        z = text_encoder.text_model.final_layer_norm(encode['hidden_states'][-1])
                        del encode
                else:
                    if clip_skip:
                        encode = text_encoder(chunk, output_hidden_states=True)
                        z = torch.cat((z, text_encoder.text_model.final_layer_norm(encode['hidden_states'][-2])), dim=-2)
                        del encode
                    else:
                        encode = text_encoder(chunk, output_hidden_states=True)
                        z = torch.cat((z, text_encoder.text_model.final_layer_norm(encode['hidden_states'][-1])), dim=-2)
                        del encode

                clamp_chunk += 1
                del chunk
            encoder_hidden_states = torch.stack(tuple(z))
            del n_batch
            del tru_len
            del max_chunks
            del max_len
            del z
            del chunks
            del clamp_chunk
        else:
            if clip_skip == True:
                encoder_hidden_states = text_encoder(batch[0][1],output_hidden_states=True)
                encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
            else:
                encoder_hidden_states = text_encoder(batch[0][1])[0]
        del clamp_event
        return encoder_hidden_states

# Predict with unet
def predict_unet_noise(noisy_latents, timesteps, encoder_hidden_states, unet):
    #if args.model_variant == 'inpainting':
        #if random.uniform(0, 1) < 0.25:
            # for some steps, predict the unmasked image
            #conditioning_latents = torch.stack([extra_latent[tuple([latents.shape[3]*8, latents.shape[2]*8])].squeeze()] * bsz)
            #mask = torch.ones(bsz, 1, latents.shape[2], latents.shape[3]).to(accelerator.device, dtype=weight_dtype)

        #noisy_inpaint_latents = torch.concat([noisy_latents, mask, conditioning_latents], 1)
        #model_pred = unet(noisy_inpaint_latents, timesteps, encoder_hidden_states).sample
    #elif args.model_variant == 'depth2img':
        #noisy_latents = torch.cat([noisy_latents, depth], dim=1)
        #model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, depth).sample
    #else: #args.model_variant == "base":
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    return model_pred

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# Min SNR related:
@torch.jit.script
def apply_snr_weight_neo(is_v_prediction, loss, timesteps, noise_scheduler, gamma, accelerator):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.tensor(gamma, dtype=torch.float))
    if is_v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(accelerator.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(accelerator.device)
    loss = loss * snr_weight
    return loss.to(accelerator.device)

# Zero SNR related:
def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)

def enforce_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()
    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    return 1 - alphas

# flash attention forwards and backwards
# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.function.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 2 in the paper """

        device = q.device
        dtype = q.dtype
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros(
            (*q.shape[:-1], 1), dtype=dtype, device=device)
        all_row_maxes = torch.full(
            (*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

        scale = (q.shape[-1] ** -0.5)

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, 'b n -> b 1 1 n')
            mask = mask.split(q_bucket_size, dim=-1)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum(
                    '... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                             device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.)

                block_row_sums = exp_weights.sum(
                    dim=-1, keepdims=True).clamp(min=EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = einsum(
                    '... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(
                    block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + \
                               exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_(
                    (exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 4 in the paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l, m = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            l.split(q_bucket_size, dim=-2),
            m.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2)
        )

        for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum(
                    '... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                             device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                exp_attn_weights = torch.exp(attn_weights - mc)

                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.)

                p = exp_attn_weights / lc

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def replace_unet_cross_attn_to_flash_attention():
    print("Using FlashAttention")

    def forward_flash_attn(self, x, context=None, mask=None):
        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)

        context = context if context is not None else x
        context = context.to(x.dtype)

        if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k = self.to_k(context_k)
        v = self.to_v(context_v)
        del context, x

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        out = FlashAttentionFunction.apply(q, k, v, mask, False,
                                           q_bucket_size, k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')

        # diffusers 0.6.0
        if type(self.to_out) is torch.nn.Sequential:
            return self.to_out(out)

        # diffusers 0.7.0
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_flash_attn
class Depth2Img:
    def __init__(self,unet,text_encoder,revision,pretrained_model_name_or_path,accelerator):
        self.unet = unet
        self.text_encoder = text_encoder
        self.revision = revision if revision != 'no' else 'fp32'
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.accelerator = accelerator
        self.pipeline = None
    def depth_images(self,paths):
        if self.pipeline is None:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                revision=self.revision,
                local_files_only=True,)
            self.pipeline.to(self.accelerator.device)
            self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        non_depth_image_files = []
        image_paths_by_path = {}
        
        for path in paths:
            #if path is list
            if isinstance(path, list):
                img = Path(path[0])
            else:
                img = Path(path)
            if self.get_depth_image_path(img).exists():
                continue
            else:
                non_depth_image_files.append(img)
        image_objects = []
        for image_path in non_depth_image_files:
            image_instance = Image.open(image_path)
            if not image_instance.mode == "RGB":
                image_instance = image_instance.convert("RGB")
            image_instance = self.pipeline.feature_extractor(
                image_instance, return_tensors="pt"
            ).pixel_values
            
            image_instance = image_instance.to(self.accelerator.device)
            image_objects.append((image_path, image_instance))
        
        for image_path, image_instance in image_objects:
            path = image_path.parent
            ogImg = Image.open(image_path)
            ogImg_x = ogImg.size[0]
            ogImg_y = ogImg.size[1]
            depth_map = self.pipeline.depth_estimator(image_instance).predicted_depth
            depth_min = torch.amin(depth_map, dim=[0, 1, 2], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[0, 1, 2], keepdim=True)
            depth_map = torch.nn.functional.interpolate(depth_map.unsqueeze(1),size=(ogImg_y, ogImg_x),mode="bicubic",align_corners=False,)           

            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            depth_map = depth_map[0,:,:]
            depth_map_image = transforms.ToPILImage()(depth_map)
            depth_map_image = depth_map_image.filter(ImageFilter.GaussianBlur(radius=1))
            depth_map_image.save(self.get_depth_image_path(image_path))
            #quit()
        return 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        
    def get_depth_image_path(self,image_path):
        #if image_path is a string, convert it to a Path object
        if isinstance(image_path, str):
            image_path = Path(image_path)
        return image_path.parent / f"{image_path.stem}-depth.png"
        
#Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14 and taken from harubaru's implementation https://github.com/harubaru/waifu-diffusion
class EMAModel:
    """
    Exponential Moving Average of models weights
    """
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]