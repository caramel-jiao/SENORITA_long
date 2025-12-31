import torch
import numpy as np
import os
from glob import glob
import argparse
import math

from control_cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from control_cogvideox.controlnet_cogvideox_transformer_3d import ControlCogVideoXTransformer3DModel
from pipeline_cogvideox_controlnet_5b_i2v_instruction2 import ControlCogVideoXPipeline
from diffusers.utils import export_to_video, load_video
from diffusers import AutoencoderKLCogVideoX
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import CogVideoXDDIMScheduler

from omegaconf import OmegaConf
from transformers import T5EncoderModel
from einops import rearrange
from typing import List
from tqdm import tqdm
from pathlib import Path 
import json
from collections import OrderedDict

import PIL
from PIL import Image
from torchvision import transforms


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_prompt(file:str):
    with open(file,'r') as f:
        a=f.readlines()
    return a #a[0]:positive prompt, a[1] negative prompt
def unwarp_model(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.split('module.')[1]] = state_dict[key]
    return new_state_dict

def init_pipe():

    i2v=True

    if i2v:
        key = "i2v"
    else:
        key = "t2v"
    noise_scheduler = CogVideoXDDIMScheduler(
        **OmegaConf.to_container(
            OmegaConf.load(f"./cogvideox-5b-{key}/scheduler/scheduler_config.json")
        )
    )

    text_encoder = T5EncoderModel.from_pretrained(f"./cogvideox-5b-{key}/", subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(f"./cogvideox-5b-{key}/", subfolder="vae", torch_dtype=torch.float16)
    tokenizer = T5Tokenizer.from_pretrained(f"./cogvideox-5b-{key}/tokenizer", torch_dtype=torch.float16)


    config = OmegaConf.to_container(
        OmegaConf.load(f"./cogvideox-5b-{key}/transformer/config.json")
    )
    if i2v:
        config["in_channels"] = 32
    else:
        config["in_channels"] = 16
    transformer = CogVideoXTransformer3DModel(**config)

    control_config = OmegaConf.to_container(
        OmegaConf.load(f"./cogvideox-5b-{key}/transformer/config.json")
    )
    if i2v:
        control_config["in_channels"] = 32
    else:
        control_config["in_channels"] = 16
    control_config['num_layers'] = 6
    control_config['control_in_channels'] = 16
    controlnet_transformer = ControlCogVideoXTransformer3DModel(**control_config)

    all_state_dicts = torch.load("./senorita-2m/models_half/ff_controlnet_half.pth", map_location="cpu",weights_only=True)
    transformer_state_dict = unwarp_model(all_state_dicts["transformer_state_dict"])
    controlnet_transformer_state_dict = unwarp_model(all_state_dicts["controlnet_transformer_state_dict"])

    transformer.load_state_dict(transformer_state_dict, strict=True)
    controlnet_transformer.load_state_dict(controlnet_transformer_state_dict, strict=True)

    transformer = transformer.half()
    controlnet_transformer = controlnet_transformer.half()

    vae = vae.eval()
    text_encoder = text_encoder.eval()
    transformer = transformer.eval()
    controlnet_transformer = controlnet_transformer.eval()

    pipe = ControlCogVideoXPipeline(tokenizer,
            text_encoder,
            vae,
            transformer,
            noise_scheduler,
            controlnet_transformer,
    )

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    return pipe
    

def inference(source_images, 
        target_images, 
        text_prompt, negative_prompt, 
        pipe, vae, guidance_scale, 
        h, w, random_seed)->List[PIL.Image.Image]:
    torch.manual_seed(random_seed)
    
    pipe.vae.to(DEVICE)
    pipe.transformer.to(DEVICE)
    pipe.controlnet_transformer.to(DEVICE)

    source_pixel_values = source_images/127.5 - 1.0
    source_pixel_values = source_pixel_values.to(torch.float16).to(DEVICE)
    if target_images is not None:
        target_pixel_values = target_images/127.5 - 1.0
        target_pixel_values = target_pixel_values.to(torch.float16).to(DEVICE)
    bsz, f, h, w, c = source_pixel_values.shape

    with torch.no_grad():
        source_pixel_values = rearrange(source_pixel_values, "b f w h c -> b c f w h")
        source_latents = vae.encode(source_pixel_values).latent_dist.sample()
        source_latents = source_latents.to(torch.float16)
        source_latents = source_latents * vae.config.scaling_factor
        source_latents = rearrange(source_latents, "b c f h w -> b f c h w")

        if target_images is not None:
            target_pixel_values = rearrange(target_pixel_values, "b f w h c -> b c f w h")
            images = target_pixel_values[:,:,:1,...]
            image_latents = vae.encode(images).latent_dist.sample()
            image_latents = image_latents.to(torch.float16)
            image_latents = image_latents * vae.config.scaling_factor
            image_latents = rearrange(image_latents, "b c f h w -> b f c h w")
            image_latents = torch.cat([image_latents, torch.zeros_like(source_latents)[:,1:]],dim=1)
            latents = torch.cat([image_latents, source_latents], dim=2)
        else:
            image_latents = None
            latents = source_latents

    video = pipe(
        prompt = text_prompt,
        negative_prompt = negative_prompt,
        video_condition = source_latents, # input to controlnet
        video_condition2 = image_latents, # concat with latents
        height = h,
        width = w,
        num_frames = f,
        num_inference_steps = 30,
        interval = 6,
        guidance_scale = guidance_scale,
        generator = torch.Generator(device=DEVICE).manual_seed(random_seed)
    ).frames[0]

    return video

def process_video(
    pipe, video_file, image_file, 
    positive_prompt, negative_prompt, 
    guidance, random_seed
):
    model_h = 480 # 448
    model_w = 832 # 768
    model_size = (model_w, model_h)
    shard_stride = 32   # 每一段开头都有一帧是喝trg_img对应的ref

    # get image
    trg_image = Image.open(image_file)

    # load video
    pil_images = load_video(video_file)
    src_w, src_h = pil_images[0].size
    n_frames = len(pil_images)
    resized_frames = [img.resize(model_size) for img in pil_images]
    images = torch.from_numpy(np.array(resized_frames))

    source_images = images[None, ...]
    
    video: List[PIL.Image.Image] = [trg_image]

    for i in tqdm(range(math.ceil((n_frames - 1) / shard_stride))):
        # first frame guidence
        first_frame = transforms.ToTensor()(video[-1])
        first_frame = first_frame * 255.0
        first_frame = rearrange(first_frame, "c w h -> w h c")
        target_image = first_frame[None, None, ...]
            
        video += inference(
            source_images[:, i * shard_stride: (i + 1) * shard_stride + 1],
            target_image, positive_prompt,
            negative_prompt, pipe, pipe.vae,
            guidance,
            model_h, model_w, random_seed
        )[1: ]

    video = [image.resize((src_w, src_h)) for image in video]

    return video


'''
python long_eval.py --edit_img_root '/home/gljiao/projects/aip-rjliao/gljiao/FramePackEdit/outputs/first_frame_edit/alpha=0.80-omega=5.00-step=15-model=FLUX.1-dev'
python long_eval.py --edit_img_root '/home/gljiao/projects/aip-rjliao/gljiao/FramePackEdit/outputs/first_frame_edit/alpha=0.60-omega=5.00-step=25-model=stable-diffusion-3-medium-diffusers'
python long_eval.py --guidance 2.0 --edit_img_root '/home/gljiao/projects/aip-rjliao/gljiao/FramePackEdit/outputs/first_frame_edit/alpha=0.80-omega=5.00-step=15-model=FLUX.1-dev'
python long_eval.py --guidance 2.0 --edit_img_root '/home/gljiao/projects/aip-rjliao/gljiao/FramePackEdit/outputs/first_frame_edit/alpha=0.60-omega=5.00-step=25-model=stable-diffusion-3-medium-diffusers'
python long_eval.py --guidance 6.0 --edit_img_root '/home/gljiao/projects/aip-rjliao/gljiao/FramePackEdit/outputs/first_frame_edit/alpha=0.80-omega=5.00-step=15-model=FLUX.1-dev'
python long_eval.py --guidance 6.0 --edit_img_root '/home/gljiao/projects/aip-rjliao/gljiao/FramePackEdit/outputs/first_frame_edit/alpha=0.60-omega=5.00-step=25-model=stable-diffusion-3-medium-diffusers'
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='../longvideobench')
    parser.add_argument("--save_root", type=str, default='./long_bench_eval')
    parser.add_argument("--edit_img_root", type=str, default='/home/gljiao/projects/aip-rjliao/gljiao/FramePackEdit/outputs/first_frame_edit/alpha=0.80-omega=5.00-step=25-model=FLUX.1-dev')

    parser.add_argument("--guidance", type=float, default=4.0)

    args = parser.parse_args()
    save_root = os.path.join(
        args.save_root,
        'edit_img_root=%s' % os.path.basename(args.edit_img_root),
        'guidance=%.1f' % args.guidance,
    )

    pipe = init_pipe()
    neg_prompt = "Bad quality."

    # load dataset
    json_path_list = sorted(glob(os.path.join(args.data_root, 'json_format', '*.json')))

    for json_path in tqdm(json_path_list):
        json_data = read_json(json_path)
        rel_video = os.path.basename(json_path).replace('.json', '.mp4')
        video_id = int(rel_video.replace('.mp4', ''))
        video_path = os.path.join(args.data_root, 'videos', rel_video)

        save_folder = os.path.join(save_root, '%d' % video_id)
        os.makedirs(save_folder, exist_ok=True)

        # loop target prompt
        for trg_data in json_data['part_b_target_json']['edits']:
            trg_key = trg_data['editing_type']
            trg_prompt = trg_data['target_prompt']
            # src_prompt = trg_data['source_prompt']

            save_path = os.path.join(save_folder, '%s.mp4' % trg_key)
            if os.path.exists(save_path):
                continue

            edit_img_path = os.path.join(args.edit_img_root, '%d-%s.png' % (video_id, trg_key))
            edit_video = process_video(
                pipe, video_path, edit_img_path, 
                trg_prompt, neg_prompt, 
                args.guidance, 42
            )

            torch.cuda.empty_cache()
            export_to_video(edit_video, save_path, fps=16)
