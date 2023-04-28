#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# command python3 demo_txt2img_db.py --hf-token=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 -v
import cv2
import numpy as np
from diffusers.utils import load_image
import argparse
from cuda import cudart
import tensorrt as trt
from PIL import Image
from utilities import TRT_LOGGER, add_arguments
from txt2img_pipeline import Txt2ImgPipeline
import random
import torch

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Txt2Img Demo")
    parser = add_arguments(parser)
    parser.add_argument('--scheduler', type=str, default="EulerA", choices=["PNDM", "LMSD", "DPM", "DDIM", "EulerA"], help="Scheduler for diffusion process")
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    print("[I] Initializing StableDiffusion txt2img demo using TensorRT")
    setup_seed(1000)
    args = parseArgs()

    # Process prompt
    if not isinstance(args.prompt, list):
        raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}")
    prompt = args.prompt * args.repeat_prompt

    if not isinstance(args.negative_prompt, list):
        raise ValueError(f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}")
    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    # Validate image dimensions
    image_height = args.height
    image_width = args.width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}.")

    # Register TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    max_batch_size = 16
    # FIXME VAE build fails due to element limit. Limitting batch size is WAR
    if args.build_dynamic_shape or image_height > 512 or image_width > 512:
        max_batch_size = 4

    batch_size = len(prompt)
    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {len(prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4")
    image = load_image('https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png')
    #image = load_image('https://cdn-lfs.huggingface.co/repos/03/0a/030ada8d2e813be0895828822222c96f078a306f15cde39c42045345f7885c1f/dc0d345b20453a5f835c78ac724f511e415f3746b5d25b08e360bf940c67222c?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27boy.png%3B+filename%3D%22boy.png%22%3B&response-content-type=image%2Fpng&Expires=1682683713&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzAzLzBhLzAzMGFkYThkMmU4MTNiZTA4OTU4Mjg4MjIyMjJjOTZmMDc4YTMwNmYxNWNkZTM5YzQyMDQ1MzQ1Zjc4ODVjMWYvZGMwZDM0NWIyMDQ1M2E1ZjgzNWM3OGFjNzI0ZjUxMWU0MTVmMzc0NmI1ZDI1YjA4ZTM2MGJmOTQwYzY3MjIyYz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODI2ODM3MTN9fX1dfQ__&Signature=F7t-z-DlKfP4K2rNQgNXXRxdx9gQF3xrplyS503x84xzE27WZRzRuWbLDWOjenDSUbCATWx1fe5swoH3UWzIqLAj8gNwa8xbWrFfk1V83UGvIegmgFHmROO1wwvzJ0jTqgMQqKNrJuY-59XiQQ5uOW2dIBq3FVxK92IH13tRpp88GOpMVlcZHVA3IbpZqOLcQ2XCeN6%7EWuFinGdSmS-wBjxVz0-F212YyEc9LsGCDVzTjg939fWqWDagp%7EBke84A5un5vDnH3%7Evea707-efRMMBhPHw2IoBQu8A%7ET%7EpXpMJL7cPkJ6qv25lK7RPWjr79andmxABt82zJ%7E89dNmH%7E0A__&Key-Pair-Id=KVTP0A1DKRTAX')
    canny_image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
    image = canny_image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # Initialize demo
    demo = Txt2ImgPipeline(
        scheduler=args.scheduler,
        denoising_steps=args.denoising_steps,
        output_dir=args.output_dir,
        version=args.version,
        hf_token=args.hf_token,
        verbose=args.verbose,
        nvtx_profile=args.nvtx_profile,
        max_batch_size=max_batch_size)

    # Load TensorRT engines and pytorch modules
    demo.loadEngines(args.engine_dir, args.onnx_dir, args.onnx_opset,
        opt_batch_size=len(prompt), opt_image_height=image_height, opt_image_width=image_width, \
        force_export=args.force_onnx_export, force_optimize=args.force_onnx_optimize, \
        force_build=args.force_engine_build, \
        static_batch=args.build_static_batch, static_shape=not args.build_dynamic_shape, \
        enable_refit=args.build_enable_refit, enable_preview=args.build_preview_features, enable_all_tactics=args.build_all_tactics, \
        timing_cache=args.timing_cache, onnx_refit_dir=args.onnx_refit_dir)
    demo.loadResources(image_height, image_width, batch_size, args.seed)

    print("[I] Warming up ..")
    # for _ in range(args.num_warmup_runs):
    #     images = demo.infer(canny_image, prompt, negative_prompt, image_height, image_width, warmup=True, verbose=False)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images = demo.infer(canny_image, prompt, negative_prompt, image_height, image_width, seed=args.seed, verbose=args.verbose)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    demo.teardown()
