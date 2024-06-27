# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union
from tqdm import tqdm
import torch
from utils.config.Configuration import default


class DDIMPipeline:

    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        y_cond,
        y_t,
        mask,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        start_time_step_inx=0,
        **args
    ):

        image = y_t
        txt_cond = default(args, "txt_cond", None)
        class_label = default(args, "class_label", None)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        bs = y_t.shape[0]

        feat_chn = image.shape[1]
        mask_adaptive = mask.repeat([1, feat_chn] + (len(mask.shape) - 2) * [1])

        for t in tqdm(
            self.scheduler.timesteps[start_time_step_inx:],
            desc="sampling loop time step",
            total=num_inference_steps - start_time_step_inx,
        ):
            # for t in self.scheduler.timesteps:
            # 1. predict noise model_output
            if y_cond is not None:
                model_input = torch.cat([y_cond, image], dim=1)  # mask done in unet...
            else:
                model_input = image
            alphas_cumprod = self.scheduler.alphas_cumprod.to(image.device)
            timestep_encoding = alphas_cumprod[t].repeat(bs, 1).to(y_t.device)

            model_output = self.unet(
                model_input,
                timestep_encoding,
                mask=mask,
                context=txt_cond,
                y=class_label,
            )  # .sample

            image = self.scheduler.step(
                model_output,
                t,
                image,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
            ).prev_sample

            image[~mask_adaptive] = 0.0
        return image
