# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=redefined-outer-name,missing-module-docstring,g-importing-member,missing-function-docstring,g-bare-generic
import os
import pickle
from typing import Dict
from flax import serialization
import jax
import jax.numpy as jnp
from eval_models.magiclens.model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
import numpy as np 
from PIL import Image
from functools import partial
import torch

def load_model(model_size: str, model_path: str) -> Dict:
    # init model
    model = MagicLens(model_size)
    rng = jax.random.PRNGKey(0)
    dummpy_input = {
        "ids": jnp.ones((1, 1, 77), dtype=jnp.int32),
        "image": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
    }
    params = model.init(rng, dummpy_input)
    print("model initialized")
    # load model
    with open(model_path, "rb") as f:
        model_bytes = pickle.load(f)
    params = serialization.from_bytes(params, model_bytes)
    print("model loaded")
    return model, params

def process_img(image_path: str, size: int) -> np.ndarray:
    """Process a single image to 224x224 and normalize."""
    img = Image.open(image_path).convert("RGB")
    ima = jnp.array(img)[jnp.newaxis, ...] # [1, 224, 224, 3]
    ima = ima / (ima.max() + 1e-12)  # avoid 0 division
    ima = jax.image.resize(ima, (1, size, size, 3), method='bilinear')
    return np.array(ima)

def process_query_example(processor, text, img):
    qtext = text
    ima = process_img(img, 224) if img is not None else None   
    qtokens = np.array(processor(qtext))
    return qtokens, ima

def magiclens_embed_function():
  
    model_size = "large"
    model_path = os.environ["MAGICLENS"]
    # init model
    tokenizer = clip_tokenizer.build_tokenizer()
    model, model_params = load_model(model_size, model_path)

    def embed(model, model_params, tokenizer,  item: str = "", dtype: str = "text", instruction=""):
        assert dtype in ["image", "text"]
        if dtype == "image":
            qtokens, qimages = process_query_example(tokenizer, instruction, item)
            jax_embeds = model.apply(model_params, {"ids": qtokens, "image": qimages})[
            "multimodal_embed_norm"
            ]
        else:
            qtokens, qimages = process_query_example(tokenizer, item, None)
            jax_embeds = model.apply(model_params, {"ids": qtokens})[
            "multimodal_embed_norm"
            ]



        np_arr = np.array(jax_embeds)
        tensor = torch.tensor(np_arr)
        return tensor
    return partial(embed, model, model_params, tokenizer)