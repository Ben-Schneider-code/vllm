
from eval_models.UniIR.src.models.uniir_clip.clip_scorefusion.clip_sf import CLIPScoreFusion
from functools import partial
import os
from PIL import Image
import torch

def uniir_embed_function():
    path = os.environ["UNIIR"]
    model = CLIPScoreFusion(download_root=path)
    model.cuda()
    img_proc = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()
    
    def embed(model, img_proc, tokenizer,  item: str = "", dtype: str = "text", instruction=""):
        assert dtype in ["image", "text"]
        with torch.no_grad():
            if dtype == "image":
                image = Image.open(item)
                img_out = img_proc(image).cuda()
                txt_out = tokenizer(instruction).cuda()

                img_out = torch.unsqueeze(img_out, dim=0)
                out = model.encode_multimodal_input(txt_out, img_out)

            else:
                txt_out = tokenizer(item).cuda()
                out = model.encode_text(txt_out)
            return out.detach().cpu()

    return partial(embed, model, img_proc, tokenizer)