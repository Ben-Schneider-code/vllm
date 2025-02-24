base_model = "Qwen/Qwen2-VL-7B-Instruct"
pretrain_adapter = "/home/b3schnei/output/QwenVL-8B-Large-Batch/checkpoint-2000"
inst_adapter = "/home/b3schnei/output/QwenVL-8B-Instruct-Large/checkpoint-100"

from functional.embed_function import get_embed_function
new_fxn = get_embed_function(base_model, pretrain_adapter, inst_adapter)

# compare text unsupervised embeddings
inp_img = "./examples/cat.jpg"
inp_text = "what kind of animal is this?"

new_emb = new_fxn(image=inp_img, text=inp_text)
