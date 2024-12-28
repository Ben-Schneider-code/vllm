# ------------
# Evals against mscoco, flicker30k, and imagenet variants
#
# ------------

import os
import sys
from tqdm import tqdm
import json
from eval.embed_function import get_model_with_embed_function

supported_models = ["abcQwenVL"]

def load(model_type: str, model_path: str):
    if model_type == "abcQwenVL":
        return get_model_with_embed_function(model_type, model_path)

def eval_mscoco(fxn):
    
    mscoco_eval_path = os.environ.get("MSCOCO_EVAL", None)
    assert(mscoco_eval_path is not None)
    with open(mscoco_eval_path, "rb") as f:
        mscoco_json = json.loads(f.read())["images"]

    test = list(filter(lambda x : x["split"] == "test", mscoco_json))
    assert len(test == 5000)

    images = [os.path.join( os.path.dirname(mscoco_eval_path),x["filepath"],x["filename"])  for x in test]
    text = []

    for x in test:
        text.extend(x["raw"])

    assert len(text) == 25000

    for ind, item in tqdm(enumerate(images)):
        emb = fxn(item, dtype="image")
        images[ind] = (item, emb)

    for ind, item in tqdm(enumerate(text)):
        emb = fxn(item, dtype="text")
        text[ind] = (item, emb)

    


def main(model_type: str, model_path: str):

    embed_fxn = lambda x : x #load(model_type, model_path)

    eval_mscoco(embed_fxn)




if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
