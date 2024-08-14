from typing import Optional
import torch
from transformers import Trainer
from transformers.trainer import RandomSampler

class ContrastiveTrainer(Trainer):
    
       def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        query = inputs["query"]
        candidate = inputs["pos_cand"]

        query_outputs = model(**query)
        candidate_outputs = model(**candidate)

        loss = torch.tensor([1], dtype=torch.bfloat16)
        loss = loss.cuda()

        return loss
       
       def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
            return RandomSampler(self.train_dataset)