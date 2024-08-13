from transformers import Trainer

class ContrastiveTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        mini_batch_size = 4

        

        return loss