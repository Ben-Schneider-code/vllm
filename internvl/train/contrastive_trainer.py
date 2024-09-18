from typing import List, Optional, Tuple, Dict, Union, Any
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import has_length, EvalLoopOutput, EvalPrediction
from transformers.trainer import RandomSampler, logger
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.distributed as dist
from transformers.integrations import WandbCallback, deepspeed_init
import os

class WandbLogger(WandbCallback):

    def __init__(self):
        super().__init__()
        self.additional_metrics = {}

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        if state.is_world_process_zero:
            self._wandb.config.update({"pid": str(os.getpid())}, allow_val_change=True)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model=model, logs=logs | self.additional_metrics, **kwargs)
        self.additional_metrics.clear()

class ContrastiveTrainer(Trainer):

     def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.wandb_callback = WandbLogger()
          self.add_callback(self.wandb_callback)

     def log_to_wandb(self, key, value):
          self.wandb_callback.additional_metrics[key] = value

     def last_token_loss(self, model, inputs, return_outputs=False):

          query = inputs["query"]
          candidate = inputs["pos_cand"]

          # torch.cuda.memory._record_memory_history(max_entries=40)

          # MEMORY OPTIMIZATIONS
          # --------------------
          # make sure neither of these have any "labels" key
          # skips the loss computation
          query.pop("labels")
          candidate.pop("labels")
          # ensure that sequences are **not** padded to context length
          query_outputs : CausalLMOutputWithPast = model(**query, output_hidden_states=True)
          candidate_outputs : CausalLMOutputWithPast = model(**candidate, output_hidden_states=True)
          
          # ensure logits computation was skipped for memory / speed
          # saves memory, requires changing a line the transformers lib implementation of qwen
          # (or other LLM in used)
          assert(query_outputs.logits is None)
          assert(candidate_outputs.logits is None)

          q_eos_token_emb = get_last_token_embed(query["input_ids"], query_outputs.hidden_states[-1], 0)
          c_eos_token_emb= get_last_token_embed(candidate["input_ids"], candidate_outputs.hidden_states[-1], 0)

          loss, acc = self.gathered_loss(q_eos_token_emb,c_eos_token_emb, return_outputs) if self.args.gather_loss \
          else self.local_loss(q_eos_token_emb,c_eos_token_emb,return_outputs)

          return (loss, {"accuracy": acc}) if return_outputs else loss
     
     def mean_token_loss(self, model, inputs, return_outputs=False):

          query = inputs["query"]
          candidate = inputs["pos_cand"]
          query.pop("labels")
          candidate.pop("labels")

          query_outputs : CausalLMOutputWithPast = model(**query, output_hidden_states=True)
          candidate_outputs : CausalLMOutputWithPast = model(**candidate, output_hidden_states=True)
          
          # ensure logits computation was skipped for memory / speed
          # (or other LLM in used)
          assert(query_outputs.logits is None)
          assert(candidate_outputs.logits is None)

          q_eos_token_emb = get_mean_token_embed(query["input_ids"], query_outputs.hidden_states[-1], 0)
          c_eos_token_emb= get_mean_token_embed(candidate["input_ids"], candidate_outputs.hidden_states[-1], 0)

          loss, acc = self.gathered_loss(q_eos_token_emb,c_eos_token_emb, return_outputs) if self.args.gather_loss \
          else self.local_loss(q_eos_token_emb,c_eos_token_emb,return_outputs)

          return (loss, {"accuracy": acc}) if return_outputs else loss

     def prediction_step(
          self,
          model: torch.nn.Module,
          inputs: Dict[str, Union[torch.Tensor, Any]],
          prediction_loss_only: bool,
          ignore_keys: Optional[List[str]] = None,
     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

          inputs = self._prepare_inputs(inputs)

          with torch.no_grad():
               with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
               loss = loss.mean().detach()

          return loss, outputs

     def evaluation_loop(
          self,
          dataloader: torch.utils.data.DataLoader,
          description: str,
          prediction_loss_only: Optional[bool] = None,
          ignore_keys: Optional[List[str]] = None,
          metric_key_prefix: str = "eval",
     ) -> EvalLoopOutput:
          """
          Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

          Works both with or without labels.
          """
          args = self.args

          prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

          # if eval is called w/o train, handle model prep here
          if self.is_deepspeed_enabled and self.deepspeed is None:
               _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

          model = self._wrap_model(self.model, training=False, dataloader=dataloader)

          if len(self.accelerator._models) == 0 and model is self.model:
               raise Exception("NotImplementedError")

          # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
          # while ``train`` is running, cast it to the right dtype first and then put on device
          if not self.is_in_train:
               if args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=args.device)
               elif args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=args.device)

          batch_size = self.args.eval_batch_size

          logger.info(f"\n***** Running {description} *****")
          if has_length(dataloader):
               logger.info(f"  Num examples = {self.num_examples(dataloader)}")
          else:
               logger.info("  Num examples: Unknown")
          logger.info(f"  Batch size = {batch_size}")

          model.eval()

          self.callback_handler.eval_dataloader = dataloader

          if args.past_index >= 0:
               self._past = None

          # Will be useful when we have an iterable dataset so don't know its length.
          observed_num_examples = 0
          outputs = []
          losses = []

          # Main evaluation loop
          for _, inputs in enumerate(dataloader):
               # Update the observed num examples
               observed_batch_size = find_batch_size(inputs)
               if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                         batch_size = observed_batch_size

               # Prediction step
               loss, output = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
               
               outputs.append(output)
               losses.append(loss)
               
               self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)


          # Number of samples
          num_samples = observed_num_examples

          metrics = {}
     
     # Iterate through each dictionary in the list
          for d in outputs:
               for key, value in d.items():
                    metrics[key] = metrics.get(key, 0) + value
     
          metrics = {key: metrics[key] / len(outputs) for key in metrics}
          metrics["loss"] = torch.mean(torch.stack(losses, dim=0),dim=0)
          metrics=cast_loss_dict(metrics, metric_key_prefix)

          return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)

     def compute_loss(self, model, inputs, return_outputs=False):
          if self.args.loss_type == "last_token":
               return self.last_token_loss(model, inputs, return_outputs=return_outputs)
          if self.args.loss_type == "mean_token":
               return self.mean_token_loss(model, inputs, return_outputs=return_outputs)
          else:
               raise Exception("Loss type not implemented")
          
     def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
          return RandomSampler(self.train_dataset)
     
     def gathered_loss(self, q_emb, c_emb, return_outputs):
          """
          Compute the loss by gathering across GPUs.
          """

          q_emb = q_emb.float()
          c_emb = c_emb.float()

           # Get the number of GPUs (world_size)
          world_size = dist.get_world_size()
          rank = dist.get_rank()

          # Gather q_embed and c_embed from all GPUs
          q_global = [torch.zeros_like(q_emb) for _ in range(world_size)]
          c_global = [torch.zeros_like(c_emb) for _ in range(world_size)]

          dist.all_gather(q_global, q_emb)
          dist.all_gather(c_global, c_emb)
          
          q_global[rank] = q_emb
          c_global[rank] = c_emb

          # Concatenate the gathered embeddings along the batch dimension
          q_global = torch.cat(q_global, dim=0)
          c_global = torch.cat(c_global, dim=0)

          loss_global, acc_global = compute_contrastive_loss(q_global, c_global)
          loss_local, acc_local = compute_contrastive_loss(q_emb.detach(), c_emb.detach())
          
          dist.all_reduce(acc_local,op=dist.ReduceOp.SUM)
          dist.all_reduce(loss_local,op=dist.ReduceOp.SUM)

          # log only on main process
          if not return_outputs and dist.get_rank() == 0:
               self.log_to_wandb("global_accuracy", acc_global.detach())
               self.log_to_wandb("global_loss", loss_global.detach())
               self.log_to_wandb("local_accuracy", acc_local/world_size)
               self.log_to_wandb("local_loss", loss_local/world_size)

          return loss_global, acc_global
     
     def local_loss(self, q_emb, c_emb, return_outputs):
          """
          Compute the loss locally on each GPU, average later.
          """
          q_emb = q_emb.float()
          c_emb = c_emb.float()

          local_loss, local_acc = compute_contrastive_loss(q_emb, c_emb)


          # reduce for logging
          log_acc = local_acc.detach().clone()
          log_loss = local_loss.detach().clone()
          world_size = dist.get_world_size()
          dist.all_reduce(log_acc.detach(),op=dist.ReduceOp.SUM)
          dist.all_reduce(log_loss.detach(),op=dist.ReduceOp.SUM)
          
          # log only on main process
          if not return_outputs and dist.get_rank() == 0:
               self.log_to_wandb("local_accuracy", log_acc/world_size)
               self.log_to_wandb("local_loss", log_loss/world_size)

          return local_loss, local_acc

def compute_contrastive_loss(q_embeds, p_embeds):  # [batch_size, embed_dim]
    # Normalized features
    q_embeds = F.normalize(q_embeds, dim=-1)
    p_embeds = F.normalize(p_embeds, dim=-1)
    bs = q_embeds.size(0)

    score = torch.matmul(q_embeds, p_embeds.t())  # * self.logit_scale  # [bs, bs]
    sim_targets = torch.arange(bs).to(score.device)  # [bs]

    # compute loss
    loss = F.cross_entropy(score, sim_targets)
    _max_score, max_idxs = torch.max(score, 1)

    accuracy = (max_idxs == sim_targets).sum() / bs

    return loss, accuracy

def get_last_token_embed(input_ids, hidden_state, padding_token_id):
    # Find the position of the last non-padding token for each sequence
    mask = input_ids != padding_token_id  # Create a mask where padding tokens are False
    last_token_pos = mask.sum(dim=1) - 1  # Get the index of the last non-padding token

    # Create a range tensor for batch indexing
    batch_size = input_ids.size(0)
    batch_range = torch.arange(batch_size, device=input_ids.device)

    # Extract the last token embedding for each sequence
    last_token_embeds = hidden_state[batch_range, last_token_pos]

    return last_token_embeds

def get_mean_token_embed(input_ids, hidden_state, padding_token_id):
     mask = (input_ids != padding_token_id).unsqueeze(-1)
     masked_states = mask*hidden_state
     mean_token_emb = torch.mean(masked_states,dim=1) # Average over seq_len
     return mean_token_emb

def cast_loss_dict(d: Dict, dataset_name: str):
     return {dataset_name+"_"+x:y.cpu().item() for (x,y) in d.items()}

