import torch
import torch.nn.functional as F
import torch.distributed as dist

def compute_gathered_loss(q_emb, c_emb):
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

    outputs_for_logging = {
        "global_accuracy": acc_global.detach(),
        "global_loss": loss_global.detach(),
        "local_accuracy": acc_local/world_size,
        "local_loss": loss_local/world_size
    }

    return loss_global, outputs_for_logging

def get_mean_token_embed(input_ids, hidden_state, padding_token_id):
     mask = (input_ids != padding_token_id).unsqueeze(-1)
     masked_states = mask*hidden_state
     mean_token_emb = torch.mean(masked_states,dim=1) # Average
     return mean_token_emb

def compute_contrastive_loss(q_embeds, p_embeds, temperature=1.0, label_smoothing=0.0):

    bs = q_embeds.size(0)

    score = torch.matmul(q_embeds, p_embeds.t()) / temperature
    sim_targets = torch.arange(bs).to(score.device)  # [bs]

    # compute loss
    loss = F.cross_entropy(score, sim_targets, label_smoothing=label_smoothing)
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