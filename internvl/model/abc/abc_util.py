from typing import Dict
import torch
import torch.nn.functional as F

def get_mean_token_embed(input_ids, hidden_state, padding_token_id):
     mask = (input_ids != padding_token_id).unsqueeze(-1)
     masked_states = mask*hidden_state
     mean_token_emb = torch.mean(masked_states,dim=1) # Average
     return mean_token_emb

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

def compute_loss(q_emb, c_emb):
    """
    Compute the loss locally on each GPU, average later.
    """
    q_emb = q_emb.float()
    c_emb = c_emb.float()

    local_loss, local_acc = compute_contrastive_loss(q_emb, c_emb)
    
    return local_loss, local_acc