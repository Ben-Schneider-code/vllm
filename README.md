## Building a Decorder-only VLM Retriever

### Hypothesis
- Overfitting issue: it takes a large amount of out-of-batch example to ensure good loss in contrastive learning.
- With low batch sizes the model finds cheap optimizations, overfits and quits.
- Solution: Larger batch sizes / better out of batch retrieval candidates.  
    - create high quality negatives using another model (CLIP, etc.)

### Experiments
- Using CC dataset and LORA with params from NVidia paper run the following grid.  
- **Note** Use LORA, params are given in the NV-embed paper. 
- **Note** llm2vec uses a mask on the embedding over pad tokens (for pooling). But applies full attn (even over padded tokens) using flash attn.  
- **Note** maybe change how the tokenization works. I.e. use a different system message, etc.
- **Note** ablate across gathering before doing loss computation.
- **Note** Check correctness of bidirectional attn.
- **Note** Find best combination of techniques @ 8B param scale.    
- Try the NVidia version of an adapter (special version of attn where K=V + dense MLP)
- **RESULTS** all techniques have very similar error, bidirectional *slightly* better than causal. Basically no difference between mean and last token. Unintuitively, averaging negatives across GPUs doesn't help (?). Implementation error?  
- **PROBLEM** incredible overfitting. As little as 100 steps can essentially halt progress.  
    - Try mined hard negatives (using CLIP, following Nvidia retriever paper)
    - No warmup (?), lower lr (?), Cosine lr (?)
### To Do  
- **Visualize**
    - Visualize batches of mscoco.
- **Check how the instructions/settings changes things**
    - No instruction vs. current vs. different version of current.
    - Mess with system message (?)  
    - No warmup (?), lower lr (?), Cosine lr (?)
    - Maybe decaying lr leads to overfitting
- **Add another dataset like scienceQA for eval**
- **Build a version of InternVL that follows NVidia's decoder-only appraoch.**
    - *Ablate* to show which techniques add value:
    - Remove casual attention [use all 1s for attention mask].
    - Using last token embed, average token embed, Attention into dense on hidden states.
    - Use LORA on the base model.
        - Nvidia paper provides settings.
    - Instruction tuning **(?)** - The NVidia paper uses it on queries.
    - Compare vs. CLIP on MSCOCO.
- **Long-run potential improvements**
    - Better data, NVidia paper provides hard-negatives dereived from an encoder model. We could use **CLIP** in a similar way?
        - [NV-Retriever](https://arxiv.org/pdf/2407.15831) provides insight on hard negative mining.

### Done
- **Build sanity checker for model** - Done
    - i.e. making sure it cananswer basic questions with our tokenization + instruction.
- **Add a basic dataset** - Done
    - Conceptual captions is already downloaded. 
    - Across the board, it doesn't seem like large data is required, Nvidia paper and LLM2VEC only require a few thousand steps (batch_size=128).

## Questions:
- What is the ideal way to pass instructions to InternVL, can we make it similar to NV-embed?
- Should I be using the **dynamic_image_size** flag while training my models to compensate for different sized images in dataset?
### Resources:  
[NV-Embed](https://arxiv.org/abs/2405.17428)  
[LLM2VEC](https://arxiv.org/abs/2404.05961)  
[NV-Retriever](https://arxiv.org/pdf/2407.15831) - Haven't read yet.  
[ALIGNModel](https://arxiv.org/abs/2102.05918) - Haven't read yet.
