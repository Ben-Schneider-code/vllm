## Building a Decorder-only VLM Retriever
### Experiments
- Using CC dataset and LORA with params from NVidia paper run the following grid.  

| Mask        | Pooling    |
|-------------|------------|
| Full Attn   | EOS Token  |
| Casual Attn | Mean Token |

- Try the NVidia version of an adapter (special version of attn where K=V + dense MLP)
### To Do  
- **Build sanity checker for model**, i.e. making sure it cananswer basic questions with our tokenization + instruction.
- **Add a basic dataset**
    - Conceptual captions is already downloaded.
    - Across the board, it doesn't seem like large data is required, Nvidia paper and LLM2VEC only require a few thousand steps (batch_size=128).  
- **Build a version of InternVL that follows NVidia's decoder-only appraoch.**
    - *Ablate* to show which techniques add value:
    - Remove casual attention [use all 1s for attention mask].
    - Using last token embed, average token embed, Attention into dense on hidden states.
    - Use LORA on the base model.
        - Nvida paper provides settings.
    - Instruction tuning **(?)** - The NVidia paper uses it on queries.
    - Compare vs. CLIP on MSCOCO.
- **Long-run potentiall improvements**
    - Better data, NVidia paper provides hard-negatives dereived from an encoder model. We could use **CLIP** in a similar way?
        - [NV-Retriever](https://arxiv.org/pdf/2407.15831) provides insight on hard negative mining.

## Questions:
- What is the ideal way to pass instructions to InternVL, can we make it similar to NV-embed?

### Resources:  
[NV-Embed](https://arxiv.org/abs/2405.17428)  
[LLM2VEC](https://arxiv.org/abs/2404.05961)  
[NV-Retriever](https://arxiv.org/pdf/2407.15831) - Haven't read yet. 
