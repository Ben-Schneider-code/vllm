from transformers import CLIPProcessor, CLIPModel
from dataset.mscoco import MSCOCO, get_CLIP_collate_fn
from torch.utils.data import DataLoader
import torch
from utils import save
import sys

out_dir = sys.argv[1]

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
print(sum(p.numel() for p in model.parameters()))

# Move the model to GPU
model = model.to(device)

dataset = MSCOCO()
dl = DataLoader(dataset, num_workers=4, batch_size=128, collate_fn=get_CLIP_collate_fn(processor=processor))

total_correct = 0
total_samples = 0

dataset_info = {
    "model_name": model_name,
    "dataset_name": "mscoco"
}

c=[]
q=[]
meta=[]

model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    for batch, meta_for_batch in dl:
        # Move inputs to GPU
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # Get the indices of the highest probability for each image
        predicted_matches = torch.argmax(probs, dim=1)

        # Create a tensor representing the correct matches
        correct_matches = torch.arange(probs.shape[0], device=device)

        # Calculate the number of correct predictions for this batch
        correct_predictions = (predicted_matches == correct_matches).sum().item()
        meta.append(meta_for_batch)
        q.append(outputs.text_embeds.cpu())
        c.append(outputs.image_embeds.cpu())
        total_correct += correct_predictions
        total_samples += probs.shape[0]

q_tensor = torch.cat(q , dim=0)
c_tensor = torch.cat(c, dim=0)

save(dataset_info, meta, q_tensor, c_tensor, out_dir)

# Calculate overall accuracy
overall_accuracy = total_correct / total_samples

print(f"Overall Accuracy: {overall_accuracy:.4f}")