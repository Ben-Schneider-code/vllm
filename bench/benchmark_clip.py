from transformers import CLIPProcessor, CLIPModel
from datasets.mscoco import MSCOCO, get_CLIP_collate_fn, get_first_n
from torch.utils.data import DataLoader
import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
print(sum(p.numel() for p in model.parameters()))

# Move the model to GPU
model = model.to(device)

dataset = MSCOCO()

# Adjust this number based on how much of the dataset you want to use
num_samples = 20 * 128  # This will give us 20 batches of size 128
ds = get_first_n(num_samples)
dl = DataLoader(ds, num_workers=4, batch_size=128, collate_fn=get_CLIP_collate_fn(processor=processor))

total_correct = 0
total_samples = 0

model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    for batch in dl:
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

        total_correct += correct_predictions
        total_samples += probs.shape[0]

# Calculate overall accuracy
overall_accuracy = total_correct / total_samples

print(f"Overall Accuracy: {overall_accuracy:.4f}")