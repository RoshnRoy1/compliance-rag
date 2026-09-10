import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load triplets
with open("finetune/triplets.json") as f:
    triplets = json.load(f)

print(f"Training on {len(triplets)} triplets")

# Convert to InputExample format
train_examples = []
for t in triplets:
    train_examples.append(InputExample(
        texts=[t["question"], t["positive"], t["negative"]]
    ))

# Load the base model (same one you've been using)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create dataloader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# TripletLoss: pushes question closer to positive, further from negative
train_loss = losses.TripletLoss(model=model)

# Train
output_path = "finetune/compliance-MiniLM"
print("Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=10,
    output_path=output_path,
    show_progress_bar=True,
)

print(f"\nFine-tuned model saved to {output_path}/")