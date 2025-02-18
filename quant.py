import torch
from torch.utils.data import Dataset
import transformers
from transformers import BertForSequenceClassification, AdamW

# load the pretrained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# # Load the model from a specific epoch
checkpoint_path = "/home/deepali/codes/sentiment_analysis/Data/weights/model_epoch_1.pt"
# Load the saved state_dict
state_dict = torch.load(checkpoint_path, map_location=device)
# Remove unexpected keys (if any)
state_dict.pop("bert.embeddings.position_ids", None)
# Load the updated state_dict into the model
model.load_state_dict(state_dict, strict=False)
model.eval()
print("Model loaded successfully with adjusted state_dict.")