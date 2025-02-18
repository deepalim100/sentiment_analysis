from fastapi import FastAPI
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pydantic import BaseModel

app = FastAPI()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the pretrained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
state_dict = torch.load('/app/model_epoch_1.pt', map_location=device)
# Remove unexpected keys (if any)
state_dict.pop("bert.embeddings.position_ids", None)
# Load the updated state_dict into the model
model.load_state_dict(state_dict, strict=False)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TestInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TestInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}
