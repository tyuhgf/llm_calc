## LLM calc
Finetune OpenLLaMA for long integers arithmetics

## Evaluate
```python
from train import get_model, tokenizer
from evaluate import evaluate

model = get_model('tyuhgf/llm_calc')
evaluate(model, tokenizer)
```
Also:
```
python run_gradio.py
```

## Save dataset
```python
import json
from dataset import AdditionDataset, tokenizer

ad = AdditionDataset(tokenizer, size=10000)
with open('llama_calc_dataset.json', 'w') as f:
   json.dump([{'text': text} for text in ad.texts], f)
```

## Train
```python
from train import get_model, prepare_trainer, prepare_data, tokenizer

model = get_model()
train_data, val_data = prepare_data()
trainer = prepare_trainer(model, tokenizer, train_data, val_data)
trainer.train()
model.save_pretrained('your_path')
```