## LLM calc
Finetune OpenLLaMA for long integers arithmetics

## Evaluate
```python
from train import get_model, tokenizer
from evaluate import evaluate

model = get_model('tyuhgf/llm_calc')
evaluate(model, tokenizer)
```