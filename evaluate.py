import torch

from transformers import GenerationConfig
from dataset import generate_pairs, generate_input, response_to_answer


def evaluate(model, tokenizer, length_distribution=tuple([0] * 8 + [1]), seed=4321):
    numbers = generate_pairs(size=10, seed=seed, lengths_distribution=length_distribution)
    for i in range(10):
        a, b = numbers[:, i]
        text = generate_input(a, b, input_only=True)
        input_ids = tokenizer(
            text,
            truncation=True,
            max_length=10000,
            padding=False,
            return_tensors=None,
        )['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0).to('cuda')

        generation_config = GenerationConfig(
            temperature=0,
            top_p=0.75,
            top_k=40,
            num_beams=4,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)
        print(a + b)
        parsed = response_to_answer(output)
        print(parsed)
        if 'result' in parsed:
            print(f'Equal: {parsed["result"] == a + b}')
