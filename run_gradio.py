import torch
import gradio as gr
from transformers import GenerationConfig

from train import get_model, tokenizer


model = get_model('tyuhgf/llm_calc')

def generate(
    instruction,
    temperature=0,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=1024,
):
    input_ids = tokenizer(
        instruction,
        truncation=True,
        max_length=10000,
        padding=False,
        return_tensors=None,
    )['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0).to('cuda')

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    yield tokenizer.decode(s)

gr.Interface(
    fn=generate,
    inputs=[
        gr.components.Textbox(
            lines=1,
            label="Arithmetic",
            placeholder="Add A=3496734 and B=3498634.",
        ),
        gr.components.Slider(
            minimum=0, maximum=1, value=0., label="Temperature"
        ),
        gr.components.Slider(
            minimum=0, maximum=1, value=0.75, label="Top p"
        ),
        gr.components.Slider(
            minimum=0, maximum=100, step=1, value=40, label="Top k"
        ),
        gr.components.Slider(
            minimum=1, maximum=4, step=1, value=4, label="Beams"
        ),
        gr.components.Slider(
            minimum=1, maximum=1024, step=1, value=512, label="Max tokens"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="llm_calc",
    description="LLaMA for long integers addition",
).queue().launch(server_name="0.0.0.0")
