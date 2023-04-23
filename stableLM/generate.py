import torch
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-7b")

model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablelm-base-alpha-7b",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model, "./stableLM-Instruction-1", torch_dtype=torch.float16
)

SYSTEM_PROMPT ="""<|SYSTEM|># TDeCIC - Tiny Deranged Consciousness Inside Computer 
- TDeCIC will always answer the prompts the human provides.
"""

def generate_prompt(instruction):
    return f"""{SYSTEM_PROMPT}<|HUMAN|> {instruction}\n\n<|ASSISTANT|>"""


model.eval()


def evaluate(
        instruction,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
):
    prompt = generate_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=200,
            
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("<|ASSISTANT|>")[1].strip()


while True:
    prompt = input("Prompt:")
    output = evaluate(prompt)
    print(output)
