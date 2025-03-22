from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/home/est_licenciatura_dan.muniz/proyectos/DeepSeek-R1-Distill-Llama-8B"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False
)

# Asignar el pad_token_id al eos_token_id (la mayoría de modelos LLaMA-like no tienen pad token definido).
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to("cuda")

def generate_response(
    prompt: str,
    model,
    tokenizer,
    temperature: float = 0.6,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.2
):
   
    # Append "<think>\n" at the end of the prompt to encourage thorough reasoning
    input_text = prompt + "\n<think>\n"
    
    # Incluir la attention_mask al tokenizar
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    
    # Generate tokens (pasando la attention_mask y el pad_token_id)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id  # For LLaMA-like models
    )
    
    # Decode only the newly generated tokens
    generated_text = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    user_prompt = (
        "Porfavor crea un archivo XML con *30* frases que puede usar paciente de hospital para comunicar sus necesidades"
        "separa cada frase con <frase></frase> en el archivo"
    )
    
    response = generate_response(
        prompt=user_prompt,
        model=model,
        tokenizer=tokenizer,
        temperature=0.6,          # recomendado en [0.5, 0.7]
        max_new_tokens=4096,
        repetition_penalty=1.2
    )
    
    print("Model Output:")
    print(response)
