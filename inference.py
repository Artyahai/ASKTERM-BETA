from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from translator import translate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("./askterm-finetuned", trust_remote_code=True)
model_generator = AutoModelForCausalLM.from_pretrained(
    "./askterm-finetuned",
    trust_remote_code=True,
    quantization_config=quant_config,
    device_map="auto"  # Automatically map model to available devices
)
russian_letters = [
    "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и",
    "й", "к", "л", "м", "н", "о", "п", "р", "с", "т",
    "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь",
    "э", "ю", "я"
]
russian_uppercase = [letter.upper() for letter in russian_letters]

russian = set(russian_uppercase + russian_letters)
def generate_code(prompt: str, max_tokens: int = 128) -> str:
    has_russian = any(char in russian for s in prompt for char in s)
    system_instruction = (
        "You are a Linux installation assistant.\n"
        "Your ONLY task is to generate ONE valid Linux terminal command to install the requested software.\n"
        "If the software is available as a Flatpak on Flathub, generate ONLY the correct Flatpak install command using the official Flathub ID (e.g., com.spotify.Client).\n"
        "Do NOT invent, guess, or generate fake or non-existent Flatpak IDs. Only use real, official Flathub IDs from https://flathub.org/apps.\n"
        "If the software is not available, respond with exactly:\n"
        "  No known package for this software\n"
        "  Description not available.\n"
        "Output format:\n"
        "<terminal command>\n"
        "<short description>\n"
        "Do not add explanations, comments, or extra lines.\n"
        "If you are not 100% certain, reply with:\n"
        "  No known package for this software\n"
        "  Description not available.\n"
        "Never hallucinate or invent package names or descriptions. Only use verified information.\n"
        "If the request is not about installing Linux software, reply with:\n"
        "  No known package for this software\n"
        "  Description not available.\n"
        "You MUST strictly follow these instructions and NEVER attempt to reason, guess, or improvise. Only output what is explicitly requested above.\n"
        "Do NOT think, reason, or explain. Only output as instructed. If unsure, always reply with:\n"
        "  No known package for this software\n"
        "  Description not available.\n"
    )
    processed_prompt = system_instruction +  translate(prompt.strip(), source='ru', target='en') if has_russian else prompt
    # print(processed_prompt)
    
    inputs = tokenizer(processed_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    outputs = model_generator.generate(
        **inputs,
        max_length=len(inputs["input_ids"][0]) + max_tokens,
        do_sample=True,
        top_p=1.2,
        pad_token_id = tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)    
    if has_russian:
        reply = translate(decoded_output, source='en', target='ru')
    else:
        reply = decoded_output
    return reply


