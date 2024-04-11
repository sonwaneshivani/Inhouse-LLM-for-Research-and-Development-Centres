import torch
from transformers import BartForConditionalGeneration, BartTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loading pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = model.to(device)

# New input text
new_input_text = '''In organic chemistry, a hydrocarbon is an organic compound consisting entirely of hydrogen and carbon.[1]: 620  Hydrocarbons are examples of group 14 hydrides. Hydrocarbons are generally colourless and hydrophobic; their odor is usually faint, and may be similar to that of gasoline or lighter fluid. They occur in a diverse range of molecular structures and phases: they can be gases (such as methane and propane), liquids (such as hexane and benzene), low melting solids (such as paraffin wax and naphthalene) or polymers (such as polyethylene and polystyrene).'''

# Tokenization and generating summary
inputs = tokenizer(new_input_text, return_tensors="pt", max_length=1024, truncation=True)
inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the same device
summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decoding and printing the generated summary
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Summary:", generated_summary)
