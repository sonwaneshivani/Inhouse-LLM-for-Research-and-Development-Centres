import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline", legacy=False)
model = model.to(device)

sample_article = '''In organic chemistry, a hydrocarbon is an organic compound consisting entirely of hydrogen and carbon.[1]: 620  Hydrocarbons are examples of group 14 hydrides. Hydrocarbons are generally colourless and hydrophobic; their odor is usually faint, and may be similar to that of gasoline or lighter fluid. They occur in a diverse range of molecular structures and phases: they can be gases (such as methane and propane), liquids (such as hexane and benzene), low melting solids (such as paraffin wax and naphthalene) or polymers (such as polyethylene and polystyrene).'''

text_to_summarize = "headline: " + sample_article

max_length = 256

encoding = tokenizer.encode_plus(text_to_summarize, return_tensors="pt")
input_ids = encoding["input_ids"].to(device)
attention_masks = encoding["attention_mask"].to(device)

beam_outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_masks,
    max_length=64,
    num_beams=3,
    early_stopping=True,
)

summarized_text = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
print(summarized_text)
