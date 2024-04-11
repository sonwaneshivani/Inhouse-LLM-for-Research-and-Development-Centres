from flask import Flask, render_template, request
import subprocess
import os
from PIL import Image
from transformers import T5ForConditionalGeneration, T5Tokenizer, DistilBertTokenizer, DistilBertForQuestionAnswering, BartForConditionalGeneration, BartTokenizer
import torch

app = Flask(__name__)

# Loading pre-trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using T5 model for headline generation
headline_model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
headline_tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline", legacy=False)
headline_model = headline_model.to(device)

# Question & Answering model
qna_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
qna_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
qna_model.to(device)

# Generating summary model
summary_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summary_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summary_model = summary_model.to(device)

# Html rendering
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    text_input = request.form.get('text')
    file_input = request.files.get('file')
    questions_input = request.form.get('questionsInput')  # Input is taken

    # Saving the uploaded file
    if file_input:
        file_path = 'uploads/' + file_input.filename
        file_input.save(file_path)
    else:
        file_path = None

    # Choosing the appropriate Python script based on the action
    action = request.form.get('action')
    if action == 'headline':
        result = generate_headline(text_input)
    elif action == 'summarize':
        result = generate_summary(text_input)
    elif action == 'generateQuestions':
        result = generate_auto_questions(text_input)
    else:
        result = 'Invalid action'

    return result

def generate_auto_questions(text):
    # We can implement our logic here to generate questions automatically
    # For now, let's generate a dummy list of questions
    dummy_questions = ["What are the key features or attributes of the data?","Can you provide a brief overview of the data distribution?","How frequently is the data updated?","What are the potential challenges or limitations associated with this data?","Is there any sensitive information in the dataset that needs special handling?","Who are the primary stakeholders or users of this data?","How frequently is the data updated?","What is the purpose of the data?", "How was the data collected? ","What is the context?", "What is this about?", "Wanna know more?"]
    return '\n'.join(dummy_questions)

def generate_headline(text):
    max_len = 200

    encoding = headline_tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    beam_outputs = headline_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=64,
        num_beams=3,
        early_stopping=True,
    )

    result = headline_tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
    return result


def process_text_with_questions(text, questions_input):
    questions = [q.strip() for q in questions_input.split('\n') if q.strip()]

    text = text
    answers = []
    for question in questions:
        answer = answer_question(question, text)
        answers.append({'question': question, 'answer': answer})

    return answers

def generate_qna(questions_input, text):
    answers = process_text_with_questions(text, questions_input)
    result = ''
    for answer in answers:
        result += f"Question: {answer['question']}<br>Answer: {answer['answer']}<br><br>"

    return result

def answer_question(question, text):
    question_tokens = qna_tokenizer.tokenize(question)
    text_tokens = qna_tokenizer.tokenize(text)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + text_tokens + ['[SEP]']
    input_ids = qna_tokenizer.convert_tokens_to_ids(tokens)

    inputs = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = qna_model(inputs)

    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits)

    predict_answer_tokens = input_ids[answer_start_index: answer_end_index + 1]
    predict_answer = qna_tokenizer.decode(predict_answer_tokens)

    return predict_answer

def generate_summary(input_text):
    inputs = summary_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  
    summary_ids = summary_model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    return summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
