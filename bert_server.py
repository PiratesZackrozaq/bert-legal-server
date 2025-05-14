from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = Flask(__name__)

# Load BERT Legal model and tokenizer
MODEL_NAME = "nlpaueb/bert-legal-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# Dummy legal context (you can enhance this with a database or document store)
CONTEXT = """
In California, the statute of limitations for a breach of contract is generally 4 years for written contracts and 2 years for oral contracts.
"""

@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json()
    question = data.get("inputs")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    inputs = tokenizer.encode_plus(question, CONTEXT, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return jsonify({"answer": answer.strip()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
