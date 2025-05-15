from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

MODEL_NAME = "zlucia/legal-bert-base-uncased"

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

class QuestionInput(BaseModel):
    inputs: str

@app.post("/inference")
async def inference(data: QuestionInput):
    question = data.inputs
    context = (
        "Under the law, an agreement between two parties constitutes a binding contract "
        "when certain legal conditions are met. These include offer, acceptance, consideration, and intent."
    )

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    answer_start_scores, answer_end_scores = model(**inputs).values()

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    return {"answer": answer}
