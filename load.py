import os
from contextlib import asynccontextmanager

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = os.getenv("MODEL_PATH", "models/ru-modernbert-sentiment")
MAX_LENGTH = 512
LABELS = ["Bad", "Neutral", "Good"]

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH)
    state["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).eval()
    yield


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    text: str


def score(text: str) -> list[float]:
    inputs = state["tokenizer"](text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.inference_mode():
        logits = state["model"](**inputs).logits
    return torch.softmax(logits, dim=-1)[0].tolist()


@app.post("/predict")
def predict(q: Query):
    probs = score(q.text)
    best = max(range(len(LABELS)), key=probs.__getitem__)
    return {
        "label": LABELS[best],
        "probs": {name: round(p, 4) for name, p in zip(LABELS, probs)},
    }


def classify(text: str):
    text = (text or "").strip()
    if not text:
        return None
    return dict(zip(LABELS, score(text)))


theme = gr.themes.Base(
    primary_hue="red",
    secondary_hue="red",
    neutral_hue="zinc",
).set(
    body_background_fill_dark="#100506",
    block_background_fill_dark="#1c0a0c",
    input_background_fill_dark="#260d10",
    border_color_primary_dark="#5c1a20",
    button_primary_background_fill="#b91c1c",
    button_primary_background_fill_hover="#dc2626",
    button_primary_text_color="white",
)

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=8, label="Текст отзыва", placeholder="Вставьте отзыв"),
    outputs=gr.Label(num_top_classes=3, label="Тональность"),
    title="Тональность отзыва",
    submit_btn="Оценить",
    clear_btn="Очистить",
    flagging_mode="never",
    theme=theme,
    js="() => { document.body.classList.add('dark'); }",
)

app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)