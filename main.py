import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Замени на свой репозиторий после push_to_hub, например: "ivanov/toxic-comment-distilbert"
MODEL_NAME = "ЗАМЕНИ_НА_СВОЙ_USERNAME/toxic-comment-distilbert"


@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()

st.set_page_config(page_title="Toxic Comment Detector", page_icon="🛡️")

st.title("🛡️ Toxic Comment Detector")
st.write(
    "Модель на основе DistilBERT, дообученная (fine-tuned) на датасете "
    "Jigsaw Toxic Comment Classification для определения токсичности комментариев."
)

text_input = st.text_area("Введите текст комментария:", height=120, placeholder="Type or paste a comment here...")

threshold = st.slider(
    "Порог классификации (threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="Ниже порога — модель более чувствительна (выше recall). Выше порога — более консервативна (выше precision)."
)

if st.button("Проверить", type="primary"):
    if text_input.strip():
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        toxic_prob = probs[0][1].item()

        st.metric("Вероятность токсичности", f"{toxic_prob:.1%}")
        st.progress(toxic_prob)

        if toxic_prob >= threshold:
            st.error("⚠️ Комментарий классифицирован как токсичный")
        else:
            st.success("✅ Комментарий не классифицирован как токсичный")
    else:
        st.warning("Введите текст для проверки")

with st.expander("О проекте"):
    st.write(
        """
        **Датасет:** Jigsaw Toxic Comment Classification Challenge (Kaggle)

        **Baseline:** TF-IDF + Logistic Regression (class_weight='balanced')

        **Финальная модель:** DistilBERT (fine-tuned), 2 эпохи на подвыборке 20 000 комментариев

        Из-за сильного дисбаланса классов (~10:1) в проекте отдельно исследован
        trade-off между precision и recall, порог классификации настраивается вручную.
        """
    )
