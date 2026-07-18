import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "haisassan/toxic-comment-distilbert"


@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()

st.set_page_config(page_title="Toxic Comment Detector", page_icon="🛡️")

st.title("🛡️ Toxic Comment Detector")

text_input = st.text_area("Type your comment: ", height=120, placeholder="Type or paste a comment here...")

threshold = st.slider(
    "Classification threshold (threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="Below the threshold, the model is more sensitive (higher recall). Above the threshold, it is more conservative (higher precision)."
)

if st.button("Check", type="primary"):
    if text_input.strip():
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        toxic_prob = probs[0][1].item()

        st.metric("Toxic probability:", f"{toxic_prob:.1%}")
        st.progress(toxic_prob)

        if toxic_prob >= threshold:
            st.error("⚠️ The comment has been classified as toxic")
        else:
            st.success("✅ The comment is not classified as toxic.")
    else:
        st.warning("Write text to check")

with st.expander("About project"):
    st.write(
        """
        **Dataset:** Jigsaw Toxic Comment Classification Challenge (Kaggle)

        **Baseline:** TF-IDF + Logistic Regression (class_weight='balanced')

        **Final model:** DistilBERT (fine-tuned), 2 epochs

        Due to a significant class imbalance (~10:1), the trade-off between precision and recall was specifically investigated in the project, and the classification threshold is adjusted manually.
        """
    )
