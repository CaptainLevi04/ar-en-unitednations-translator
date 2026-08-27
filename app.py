import re
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(page_title="AR <-> EN Translator", page_icon="\U0001F310")


# Models are loaded straight from the Hugging Face Hub
MODEL_PATHS = {
    "ar-en": "Amr04/opus-mt-ar-en-un-finetuned",   # Arabic -> English
    "en-ar": "Amr04/opus-mt-en-ar-un-finetuned",   # English -> Arabic
}


ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
ENGLISH_RE = re.compile(r"[A-Za-z]")


# Load each model only once and cache it
@st.cache_resource
def load_model(repo_id):
    tokenizer = MarianTokenizer.from_pretrained(repo_id)
    model = MarianMTModel.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model


def detect_direction(text: str):
    """Counts Arabic vs English letters and picks the translation direction.
    Returns 'ar-en' if Arabic letters are more (or equal), else 'en-ar'."""

    arabic_count = len(ARABIC_RE.findall(text))
    english_count = len(ENGLISH_RE.findall(text))

    if arabic_count == 0 and english_count == 0:
        return None

    return "ar-en" if arabic_count >= english_count else "en-ar"


def translate(text, tokenizer, model, max_length=512, num_beams=4):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    generated = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )

    return tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )


st.title("Paragraphs Alligned Translator")
st.caption("Unlike most of NMT this is not only sentence alligned translator it can also translate paragraphs.")


text = st.text_area(
    "Enter text to translate / أدخل النص",
    height=150
)


if st.button("Translate / ترجم"):

    if not text.strip():

        st.warning(
            "Please enter some text first. / من فضلك اكتب نص."
        )

    else:

        direction_key = detect_direction(text)

        if direction_key is None:

            st.error(
                "Couldn't detect the language. Please enter Arabic or English text."
            )

        else:

            direction_label = (
                "Arabic -> English"
                if direction_key == "ar-en"
                else "English -> Arabic"
            )

            st.success(
                f"Detected direction: **{direction_label}**"
            )

            with st.spinner("Translating..."):

                # The model is downloaded/loaded only the first time.
                # Streamlit caches it afterwards.
                tokenizer, model = load_model(
                    MODEL_PATHS[direction_key]
                )

                result = translate(
                    text,
                    tokenizer,
                    model
                )

            st.write(result)


with st.expander("The app auto detects your language"):

    st.write(
        "Works better on the united nations documents or the UN style."
    )
