import streamlit as st
from src.classifier import train_or_load
from src.llm_optional import have_llm, llm_summarize, llm_reply

# ---------- Page setup ----------
st.set_page_config(page_title="Email Priority Assistant", page_icon="📧", layout="wide")
st.title("📧 Email Priority Assistant")

with st.sidebar:
    st.markdown("### How to use")
    st.markdown(
        "1) Enter an email **Subject** and **Body**\n"
        "2) Click **Classify & Draft**\n"
        "3) Review the **priority**, **summary**, and **reply**"
    )

@st.cache_resource(show_spinner=False)
def load_model():
    return train_or_load()

model = load_model()

# ---------- Layout ----------
colL, colR = st.columns([1, 1.2])

with colL:
    subject = st.text_input("Subject", value="")
    body = st.text_area("Body", height=240, value="")
    run = st.button("Classify & Draft", use_container_width=True)

with colR:
    if run:
        # Basic validation
        if not subject and not body:
            st.warning("Please enter a Subject or Body.")
            st.stop()

        # Require LLM for summary/reply
        if not have_llm():
            st.error(
                "LLM is required for Intent Summary and Reply.\n\n"
                "Add your OpenAI key in Streamlit Secrets:\n"
                "• Local: `.streamlit/secrets.toml` → `OPENAI_API_KEY = \"sk-...\"`\n"
                "• Streamlit Cloud: App → Settings → Secrets → `OPENAI_API_KEY`"
            )
            st.stop()

        # ---------- 1) ML priority ----------
        label, conf, proba, _ = model.predict(subject, body)
        color = {"Urgent": "red", "Normal": "blue", "Low": "gray"}[label]
        st.markdown(
            f"**Email Priority:** <span style='color:{color}'>{label}</span>",
            unsafe_allow_html=True,
        )

        full_text = (subject or "") + "\n\n" + (body or "")

        # ---------- 2) LLM summary ----------
        with st.spinner("Generating summary..."):
            summary, err_sum = llm_summarize(subject, body, label)
        if not summary:
            from src.heuristics import summarize_intent
            summary = summarize_intent(subject, body, label)
            st.warning("LLM could not produce a summary in time. Showing a simplified summary instead.")
        st.subheader("Intent Summary")
        st.write(summary)

        # ---------- 3) LLM reply ----------
        with st.spinner("Drafting reply..."):
            reply, err_rep = llm_reply(subject, body, label)
        if not reply:
            from src.heuristics import draft_reply
            reply = draft_reply(subject, body, label)
            st.warning("LLM could not draft a reply in time. Showing a basic fallback instead.")
        st.subheader("Reply Draft")
        reply_text = st.text_area("You can edit before copying:", value=reply, height=220, key="reply_box")

        # Optional: quick download of the reply as a .txt file
        st.download_button(
            "Download Reply (.txt)",
            data=reply_text,
            file_name="reply_draft.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("Enter an email subject and body, then click **Classify & Draft**.")

st.divider()
