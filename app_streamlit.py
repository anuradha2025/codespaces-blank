"""
app.py - Streamlit chat UI for the diabetic-retinopathy assistant.

Flow: the user attaches a fundus photo (paper-clip in the chat box) and/or types a question.
  1. the photo is decoded + checked, then classified by the CNN on the server (CPU),
  2. a result card with Grad-CAM is shown,
  3. a chat LLM explains the result and answers follow-up questions. The LLM receives only the
     numeric result (never the image), so patient images are not sent to a third party.
"""
import logging

import pandas as pd
import streamlit as st

from dr_core import (MAX_LLM_CALLS, MODEL_FACTS, DRClassifier, OfflineExplainer, build_system_prompt,
                     decode_image, make_llm, merge_messages)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dr_chatbot")

st.set_page_config(page_title="DR Screening Assistant", page_icon="👁️", layout="centered")


@st.cache_resource(show_spinner="Loading the retinopathy model (first start only)...")
def load_classifier() -> DRClassifier:
    return DRClassifier()


@st.cache_resource
def load_llm():
    return make_llm()


ss = st.session_state
ss.setdefault("messages", [])      # everything shown in the chat
ss.setdefault("analysis", None)    # latest classifier result (given to the LLM as context)
ss.setdefault("llm_calls", 0)      # per-session cost guard

try:
    classifier = load_classifier()
except Exception as e:             # most common cause: model files missing / Keras version mismatch
    st.error(f"The model could not be loaded: {e}")
    st.stop()
llm = load_llm()

# --------------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------------
with st.sidebar:
    st.subheader("About")
    st.caption(f"Chat engine: {llm.name}")
    with st.expander("How reliable is the model?"):
        f = MODEL_FACTS
        st.markdown(
            f"- Test accuracy **{f['test_accuracy']:.0%}**, macro-F1 **{f['test_macro_f1']:.2f}**, kappa **{f['test_qwk']:.2f}**\n"
            f"- Recall per stage: " + ", ".join(f"{k} {v:.0%}" for k, v in f["per_class_recall"].items()) + "\n"
            f"- Any DR: sensitivity {f['dr_present']['sensitivity']:.0%}, specificity {f['dr_present']['specificity']:.0%}\n"
            f"- Referable DR: sensitivity {f['referable_dr']['sensitivity']:.0%}, specificity {f['referable_dr']['specificity']:.0%}\n\n"
            "Severe and Proliferative cases are often under-called. Single dataset, no external validation.")
    if st.button("Clear conversation"):
        ss.messages, ss.analysis, ss.llm_calls = [], None, 0
        st.rerun()

st.title("👁️ Diabetic Retinopathy Assistant")
st.warning("Educational prototype - **not a medical device and not a diagnosis**. "
           "Always have retinal images reviewed by a qualified eye doctor. "
           "Uploaded images are processed in memory and are not stored. "
           "Do not type personal or identifying information into the chat.")


# --------------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------------
def render_analysis(a):
    c1, c2 = st.columns(2)
    if a.display is not None:
        c1.image(a.display, caption="Retina (cropped)", use_container_width=True)
    if a.overlay is not None:
        c2.image(a.overlay, caption="Grad-CAM: regions that influenced the estimate", use_container_width=True)
    st.markdown(a.card_markdown())
    st.bar_chart(pd.DataFrame({"probability": a.probs}, index=a.class_names))
    for w in a.warnings:
        st.warning(w)


def render(m: dict):
    with st.chat_message(m["role"]):
        if m.get("kind") == "analysis":
            render_analysis(m["analysis"])
        else:
            if m.get("image") is not None:
                st.image(m["image"], width=220)
            st.markdown(m["content"])


if not ss.messages:
    st.info("Attach a retinal fundus photo with the paper-clip in the box below, then ask me about the result. "
            "You can also ask general questions about diabetic retinopathy.")
for m in ss.messages:
    render(m)

# --------------------------------------------------------------------------------------------
# Handle new input
# --------------------------------------------------------------------------------------------
prompt = st.chat_input("Attach a retinal photo and/or ask a question",
                       accept_file=True, file_type=["png", "jpg", "jpeg"])
if prompt:
    if isinstance(prompt, str):
        text, files = prompt, []
    else:
        text, files = (prompt["text"] or ""), list(prompt["files"] or [])
    text = text.strip()

    analysis = None
    if files:
        try:
            img = decode_image(files[0].getvalue())
            with st.spinner("Analysing the image..."):
                analysis = classifier.analyze(img)
        except ValueError as e:
            st.error(str(e))
        except Exception:
            log.exception("analysis failed")
            st.error("Sorry, something went wrong while analysing that image.")
        if len(files) > 1:
            st.info("Only the first image was analysed.")

    if analysis is None and not text:
        st.stop()

    user_text = text or "Please analyse this image."
    user_msg = {"role": "user", "content": user_text,
                "image": analysis.display if analysis is not None else None,
                "llm_text": user_text + ("\n(The user attached a retinal image; the classifier result is in the system prompt.)"
                                         if analysis is not None else "")}
    ss.messages.append(user_msg)
    render(user_msg)

    if analysis is not None:
        ss.analysis = analysis
        a_msg = {"role": "assistant", "kind": "analysis", "analysis": analysis, "content": analysis.card_markdown()}
        ss.messages.append(a_msg)
        render(a_msg)

    # ---- chat LLM reply ---------------------------------------------------------------------
    history = [{"role": m["role"], "content": m.get("llm_text", m["content"])}
               for m in ss.messages if m.get("kind") != "analysis"]
    system = build_system_prompt(ss.analysis)
    msgs = merge_messages(history)

    with st.chat_message("assistant"):
        if ss.llm_calls >= MAX_LLM_CALLS:
            reply = "This session has reached its message limit. Please click 'Clear conversation' to start again."
            st.markdown(reply)
        else:
            try:
                reply = st.write_stream(llm.stream(system, msgs))
            except Exception:
                log.exception("LLM call failed")
                st.caption("The AI chat service is unavailable right now, so here is a built-in explanation instead.")
                reply = st.write_stream(OfflineExplainer().stream(system, msgs))
            ss.llm_calls += 1
    ss.messages.append({"role": "assistant", "kind": "chat", "content": reply})
