"""
app_gradio.py
-------------
The existing Gradio UI (image in -> predicted stage + Grad-CAM out) PLUS a floating chat assistant
in the bottom-right corner. Click the chat button to open/close the panel. In the panel the user can
type questions and/or attach a retinal photo. The assistant also knows the result of the image analysed
in the main panel, so the user can simply ask "what does my result mean?".

Run:  python app_gradio.py          (reads .env style variables from the environment)
"""
import logging
import os
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr

from chat_logic import GREETING, handle_chat, slim
from dr_core import BASE_DIR, DRClassifier, make_llm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dr_chatbot")

classifier = DRClassifier()          # loads model/dr_model.keras once
llm = make_llm()                     # Claude / Gemini / Groq / ... or offline mode
log.info("Chat engine: %s", llm.name)

# Small PNGs shown inside the chat window are written here and deleted after one hour.
IMG_DIR = Path(os.getenv("CHAT_IMG_DIR", BASE_DIR / "chat_images"))
IMG_DIR.mkdir(parents=True, exist_ok=True)


def save_image(arr) -> str:
    now = time.time()
    for old in IMG_DIR.glob("*.png"):
        try:
            if now - old.stat().st_mtime > 3600:
                old.unlink()
        except OSError:
            pass
    p = IMG_DIR / f"{uuid.uuid4().hex}.png"
    cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return str(p)


# --------------------------------------------------------------------------------------------
# Styling of the floating widget (injected through gr.HTML so it works across Gradio versions)
# --------------------------------------------------------------------------------------------
STYLE = """
<style>
#css-holder { display: none !important; }
#chat-fab {
    position: fixed !important; right: 22px; bottom: 22px; z-index: 1001;
    width: auto !important; min-width: 0 !important; border-radius: 999px !important;
    padding: 12px 20px !important; box-shadow: 0 4px 16px rgba(0,0,0,.4);
}
#chat-panel {
    position: fixed !important; right: 22px; bottom: 80px; z-index: 1000;
    width: 390px; max-width: calc(100vw - 44px); max-height: calc(100vh - 110px); overflow-y: auto;
    background: var(--background-fill-primary); border: 1px solid var(--border-color-primary);
    border-radius: 14px; box-shadow: 0 10px 34px rgba(0,0,0,.5); padding: 10px;
}
</style>
"""


def make_chatbot():
    kwargs = dict(value=[{"role": "assistant", "content": GREETING}], height=340, show_label=False)
    try:
        return gr.Chatbot(type="messages", **kwargs)          # Gradio 5.x
    except TypeError:
        return gr.Chatbot(**kwargs)                            # Gradio 6+ (messages format is the only one)


# --------------------------------------------------------------------------------------------
# Main panel logic (unchanged behaviour: stage probabilities + Grad-CAM)
# --------------------------------------------------------------------------------------------
def predict_fundus(img_rgb, analysis):
    if img_rgb is None:
        return None, None, analysis
    a = classifier.analyze(img_rgb)
    for w in a.warnings:
        gr.Warning(w)
    probs = {name: float(p) for name, p in zip(a.class_names, a.probs)}
    return probs, (a.overlay if a.overlay is not None else a.display), slim(a)   # slim copy is shared with the chat


def chat_submit(message, ui_hist, analysis, llm_hist, calls):
    cleared = gr.MultimodalTextbox(value=None, interactive=True)
    for ui, new_analysis, new_lh, new_calls in handle_chat(message, ui_hist, analysis, llm_hist, calls,
                                                           classifier, llm, save_image):
        yield ui, new_analysis, new_lh, new_calls, cleared


def chat_clear():
    return [{"role": "assistant", "content": GREETING}], [], 0


def toggle(is_open):
    now_open = not is_open
    return (gr.update(visible=now_open), now_open,
            gr.update(value="✕ Close chat" if now_open else "💬 Chat with AI assistant"))


# --------------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------------
with gr.Blocks(title="Diabetic Retinopathy Stage Detection") as demo:
    gr.HTML(STYLE, elem_id="css-holder")
    gr.Markdown("<h1 style='text-align:center'>Diabetic Retinopathy Stage Detection (educational prototype)</h1>")
    gr.Markdown("NOT a medical device. For coursework demonstration only.")

    analysis_state = gr.State(None)    # latest analysis, shared between the main panel and the chat
    llm_hist = gr.State([])
    calls = gr.State(0)
    panel_open = gr.State(False)

    # ---- main panel (same layout as before) -------------------------------------------------
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Retinal fundus photograph")
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=5, label="Predicted DR stage")
            out_cam = gr.Image(label="Grad-CAM")

    submit_btn.click(predict_fundus, inputs=[inp, analysis_state], outputs=[out_label, out_cam, analysis_state])
    clear_btn.click(lambda: (None, None, None), outputs=[inp, out_label, out_cam])

    # ---- floating chat widget (bottom-right) ------------------------------------------------
    fab = gr.Button("💬 Chat with AI assistant", elem_id="chat-fab", variant="primary")
    with gr.Column(visible=False, elem_id="chat-panel") as panel:
        gr.Markdown("**AI assistant** - educational only, not medical advice")
        chatbot = make_chatbot()
        chat_in = gr.MultimodalTextbox(file_types=["image"], show_label=False,
                                       placeholder="Ask a question or attach a retinal photo...")
        with gr.Row():
            chat_clear_btn = gr.Button("Clear chat", size="sm")
            close_btn = gr.Button("Close", size="sm")

    fab.click(toggle, inputs=panel_open, outputs=[panel, panel_open, fab])
    close_btn.click(lambda: (gr.update(visible=False), False, gr.update(value="💬 Chat with AI assistant")),
                    outputs=[panel, panel_open, fab])
    chat_in.submit(chat_submit, inputs=[chat_in, chatbot, analysis_state, llm_hist, calls],
                   outputs=[chatbot, analysis_state, llm_hist, calls, chat_in])
    chat_clear_btn.click(chat_clear, outputs=[chatbot, llm_hist, calls])

if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
                share=os.getenv("GRADIO_SHARE", "0") == "1",          # GRADIO_SHARE=1 for a public link from Colab
                allowed_paths=[str(IMG_DIR)])
