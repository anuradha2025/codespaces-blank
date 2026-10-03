"""
Run with:  python tests/test_chat.py     (no TensorFlow / Gradio / API key needed)
Tests the chat behaviour (chat_logic.py) and exercises app_gradio.py's wiring with a fake `gradio` module.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dr_core as core          # noqa: E402
import chat_logic as cl         # noqa: E402


class FakeModel:
    def __init__(self, p): self.p = np.array([p], dtype="float32")
    def predict_on_batch(self, x): return self.p


class FakeClassifier(core.DRClassifier):
    """Real DRClassifier logic with a fake model, and a fake Grad-CAM overlay."""
    def __init__(self, probs=(0.05, 0.1, 0.7, 0.1, 0.05)):
        super().__init__(model=FakeModel(list(probs)))
    def analyze(self, img, with_cam=True):
        a = super().analyze(img, with_cam=False)
        a.overlay = a.display.copy()
        return a


class FakeLLM:
    name = "fake"
    def __init__(self, chunks=("Hello", " there"), fail_after=None):
        self.chunks, self.fail_after, self.seen = chunks, fail_after, []
    def stream(self, system, messages):
        self.seen.append((system, messages))
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("boom")
            yield c


def make_image_file():
    import cv2
    img = np.zeros((600, 800, 3), np.uint8)
    cv2.circle(img, (400, 300), 270, (190, 90, 40), -1)
    p = os.path.join(tempfile.mkdtemp(), "eye.png")
    Image.fromarray(img).save(p)
    return p


def run(*a, **k):
    last = None
    states = []
    for last in cl.handle_chat(*a, **k):
        states.append(last)
    return last, states


def saver(arr):
    return "/tmp/fake.png"


def test_extract_message():
    assert cl.extract_message({"text": " hi ", "files": []}) == ("hi", None)
    assert cl.extract_message({"text": "", "files": ["/a.png"]}) == ("", "/a.png")
    assert cl.extract_message({"text": "x", "files": [{"path": "/b.png"}]}) == ("x", "/b.png")
    assert cl.extract_message(types.SimpleNamespace(text="t") if False else "plain") == ("plain", None)
    assert cl.extract_message(None) == ("", None)


def test_text_only_question_uses_shared_analysis():
    prior = cl.slim(FakeClassifier().analyze(np.zeros((300, 300, 3), np.uint8)))
    llm = FakeLLM()
    (ui, an, lh, calls), states = run({"text": "what does this mean?", "files": []}, [], prior, [], 0, FakeClassifier(), llm, saver)
    assert [m["role"] for m in ui] == ["user", "assistant"] and ui[-1]["content"] == "Hello there"
    assert calls == 1 and an is prior
    assert "Predicted stage: 2" in llm.seen[0][0]            # result from the main panel reaches the LLM
    assert lh[-1] == {"role": "assistant", "content": "Hello there"}
    assert len(states) >= 4                                   # streamed incrementally


def test_image_upload_in_chat():
    llm = FakeLLM()
    (ui, an, lh, calls), _ = run({"text": "", "files": [make_image_file()]}, [], None, [], 0, FakeClassifier(), llm, saver)
    assert an is not None and an.stage == 2 and an.display is None        # slim copy stored
    kinds = [(m["role"], isinstance(m["content"], dict)) for m in ui]
    assert kinds[:2] == [("user", True), ("assistant", True)]              # retina image + Grad-CAM image
    assert "stage 2" in ui[2]["content"]                                   # result card
    assert ui[-1]["content"] == "Hello there"
    assert "attached a retinal image" in lh[0]["content"]
    assert all(isinstance(m["content"], str) for m in lh)                  # LLM history is text only
    assert "Predicted stage: 2" in llm.seen[0][0]


def test_bad_upload_and_empty_message():
    bad = os.path.join(tempfile.mkdtemp(), "x.png")
    open(bad, "wb").write(b"not an image")
    (ui, an, lh, calls), _ = run({"text": "", "files": [bad]}, [], None, [], 0, FakeClassifier(), FakeLLM(), saver)
    assert "PNG or JPEG" in ui[-1]["content"] and calls == 0 and lh == [] and an is None
    (ui, an, lh, calls), _ = run({"text": "", "files": []}, [], None, [], 0, FakeClassifier(), FakeLLM(), saver)
    assert ui == [] and calls == 0


def test_llm_failure_falls_back_offline():
    prior = cl.slim(FakeClassifier((0, 0, 0, 0, 1.0)).analyze(np.zeros((300, 300, 3), np.uint8)))
    (ui, _, lh, calls), _ = run({"text": "explain", "files": []}, [], prior, [], 0, FakeClassifier(), FakeLLM(fail_after=0), saver)
    assert "unavailable" in ui[-1]["content"] and "prompt assessment" in ui[-1]["content"] and calls == 1
    (ui, _, _, _), _ = run({"text": "explain", "files": []}, [], prior, [], 0, FakeClassifier(), FakeLLM(fail_after=1), saver)
    assert ui[-1]["content"].startswith("Hello") and "interrupted" in ui[-1]["content"]


def test_message_limit():
    llm = FakeLLM()
    (ui, _, _, calls), _ = run({"text": "hi", "files": []}, [], None, [], 5, FakeClassifier(), llm, saver, max_calls=5)
    assert "message limit" in ui[-1]["content"] and calls == 5 and llm.seen == []


# ---------------------------------------------------------------------------------------------
# Wiring test of app_gradio.py with a permissive fake `gradio` module
# ---------------------------------------------------------------------------------------------
class Dummy:
    registry = []
    def __init__(self, *a, **k):
        self.a, self.k = a, k
        Dummy.registry.append(self)
    def __enter__(self): return self
    def __exit__(self, *x): return False
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return lambda *a, **k: None          # .click / .submit / .launch ...


def install_fake_gradio():
    g = types.ModuleType("gradio")
    for n in ("Blocks", "HTML", "Markdown", "State", "Row", "Column", "Image", "Button", "Label",
              "Chatbot", "MultimodalTextbox"):
        setattr(g, n, type(n, (Dummy,), {}))
    g.update = lambda **k: k
    g.Warning = lambda msg: None
    sys.modules["gradio"] = g
    return g


def test_app_gradio_wiring():
    install_fake_gradio()
    import dr_core
    dr_core.DRClassifier = lambda *a, **k: FakeClassifier()          # app does `from dr_core import DRClassifier`
    os.environ["CHAT_IMG_DIR"] = tempfile.mkdtemp()
    for k in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "LLM_API_KEY", "AZURE_OPENAI_API_KEY"):
        os.environ.pop(k, None)
    sys.modules.pop("app_gradio", None)
    app = __import__("app_gradio")                                   # builds the whole layout; must not raise
    # main panel
    img = np.zeros((600, 800, 3), np.uint8)
    probs, cam, shared = app.predict_fundus(img, None)
    assert abs(sum(probs.values()) - 1) < 1e-6 and cam.shape[2] == 3 and shared.display is None
    assert app.predict_fundus(None, shared)[2] is shared
    # chat: shared analysis flows into the chat; images get saved as PNG files; textbox is cleared each yield
    app.llm = FakeLLM()
    outs = list(app.chat_submit({"text": "explain my result", "files": []}, [], shared, [], 0))
    assert outs[-1][0][-1]["content"] == "Hello there" and outs[-1][3] == 1
    assert "Predicted stage: 2" in app.llm.seen[0][0]
    outs = list(app.chat_submit({"text": "", "files": [make_image_file()]}, [], None, [], 0))
    pngs = [m["content"]["path"] for m in outs[-1][0] if isinstance(m["content"], dict)]
    assert len(pngs) == 2 and all(os.path.exists(p) for p in pngs)
    # toggle
    vis, state, fab = app.toggle(False)
    assert vis == {"visible": True} and state is True and "Close" in fab["value"]
    vis, state, fab = app.toggle(True)
    assert vis == {"visible": False} and state is False
    ui, lh, calls = app.chat_clear()
    assert ui[0]["role"] == "assistant" and lh == [] and calls == 0
    # the floating-widget CSS targets the element ids used in the layout
    ids = {getattr(o, "k", {}).get("elem_id") for o in Dummy.registry}
    assert {"chat-fab", "chat-panel", "css-holder"} <= ids


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
