"""
Run with:  python tests/test_core.py      (no TensorFlow / Streamlit / API key needed)
Uses a fake model and a fake `anthropic` module to test everything except the real CNN and the UI.
"""
import io
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dr_core as core  # noqa: E402


def synthetic_fundus(h=600, w=800):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), int(min(h, w) * 0.45), (190, 90, 40), -1)
    for i in range(20):
        cv2.line(img, (w // 2, h // 2), (w // 2 + int(250 * np.cos(i)), h // 2 + int(250 * np.sin(i))), (120, 30, 20), 2)
    return img


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class FakeModel:
    def __init__(self, probs):
        self.probs = np.array([probs], dtype="float32")
        self.calls = 0

    def predict_on_batch(self, x):
        assert x.shape[1:] == (256, 256, 3) and x.dtype == np.float32
        self.calls += 1
        return self.probs


def test_decode_image():
    arr = synthetic_fundus()
    out = core.decode_image(to_png_bytes(arr))
    assert out.shape == arr.shape and out.dtype == np.uint8
    for bad in (b"not an image", b""):
        try:
            core.decode_image(bad)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "PNG or JPEG" in str(e)
    old = core.MAX_PIXELS
    core.MAX_PIXELS = 1000
    try:
        core.decode_image(to_png_bytes(arr))
        raise AssertionError("expected ValueError for huge image")
    except ValueError:
        pass
    finally:
        core.MAX_PIXELS = old


def test_fundus_warnings():
    assert core.fundus_warnings(synthetic_fundus()) == []                       # looks like a fundus photo
    rng = np.random.RandomState(0)
    photo = (rng.rand(480, 640, 3) * 255).astype(np.uint8)                       # random colourful picture
    assert any("does not look like" in w for w in core.fundus_warnings(photo))
    white = np.full((400, 400, 3), 240, np.uint8)                                # a white page
    assert any("does not look like" in w for w in core.fundus_warnings(white))
    tiny = synthetic_fundus(100, 120)
    assert any("small" in w for w in core.fundus_warnings(tiny))


def test_classifier_and_analysis():
    clf = core.DRClassifier(model=FakeModel([0.05, 0.10, 0.70, 0.10, 0.05]))
    a = clf.analyze(synthetic_fundus(), with_cam=False)
    assert a.stage == 2 and a.stage_name == "Moderate"
    assert abs(sum(a.probs) - 1) < 1e-6
    assert abs(a.p_any - 0.95) < 1e-6 and abs(a.p_referable - 0.85) < 1e-6
    assert a.display.shape == (256, 256, 3)
    ctx = a.to_context()
    assert "Predicted stage: 2 (Moderate)" in ctx and "referable DR" in ctx
    assert "stage 2" in a.card_markdown()
    # low-confidence flag
    clf2 = core.DRClassifier(model=FakeModel([0.3, 0.3, 0.2, 0.1, 0.1]))
    assert any("not confident" in w for w in clf2.analyze(synthetic_fundus(), with_cam=False).warnings)
    # a 9-output (v2-style) model: only the first 5 values are used
    clf3 = core.DRClassifier(model=FakeModel([0.9, 0.05, 0.02, 0.02, 0.01, 0.9, 0.1, 0.1, 0.1]))
    assert clf3.analyze(synthetic_fundus(), with_cam=False).stage == 0


def test_prompting():
    a = core.Analysis(probs=[0.1, 0.1, 0.1, 0.6, 0.1], class_names=core.DEFAULT_CLASS_NAMES)
    s = core.build_system_prompt(a)
    assert "<classifier_result>" in s and "Predicted stage: 3" in s and "Per-stage recall" in s
    assert "No image has been analysed" in core.build_system_prompt(None)
    merged = core.merge_messages([
        {"role": "assistant", "content": "hi"},          # must be dropped (conversation must start with user)
        {"role": "user", "content": "a"}, {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"}, {"role": "user", "content": "d"}])
    assert [m["role"] for m in merged] == ["user", "assistant", "user"]
    assert merged[0]["content"] == "a\n\nb"
    assert len(core.merge_messages([{"role": "user", "content": str(i)} for i in range(50)], max_messages=12)) == 1


def test_offline_explainer():
    off = core.OfflineExplainer()
    a = core.Analysis(probs=[0.0, 0.0, 0.0, 0.0, 1.0], class_names=core.DEFAULT_CLASS_NAMES)
    txt = "".join(off.stream(core.build_system_prompt(a), []))
    assert "prompt assessment" in txt and "not a diagnosis" in txt
    assert "attach a retinal" in "".join(off.stream(core.build_system_prompt(None), [])).lower()


def test_llm_selection_and_claude_client():
    for k in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_MODEL"):
        os.environ.pop(k, None)
    assert isinstance(core.make_llm(), core.OfflineExplainer)                    # nothing configured

    # fake `anthropic` package that records the call
    seen = {}

    class FakeStream:
        def __init__(self, **kw):
            seen.update(kw)

        def __enter__(self):
            self.text_stream = iter(["Hello", " world"])
            return self

        def __exit__(self, *a):
            return False

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: types.SimpleNamespace(messages=types.SimpleNamespace(stream=lambda **kw: FakeStream(**kw)))
    sys.modules["anthropic"] = fake
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    llm = core.make_llm()
    assert isinstance(llm, core.ClaudeLLM) and "haiku" in llm.model
    out = "".join(llm.stream("SYS", [{"role": "user", "content": "hi"}]))
    assert out == "Hello world"
    assert seen["system"] == "SYS" and seen["messages"][0]["content"] == "hi" and seen["max_tokens"] > 0
    os.environ["ANTHROPIC_MODEL"] = "claude-sonnet-5-5"
    assert core.make_llm().model == "claude-sonnet-5-5"
    # missing optional package -> graceful fallback, no crash
    os.environ["LLM_PROVIDER"] = "azure_openai"
    os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
    assert isinstance(core.make_llm(), core.OfflineExplainer)
    for k in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        os.environ.pop(k, None)


def test_openai_compatible_provider():
    for k in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        os.environ.pop(k, None)
    seen = {}

    def chunk(text):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=text))])

    class FakeCompletions:
        def create(self, **kw):
            seen.update(kw)
            empty = types.SimpleNamespace(choices=[])                 # some servers send an empty first chunk
            return iter([empty, chunk("Hi"), chunk(None), chunk(" there")])

    class FakeOpenAI:
        def __init__(self, base_url, api_key):
            seen["base_url"], seen["api_key"] = base_url, api_key
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake = types.ModuleType("openai")
    fake.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake
    os.environ.update(LLM_API_KEY="k", LLM_BASE_URL="https://api.groq.com/openai/v1", LLM_MODEL="openai/gpt-oss-120b")
    llm = core.make_llm()                                             # auto-detected from LLM_API_KEY
    assert isinstance(llm, core.OpenAICompatLLM) and "groq" in llm.name
    out = "".join(llm.stream("SYS", [{"role": "user", "content": "hello"}]))
    assert out == "Hi there"
    assert seen["messages"][0] == {"role": "system", "content": "SYS"} and seen["model"] == "openai/gpt-oss-120b"
    assert seen["base_url"].startswith("https://api.groq.com")
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        os.environ.pop(k, None)


def test_dotenv_loader():
    import tempfile
    p = Path(tempfile.mkdtemp()) / ".env"
    p.write_text("# comment\nFOO_TEST_A=hello   # inline comment\nFOO_TEST_B=\nFOO_TEST_C=\"quoted\"\nFOO_TEST_D=keep\n")
    os.environ.pop("FOO_TEST_A", None); os.environ.pop("FOO_TEST_B", None); os.environ.pop("FOO_TEST_C", None)
    os.environ["FOO_TEST_D"] = "already-set"
    core._load_dotenv(p)
    assert os.environ["FOO_TEST_A"] == "hello" and os.environ["FOO_TEST_C"] == "quoted"
    assert "FOO_TEST_B" not in os.environ                 # empty values ignored
    assert os.environ["FOO_TEST_D"] == "already-set"      # real environment wins
    core._load_dotenv(Path("/nonexistent/.env"))          # missing file is fine


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
