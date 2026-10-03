"""
chat_logic.py
-------------
The chat behaviour of the floating assistant, written WITHOUT any Gradio code so it can be unit-tested.
app_gradio.py only wires these functions to the widgets.

State that lives per browser session (gr.State):
  analysis  : slim Analysis of the latest image (from the main panel OR from the chat) -> given to the LLM
  llm_hist  : [{"role","content"}] text-only history sent to the LLM (analysis cards are NOT included)
  calls     : number of LLM replies so far (cost guard)
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Iterator, Optional

from dr_core import (MAX_LLM_CALLS, Analysis, OfflineExplainer, build_system_prompt, decode_image,
                     merge_messages)

log = logging.getLogger("dr_chatbot")

GREETING = ("Hi! I'm the assistant for this diabetic-retinopathy prototype. Attach a retinal photo with the "
            "paper-clip and I'll analyse it, or ask me about a result from the main panel. "
            "I'm not a doctor, so please have any real concern checked by an eye specialist.")


def slim(a: Optional[Analysis]) -> Optional[Analysis]:
    """Drop the image arrays so the per-session state stays tiny."""
    return None if a is None else dataclasses.replace(a, display=None, overlay=None)


def extract_message(message) -> tuple[str, Optional[str]]:
    """MultimodalTextbox value -> (text, first image path). Tolerates str / dict / FileData-like files."""
    if isinstance(message, dict):
        text = (message.get("text") or "").strip()
        files = message.get("files") or []
    else:
        text, files = str(message or "").strip(), []
    path = None
    for f in files:
        if isinstance(f, str):
            path = f
        elif isinstance(f, dict):
            path = f.get("path") or f.get("name")
        else:
            path = getattr(f, "path", None) or getattr(f, "name", None)
        if path:
            break
    return text, path


def handle_chat(message, ui_hist, analysis, llm_hist, calls, classifier, llm,
                save_image: Callable, max_calls: int = MAX_LLM_CALLS) -> Iterator[tuple]:
    """
    Generator yielding (ui_history, analysis, llm_hist, calls) after every visible change,
    so the chat window updates while the reply streams in.
    `save_image(rgb_array) -> file path` stores a small PNG that the chat window can display.
    """
    ui = list(ui_hist or [])
    lh = list(llm_hist or [])
    text, path = extract_message(message)
    if not text and not path:
        yield ui, analysis, lh, calls
        return

    attached = None                                         # Analysis of an image attached in this turn
    if path:
        try:
            with open(path, "rb") as fh:
                img = decode_image(fh.read())
            attached = classifier.analyze(img)
        except ValueError as e:                             # unreadable / too large
            ui.append({"role": "assistant", "content": str(e)})
        except Exception:
            log.exception("analysis failed")
            ui.append({"role": "assistant", "content": "Sorry, something went wrong while analysing that image."})

    if attached is not None:
        analysis = slim(attached)
        ui.append({"role": "user", "content": {"path": save_image(attached.display)}})
        if attached.overlay is not None:
            ui.append({"role": "assistant", "content": {"path": save_image(attached.overlay)}})
        card = attached.card_markdown()
        if attached.warnings:
            card += "\n\n" + "\n".join(f"⚠️ {w}" for w in attached.warnings)
        ui.append({"role": "assistant", "content": card})

    if not text and attached is None:                       # only a broken upload: nothing to ask the LLM
        yield ui, analysis, lh, calls
        return

    user_text = text or "Please analyse this image."
    if text:
        ui.append({"role": "user", "content": text})
    llm_user = user_text + ("\n(The user attached a retinal image; the classifier result is in the system prompt.)"
                            if attached is not None else "")
    lh.append({"role": "user", "content": llm_user})
    yield ui, analysis, lh, calls

    if calls >= max_calls:
        ui.append({"role": "assistant", "content": "This chat has reached its message limit. Please press 'Clear chat' to start again."})
        yield ui, analysis, lh, calls
        return

    system = build_system_prompt(analysis)
    msgs = merge_messages(lh)
    ui.append({"role": "assistant", "content": ""})
    reply = ""
    try:
        for chunk in llm.stream(system, msgs):
            reply += chunk
            ui[-1] = {"role": "assistant", "content": reply}
            yield ui, analysis, lh, calls
    except Exception:
        log.exception("LLM call failed")
        if reply:
            reply += "\n\n(The connection to the AI service was interrupted.)"
        else:
            reply = ("The AI chat service is unavailable right now, so here is a built-in explanation instead.\n\n"
                     + "".join(OfflineExplainer().stream(system, msgs)))
        ui[-1] = {"role": "assistant", "content": reply}
    if not reply:
        reply = "Sorry, I could not produce an answer. Please try again."
        ui[-1] = {"role": "assistant", "content": reply}
    lh.append({"role": "assistant", "content": reply})
    yield ui, analysis, lh, calls + 1
