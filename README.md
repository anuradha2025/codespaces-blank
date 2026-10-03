# DR Assistant: existing Gradio UI + floating AI chat (bottom-right)

`app_gradio.py` keeps your current page (image in → predicted stage + Grad-CAM out) and adds a **💬 Chat with AI assistant**
button fixed at the bottom-right. It opens a chat panel where the user can type questions and/or attach a retinal photo.
The assistant also knows the result of the image analysed in the main panel, so "what does my result mean?" just works.

```
browser ──► Caddy (HTTPS) ──► Gradio app ──► CNN (TensorFlow CPU, EfficientNet-B0 v1 model)
                                   └───────► chat LLM API (Gemini / Groq / Claude / ...) – text only, never the image
```

## Files
| File | Purpose |
|---|---|
| `app_gradio.py` | The UI: main panel + floating chat widget. **Run this.** |
| `chat_logic.py` | Chat behaviour (image upload in chat, streaming, fallbacks, message limit); no UI code |
| `dr_core.py` | Model wrapper, Grad-CAM, image checks, prompts, LLM providers |
| `preprocess_utils.py` | Same pre-processing as training |
| `app_streamlit.py` | Alternative full-page chat UI (optional, needs `pip install streamlit`) |
| `.env.example` | Settings (LLM provider/key, domain); copy to `.env` (never commit it) |
| `deploy/upload.sh`, `deploy/setup_vm.sh` | One-command deployment to the Azure VM |

## 1. Model files
Put `dr_model.keras` and `model_meta.json` (downloaded from the Colab `/content/dr_work/outputs/` folder) into `model/`.

## 2. Chat model (`.env`)
Free option with Google AI Studio:
```
LLM_PROVIDER=openai_compat
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=<your AI Studio key>
LLM_MODEL=gemini-flash-latest      # check AI Studio for the current free Flash model name
```
Other providers (Groq, OpenRouter, Claude, Azure OpenAI) are in `.env.example`. With no key the app runs in offline mode (the CNN works; the chat gives built-in explanations).

## 3. Run locally (to test)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_core.py && python tests/test_chat.py     # self-tests, no model needed
export LLM_PROVIDER=openai_compat LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...
python app_gradio.py                                       # http://localhost:7860
```
In Colab use `GRADIO_SHARE=1 python app_gradio.py` (or set `os.environ["GRADIO_SHARE"]="1"` before running it) to get a public link.

## 4. Deploy on the Azure VM (4 GB RAM, 2 vCPU)
**Portal steps (once):**
1. VM → *Networking* → *Add inbound port rule*: allow **80** (HTTP) and **443** (HTTPS). Keep **22** (SSH).
2. VM → *Overview* → click the **Public IP address** → *Configuration* → set a **DNS name label**
   (gives `<label>.<region>.cloudapp.azure.com`, which keeps working after stop/start).
3. In `.env` set `DOMAIN=<that name>` for HTTPS, or `DOMAIN=:80` to start with plain HTTP on the IP.

**Deploy (one command, from Codespaces or your computer):**
```bash
bash deploy/upload.sh AZURE_USER@VM_PUBLIC_IP path/to/key.pem     # key optional if you use a password
```
It uploads the project (model and `.env` included), adds swap, installs Docker and starts everything.
First run takes 10-15 minutes (TensorFlow is large). Re-run the same command after any code or `.env` change.

Useful on the VM: `cd ~/dr_chatbot && sudo docker compose logs -f app` | `sudo docker compose ps` | `sudo docker stats`.
To save credits, *Stop* the VM in the portal when you are not using it.

## Safety / privacy (for the report)
* Banner on the page, assistant prompt forbids definitive diagnosis, always recommends an eye doctor, urgent-symptom advice built in.
* The assistant is told the model's real weaknesses (Severe recall 35 %, Proliferative 42 %) so it does not over-reassure.
* Images are **never sent to the LLM provider**. Small preview PNGs shown in the chat are kept in `chat_images/` for at most one hour; Gradio also keeps uploads in its own temporary cache.
* Soft warning when an upload does not look like a fundus photo; per-session cap on chat replies (`MAX_LLM_CALLS_PER_SESSION`).
* Free LLM tiers may use prompts for product improvement (Google) - the UI tells users not to type personal information.

## Troubleshooting
* *Chat button works but panel looks wrong / unstyled*: report your Gradio version (`pip show gradio`).
* *"Model could not be loaded"*: files missing in `model/`, or a Keras version mismatch (model saved with Keras 3.13 / TF 2.20; pinned in `requirements.txt`).
* *Sidebar says Offline mode*: wrong key / model name; see `docker compose logs app`.
* *Out of memory*: `docker stats`, `free -h` (swap active?). The app needs roughly 1-1.5 GB.
