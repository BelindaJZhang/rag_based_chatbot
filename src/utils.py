# utils.py (Gemini version)

from typing import List, Dict, Any, Union
import os
from google import genai
import ipywidgets as widgets
from IPython.display import display
import json

# ---- Client -----------------------------------------------------------
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Prefer fast text model; you can switch to pro if you want
PREFERRED_TEXT_MODEL = "models/gemini-2.0-flash"
EMBED_MODEL = "models/text-embedding-004"

# ---- Helpers: picking a valid model ----------------------------------
def _pick_model(preferred: str = PREFERRED_TEXT_MODEL) -> str:
    names = [m.name for m in client.models.list()]
    if preferred in names:
        return preferred
    # fallbacks that support generateContent
    for cand in (
        "models/gemini-2.5-pro",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-001",
    ):
        if cand in names:
            return cand
    for m in client.models.list():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            return m.name
    raise RuntimeError("No usable generation model available for this API key.")

# ---- Param guards (keeps sweeps from crashing) -----------------------
def _clamp(x, lo, hi):
    if x is None: return None
    x = float(x)
    return max(lo, min(hi, x))

def _validate_and_prepare_config(
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    *,
    strict: bool = False,
    response_mime_type: str | None = None,
    response_schema: dict | None = None,
) -> dict:
    cfg = {}
    # Temperature
    if temperature is not None:
        if strict and not (0.0 <= float(temperature) <= 2.0):
            raise ValueError("temperature must be in [0.0, 2.0].")
        cfg["temperature"] = _clamp(temperature, 0.0, 2.0)
    # Top-p sampling
    if top_p is not None:
        p = float(top_p)
        if strict and not (0.0 < p <= 1.0):
            raise ValueError("top_p must be in (0.0, 1.0].")
        cfg["top_p"] = p if (0.0 < p <= 1.0) else 1.0
    # Token limit
    if max_tokens is not None:
        mt = int(max_tokens)
        if strict and mt < 1:
            raise ValueError("max_tokens must be >= 1.")
        cfg["max_output_tokens"] = max(1, mt)
    # ✅ New: MIME type enforcement (to get clean JSON output)
    if response_mime_type is not None:
        cfg["response_mime_type"] = response_mime_type

    # ✅ Optional: structured schema for future-proofing
    if response_schema is not None:
        cfg["response_schema"] = response_schema
    
    return cfg

# ---- Text generation: single input -----------------------------------
def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float | None = None,
    temperature: float | None = None,
    max_tokens: int = 500,
    mime_type: str = "text/plain",
    schema: dict | None = None,   # keep as dict or None
    model: str = PREFERRED_TEXT_MODEL,
    *,
    strict_params: bool = False,
    **kwargs
) -> Dict[str, Any]:
    mdl = _pick_model(model)

    # Build a plain dict config — no class import needed
    cfg_dict = _validate_and_prepare_config(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_mime_type=mime_type,  # e.g., "application/json"
        response_schema=schema,         # optional; some SDK versions ignore if unsupported
        strict=strict_params,
    )

    # Call the API (SDKs differ on arg name; try both)
    try:
        resp = client.models.generate_content(
            model=mdl,
            contents=prompt,
            config=cfg_dict,  # newer SDKs
        )
    except TypeError:
        resp = client.models.generate_content(
            model=mdl,
            contents=prompt,
            generation_config=cfg_dict,  # older SDKs
        )

    text = _extract_text(resp) or ""
    return {"role": "assistant", "content": text, "meta": _debug_resp(resp)}

# ---- Text generation: multiple inputs (chat history) -----------------
def generate_with_multiple_input(
    messages: List[Dict],
    top_p: float | None = None,
    temperature: float | None = None,
    max_tokens: int = 500,
    model: str = PREFERRED_TEXT_MODEL,
    *,
    strict_params: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    messages: [{'role': 'user'|'assistant'|'system', 'content': '...'}, ...]
    Returns: {'role': 'assistant', 'content': '<text>'}
    """
    model = _pick_model(model)

    # --- fold system into first user turn (unchanged) ---
    sys_preamble = []
    norm_msgs: List[Dict] = []
    for m in messages or []:
        r = (m.get("role") or "user").strip().lower()
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if r == "system":
            sys_preamble.append(c)
        else:
            norm_msgs.append({"role": r, "content": c})

    if sys_preamble:
        sys_text = "\n".join(sys_preamble)
        inserted = False
        for m in norm_msgs:
            if m["role"] == "user":
                m["content"] = sys_text + "\n\n" + m["content"]
                inserted = True
                break
        if not inserted:
            norm_msgs.insert(0, {"role": "user", "content": sys_text})

    # --- single-turn shortcut *after* system folding ---
    if len(norm_msgs) == 1 and norm_msgs[0]["role"] == "user":
        return generate_with_single_input(
            prompt=norm_msgs[0]["content"],
            role="user",
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            strict_params=strict_params,
            **kwargs
        )

    # --- Gemini role mapping (assistant -> model) (unchanged) ---
    def _map_role(r: str) -> str:
        return "model" if r == "assistant" else "user"

    contents = [
        {"role": _map_role(m["role"]), "parts": [{"text": m["content"]}]}
        for m in norm_msgs
    ]
    if not contents:
        raise ValueError("No valid messages to send (after cleaning).")
    
    # --- NEW: Ensure the last turn is a user turn for Gemini ---
    if contents[-1]["role"] != "user":
        contents.append({"role": "user", "parts": [{"text": ""}]})
    # --- config (unchanged) ---
    config = _validate_and_prepare_config(temperature, top_p, max_tokens, strict=strict_params)

    # --- call + robust text extraction (adds fallback for SDK arg name) ---
    try:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=(config or None),               # newer SDKs
            )
        except TypeError:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                generation_config=(config or None),    # older SDKs
            )

        text = _extract_text(resp)

        if not text or not str(text).strip():
            text = "Sorry — I couldn’t generate a response right now. Please try again."

        try:
            meta = _debug_resp(resp)
        except Exception as dbg_e:
            meta = {"_debug_resp_error": str(dbg_e)}

        return {"role": "assistant", "content": text, "meta": meta}

    except Exception as e:
        print("❌ Gemini call failed in generate_with_multiple_input:", e)
        return {
            "role": "assistant",
            "content": "Sorry — I couldn’t reach the language model just now. Please try again.",
            "meta": {"error": str(e)},
        }

# ---- Embeddings ------------------------------------------------------
from typing import List, Union

def generate_embedding(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings via Google 'text-embedding-004' across SDK minor versions.

    Tries different argument names and response shapes:
      - prefer: client.models.embed_content(model=..., contents=...)
      - response may expose:
          r.embedding.values
          r.embeddings[0].values
          r["embedding"]["values"]
          r["embeddings"][0]["values"]
    """

    def _extract_vec(r):
        # Object-style responses
        emb = getattr(r, "embedding", None)
        if emb is not None:
            vals = getattr(emb, "values", None)
            if vals is not None:
                return list(vals)
        embs = getattr(r, "embeddings", None)
        if embs:
            # embs might be a list of objects each with .values
            first = embs[0]
            vals = getattr(first, "values", None)
            if vals is not None:
                return list(vals)

        # Dict-style responses
        try:
            if isinstance(r, dict):
                if "embedding" in r and "values" in r["embedding"]:
                    return list(r["embedding"]["values"])
                if "embeddings" in r and r["embeddings"]:
                    first = r["embeddings"][0]
                    if "values" in first:
                        return list(first["values"])
        except Exception:
            pass

        raise RuntimeError("Could not extract embedding vector from response")

    def _embed_one(t: str) -> List[float]:
        last_err = None
        # Try the most common current signature first: contents=
        for kw in ("contents", "content", "input"):
            try:
                r = client.models.embed_content(model=EMBED_MODEL, **{kw: t})
                return _extract_vec(r)
            except TypeError as e:
                last_err = e  # wrong kw; try next
            except Exception as e:
                # Transport/safety/etc. — surface real errors
                raise
        # If we exhausted kw trials:
        raise TypeError(f"embed_content parameter mismatch; tried contents/content/input. Last error: {last_err}")

    if isinstance(text, str):
        return _embed_one(text)

    out: List[List[float]] = []
    for t in text:
        out.append(_embed_one(t))
    return out

# ---- Small utilities (optional) --------------------------------------
def _extract_text(resp) -> str:
    if getattr(resp, "text", None):
        return resp.text
    chunks = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", []) if content else []
        for p in parts or []:
            t = getattr(p, "text", None)
            if t:
                chunks.append(t)
    return "\n".join(chunks)

def _debug_resp(resp) -> dict:
    info = {}
    pf = getattr(resp, "prompt_feedback", None)
    if pf:
        info["blocked"] = getattr(pf, "blocked", None)
        info["safety_ratings"] = getattr(pf, "safety_ratings", None)
    info["finish_reasons"] = [getattr(c, "finish_reason", None)
                              for c in getattr(resp, "candidates", []) or []]
    return info

# ---- Keep this (unchanged) for your course exercises -----------------
def generate_params_dict(
    prompt: str, 
    temperature: float = None, 
    role: str = 'user',
    top_p: float = None,
    max_tokens: int = 500,
    model: str = PREFERRED_TEXT_MODEL
):
    return {"prompt": prompt, "role": role, "temperature": temperature,
            "top_p": top_p, "max_tokens": max_tokens, "model": model}

def call_llm_with_context(prompt: str, context: list, role: str = 'user', **kwargs):
    """
    Adds user turn to context, calls the LLM, appends the assistant reply,
    and returns the reply as a dict. Never returns None.
    """
    context.append({'role': role, 'content': prompt})
    try:
        resp = generate_with_multiple_input(context, **kwargs)
    except Exception as e:
        print(f"❌ LLM call failed in call_llm_with_context: {e}")
        resp = None

    # Coerce to a safe dict so callers never see None
    if not isinstance(resp, dict):
        resp = {"role": "assistant",
                "content": "Sorry — I couldn’t generate a reply just now. Please try again."}

    context.append(resp)
    return resp


# ---------- Chat controller (keeps state, calls your generator_function) ----------
import re, threading, os
import ipywidgets as widgets
from IPython.display import display
import markdown

class ChatBot:
    """
    Holds conversation state and delegates each user turn to your generator_function.
    generator_function(query: str) must return a dict like:
      {"prompt": "...", "role": "user", "temperature": None, "top_p": None,
       "max_tokens": 500, "model": PREFERRED_TEXT_MODEL}
    """
    def __init__(self, generator_function, model: str = None, context_window: int = 20):
        self.generator_function = generator_function
        self.context_window = context_window
        self.model = model  # optional override

        self.system_prompt = {
            "role": "system",
            "content": "You are a friendly assistant from Fashion Forward Hub. It is a cloth store selling a variety of items. Your job is to answer questions related to FAQ or Products."
        }
        self.initial_message = {"role": "assistant", "content": "Hi! How can I help you?"}
        self.conversation = [self.system_prompt, self.initial_message]

    def chat(self, prompt: str, role: str = "user"):
        recent_context = self.conversation[-self.context_window:]

        params = dict(self.generator_function(prompt) or {})   # generator might return None
        params.setdefault("prompt", prompt)
        params.setdefault("role", role)
        if self.model:
            params["model"] = self.model

        content = call_llm_with_context(context=recent_context, **params)

        # Ensure dict reply
        if not isinstance(content, dict):
            content = {"role": "assistant",
                    "content": "Sorry — something went wrong while generating a reply."}

        # Mirror into full conversation
        self.conversation.append({"role": role, "content": prompt})
        self.conversation.append(recent_context[-1])  # assistant reply appended by call_llm_with_context
        return content

# ---------- Notebook UI (ipywidgets chat + optional image strip) ----------
class ChatWidget:
    """
    Simple ipywidgets chat UI. If the model’s reply contains 'ID: 123, 456',
    it will attempt to load ./images/{id}.jpg (configurable) and display them.
    """
    def __init__(self, generator_function, image_base: str = "./images", enable_images: bool = True):
        self.chat_bot = ChatBot(generator_function)
        self.output_area = widgets.HTML()
        self.image_area = widgets.HBox()
        self.text_input = widgets.Text(placeholder="Type your message...", layout=widgets.Layout(width="90%"))
        self.send_button = widgets.Button(description="Send", layout=widgets.Layout(width="10%"))
        self.send_button.on_click(self.send_message)

        self.image_base = image_base
        self.enable_images = enable_images
        self.unique_ids = set()

        self.display()
        self.refresh_messages()

    # --- event & background work ---
    def send_message(self, _):
        user_message = self.text_input.value
        if not user_message or not user_message.strip():
            return
        self.display_user_message(user_message)
        self.show_thinking()
        self.text_input.value = ""
        self.image_area.children = ()
        threading.Thread(target=self.process_bot_response, args=(user_message,), daemon=True).start()

    def process_bot_response(self, user_message: str):
        response = self.chat_bot.chat(user_message)
        response_text = ""
        if isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            print("⚠️ No response returned by ChatBot.chat()")
        if self.enable_images:
            self.extract_and_process_ids(response_text)
        self.refresh_messages()

    # --- image plumbing (optional) ---
    def extract_and_process_ids(self, message: str):
        # Matches: "ID: 12" or "ID: 12, 34"
        pattern = re.compile(r"ID:\s*(\d+(?:,\s*\d+)*)", re.IGNORECASE)
        matches = pattern.findall(message or "")
        found_ids = [i.strip() for m in matches for i in m.split(",")]
        for id_str in found_ids:
            if id_str and id_str not in self.unique_ids:
                self.unique_ids.add(id_str)
                self.load_image(id_str)

    def load_image(self, id_str: str):
        # Default to ./images/{id}.jpg — change image_base when you construct the widget
        image_path = os.path.join(self.image_base, f"{id_str}.jpg")
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_data = f.read()
            img_widget = widgets.Image(value=img_data, format="jpg",
                                       layout=widgets.Layout(width="150px", height="auto", margin="5px"))
            id_label = widgets.Label(value=f"ID: {id_str}", layout=widgets.Layout(width="150px"))
            self.image_area.children += (widgets.VBox([img_widget, id_label]),)

    # --- rendering helpers ---
    def display_user_message(self, message: str):
        escaped = markdown.markdown(message or "")
        html = self.output_area.value
        html += (
            "<div style='margin:8px 0;padding:8px;border-radius:8px;background:#eef7ff;'>"
            "<strong>User:</strong>"
            f"<div style='margin-top:4px'>{escaped}</div>"
            "</div>"
        )
        self.output_area.value = html

    def show_thinking(self):
        html = self.output_area.value
        html += (
            "<div style='margin:8px 0;padding:8px;border-radius:8px;background:#fff8db;'>"
            "<strong>Assistant:</strong>"
            "<div style='margin-top:4px'>Thinking...</div>"
            "</div>"
        )
        self.output_area.value = html

    def refresh_messages(self):
        html = "<div style='font-family: Arial; max-width: 700px;'>"
        for m in self.chat_bot.conversation:
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            escaped = markdown.markdown(content)
            if role == "user":
                html += (
                    "<div style='margin:8px 0;padding:8px;border-radius:8px;background:#eef7ff;'>"
                    "<strong>User:</strong>"
                    f"<div style='margin-top:4px'>{escaped}</div>"
                    "</div>"
                )
            elif role == "assistant":
                html += (
                    "<div style='margin:8px 0;padding:8px;border-radius:8px;background:#f5f5f5;'>"
                    "<strong>Assistant:</strong>"
                    f"<div style='margin-top:4px'>{escaped}</div>"
                    "</div>"
                )
        html += "</div>"
        self.output_area.value = html

    def display(self):
        input_area = widgets.HBox([self.text_input, self.send_button])
        chat_ui = widgets.VBox([self.output_area, self.image_area, input_area],
                               layout=widgets.Layout(margin="10px"))
        display(chat_ui)
