# chat.py
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

import io
import json
import uuid
import time
from typing import List, Tuple, Generator, Optional, Dict

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from loguru import logger
from PIL import Image

# ===== Project modules =====
from predictor import HeuristicMotivePredictor, llmClient
from state_tracker import PersonaStateTracker
from prompt import (
    SYSTEM_PROMPT,
    generate_persona_system_prompt,
    generate_persona_traits,
    AGENTS,
)

# ==============================
# Logging: JSONL with rotation
# ==============================
logger.add(
    "chat_history.jsonl",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    enqueue=True,
    backtrace=False,
    diagnose=False,
    level="DEBUG",
    format="{message}",  # pure JSON lines
)

def log_json(event: str, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}, ensure_ascii=False))

# ==============================
# Helpers (2-decimal DISPLAY)
# ==============================

def round2(x: float) -> float:
    try:
        return float(f"{float(x):.2f}")
    except Exception:
        return x


def dict_round2(d: Dict[str, float]) -> Dict[str, float]:
    return {k: round2(v) for k, v in d.items()}


def df_round2(traj: list[dict]) -> pd.DataFrame:
    if not traj:
        return pd.DataFrame()
    return pd.DataFrame([{k: round2(v) for k, v in row.items()} for row in traj])

# ==============================
# LLM client (shared)
# ==============================
client = llmClient(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.70")),
    timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
)

# ==============================
# History util
# ==============================
History = List[Tuple[str, str]]

def history_to_messages(hist: History) -> list:
    msgs = []
    for u, a in hist or []:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs

# ==============================
# Streaming chat
# ==============================

def stream_chat(
    message: str,
    history: History,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    session_id: Optional[str],
) -> Generator[str, None, None]:
    if not session_id:
        session_id = str(uuid.uuid4())

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    log_json(
        "round_start",
        session_id=session_id,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
        system_prompt_len=len(system_prompt or ""),
        history_turns=len(history),
    )
    log_json("user_message", session_id=session_id, text=message)

    partial = ""
    try:
        client.change_model(model)
        client.change_temperature(temperature)

        t0 = time.time()
        for i, inc in enumerate(
            client.chat_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens if (max_tokens and max_tokens > 0) else None,
            ),
            start=1,
        ):
            partial += inc
            logger.debug(
                f"[stream_chat] chunk={i}, inc_len={len(inc)}, total_len={len(partial)}, elapsed={time.time()-t0:.2f}s"
            )
            yield partial

        log_json("assistant_message", session_id=session_id, text=partial)
        log_json("round_end", session_id=session_id, tokens=len(partial))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log_json("error", session_id=session_id, where="stream_chat", detail=err)
        yield partial + f"\n\n[Error] {err}"

# ==============================
# Trajectory plotting (Agg -> PNG)
# ==============================
DIMENSIONS = ["O", "C", "E", "A", "N"]

def render_traj_img(traj: list[dict]) -> Image.Image | None:
    if not traj:
        return None
    # Round values for display
    traj2 = [{k: round2(v) for k, v in row.items()} for row in traj]
    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.subplots()
    steps = list(range(len(traj2)))
    for d in DIMENSIONS:
        ax.plot(steps, [row[d] for row in traj2], label=d)
    ax.set_title("Persona Trajectories over Dialogue Turns")
    ax.set_xlabel("Turn (user-only)")
    ax.set_ylabel("Trait value [0..1]")
    ax.set_ylim(0.00, 1.00)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ==============================
# UI text
# ==============================
DESCRIPTION = (
    "# Big5Trajectory Chat Assistant (Persona-ID driven)\n\n"
    "Choose a persona (from prompt.AGENTS). The system uses its Big Five vector as P0.\n"
    "A dynamic system prompt is built via generate_persona_system_prompt(persona_id, Pt),\n"
    "where Pt comes from PersonaStateTracker. Click Initialize Session to apply persona\n"
    "and hyperparameters. Display values round to 2 decimals."
)

DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-3.5-turbo",
]

# ==============================
# Gradio UI (three columns)
# ==============================
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(DESCRIPTION)

    # States
    history_state: gr.State = gr.State([])      # List[Tuple[str, str]]
    session_state: gr.State = gr.State("")
    predictor_state: gr.State = gr.State(None)  # HeuristicMotivePredictor
    tracker_state: gr.State = gr.State(None)    # PersonaStateTracker
    traj_state: gr.State = gr.State([])         # List[Dict[str,float]]
    current_persona_id: gr.State = gr.State("01")
    p0_state: gr.State = gr.State({"O":0.55,"C":0.65,"E":0.35,"A":0.30,"N":0.40})
    dynamic_enabled_state: gr.State = gr.State(True)

    # Session init helper
    def _init_session():
        sid = str(uuid.uuid4())
        log_json("session_init", session_id=sid)
        return sid, f"Session ID: `{sid}` · logs → chat_history.jsonl"

    # Persona choices
    _choices = [f'{p["id"]} - {p.get("name","")}' for p in AGENTS]
    _id_by_label = {f'{p["id"]} - {p.get("name","")}' : p["id"] for p in AGENTS}

    def _persona_label_to_id(label: str) -> str:
        return _id_by_label.get(label, "01")

    def _make_preview(persona_label: str, base_task: str, O: float, C: float, E: float, A: float, N: float):
        pid = _persona_label_to_id(persona_label)
        Pt = {"O":round2(O), "C":round2(C), "E":round2(E), "A":round2(A), "N":round2(N)}
        try:
            preview = generate_persona_system_prompt(
                persona_id=pid,
                Pt=Pt,
                include_base_task_line=True,
                include_big5_details=True,
            )
            if SYSTEM_PROMPT and base_task and base_task != SYSTEM_PROMPT:
                preview = preview.replace(SYSTEM_PROMPT, base_task)
            return preview
        except Exception as e:
            return f"[Dynamic prompt error] {type(e).__name__}: {e}"

    demo.load(_init_session, inputs=None, outputs=[session_state, gr.Markdown()])

    with gr.Row():
        # ============ Left Column ============
        with gr.Column(scale=1):
            with gr.Accordion("Persona Configuration", open=True):
                persona_drop = gr.Dropdown(
                    label="Persona (from prompt.AGENTS)",
                    choices=_choices,
                    value=_choices[0],
                    interactive=True,
                    allow_custom_value=False,
                )
                model_drop = gr.Dropdown(
                    label="Model",
                    choices=DEFAULT_MODELS,
                    value=DEFAULT_MODELS[0],
                    allow_custom_value=True,
                    interactive=True,
                )
                preset_drop = gr.Dropdown(
                    label="Preset",
                    choices=["Shape-Shifter", "Situational Harmonizer", "Steady Identity", "Custom"],
                    value="Situational Harmonizer",
                    interactive=True,
                )
                system_box = gr.Textbox(
                    label="Base Task Line (SYSTEM_PROMPT)",
                    value=SYSTEM_PROMPT,
                    placeholder="Task line used inside persona system prompt",
                    lines=4,
                )
                temperature_slider = gr.Slider(0.00, 2.00, value=0.70, step=0.01, label="temperature")
                max_tokens_box = gr.Number(label="max_tokens (≤0/void = unlimited)", value=None, precision=0)
                dynamic_checkbox = gr.Checkbox(value=True, label="Enable dynamic persona (use tracker updates)")

                with gr.Accordion("OCEAN baseline (P0)", open=True):
                    with gr.Row():
                        O_slider = gr.Slider(0.00, 1.00, value=0.55, step=0.01, label="O - Openness (P0)")
                        C_slider = gr.Slider(0.00, 1.00, value=0.65, step=0.01, label="C - Conscientiousness (P0)")
                        E_slider = gr.Slider(0.00, 1.00, value=0.35, step=0.01, label="E - Extraversion (P0)")
                        A_slider = gr.Slider(0.00, 1.00, value=0.30, step=0.01, label="A - Agreeableness (P0)")
                        N_slider = gr.Slider(0.00, 1.00, value=0.40, step=0.01, label="N - Neuroticism (P0)")
                    dyn_prompt_preview = gr.Textbox(
                        label="Dynamic Persona System Prompt (read-only preview)",
                        value="",
                        lines=10,
                        interactive=False,
                    )

            with gr.Accordion("Predictor (two-stage) hyperparameters", open=False):
                pred_beta = gr.Slider(0.80, 4.00, value=2.00, step=0.10, label="beta (logit sharpness)")
                pred_eps = gr.Slider(0.00, 0.50, value=0.15, step=0.01, label="eps (direction deadzone)")
                pred_use_global = gr.Checkbox(value=True, label="use_global_factor_weight")

            with gr.Accordion("Tracker hyperparameters", open=False):
                target_step = gr.Slider(0.01, 0.50, value=0.30, step=0.01, label="target_step (± per update)")
                lambda_decay = gr.Slider(0.05, 0.95, value=0.30, step=0.01, label="lambda_decay (regression lambda)")
                alpha_cap = gr.Slider(0.05, 1.50, value=1.00, step=0.05, label="alpha_cap (max alpha per dim)")
                gate_m_norm = gr.Slider(0.00, 0.90, value=0.10, step=0.01, label="gate m_norm threshold")
                gate_min_dims = gr.Slider(1, 5, value=1, step=1, label="gate min hit dims")
                cooldown_k = gr.Slider(0, 5, value=1, step=1, label="cooldown_k (min turns between updates)")
                passive_reg_alpha = gr.Slider(0.00, 0.20, value=0.00, step=0.01, label="passive_reg_alpha (Gate=false)")
                passive_reg_use_decay = gr.Checkbox(value=True, label="passive_reg_use_decay")
                global_drift = gr.Slider(0.00, 0.05, value=0.01, step=0.001, label="global_drift (per turn)")

            with gr.Accordion("Elastic pull-back & Guard rails", open=False):
                with gr.Row():
                    eta_base = gr.Slider(0.00, 1.00, value=0.15, step=0.01, label="eta_base (min pull)")
                    eta_scale = gr.Slider(0.00, 1.00, value=0.50, step=0.01, label="eta_scale (pull vs. dist)")
                    eta_cap = gr.Slider(0.00, 1.00, value=0.75, step=0.01, label="eta_cap (pull cap)")
                with gr.Row():
                    guard_dist = gr.Slider(0.00, 1.00, value=0.35, step=0.01, label="guard_dist (trigger)")
                    guard_alpha_cap = gr.Slider(0.00, 1.00, value=0.25, step=0.01, label="guard_alpha_cap (alpha cap)")

            init_btn = gr.Button("Initialize Session (apply persona & hyperparameters)", variant="primary")

            with gr.Accordion("Environment", open=False):
                gr.Markdown(
                    f"""
- OPENAI_BASE_URL: `{os.getenv("OPENAI_BASE_URL", "") or "(not set)"}`
- OPENAI_API_KEY: `{"set" if os.getenv("OPENAI_API_KEY") else "not set"}`
- Logs: `chat_history.jsonl` (JSON Lines; rotation, 7-day retention, compression)
"""
                )

        # ============ Middle Column ============
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=520, type="messages", show_copy_button=True)
            with gr.Row():
                msg_box = gr.Textbox(placeholder="Type your message, press Enter or click Send...", lines=2, scale=8)
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

        # ============ Right Column ============
        with gr.Column(scale=1):
            session_md = gr.Markdown()
            traj_df = gr.Dataframe(label="Trajectory (per user turn)", interactive=False, wrap=True)
            traj_img = gr.Image(label="Trajectory Plot", interactive=False)

    # --- On load ---
    demo.load(_init_session, inputs=None, outputs=[session_state, session_md])

    # Sync persona -> sliders and preview
    def _sync_sliders_with_persona(persona_label: str):
        pid = _persona_label_to_id(persona_label)
        trait = generate_persona_traits(pid)
        trait2 = {k: round2(v) for k, v in trait.items()}
        return (
            pid,
            gr.update(value=trait2["O"]),
            gr.update(value=trait2["C"]),
            gr.update(value=trait2["E"]),
            gr.update(value=trait2["A"]),
            gr.update(value=trait2["N"]),
            _make_preview(persona_label, system_box.value, trait2["O"], trait2["C"], trait2["E"], trait2["A"], trait2["N"]),
            trait2,
        )

    persona_drop.change(
        _sync_sliders_with_persona,
        inputs=[persona_drop],
        outputs=[current_persona_id, O_slider, C_slider, E_slider, A_slider, N_slider, dyn_prompt_preview, p0_state],
    )

    def _update_preview_and_p0(persona_label, base_task, O, C, E, A, N):
        preview = _make_preview(persona_label, base_task, O, C, E, A, N)
        return preview, {"O":round2(O), "C":round2(C), "E":round2(E), "A":round2(A), "N":round2(N)}

    for comp in [system_box, O_slider, C_slider, E_slider, A_slider, N_slider]:
        comp.change(
            _update_preview_and_p0,
            inputs=[persona_drop, system_box, O_slider, C_slider, E_slider, A_slider, N_slider],
            outputs=[dyn_prompt_preview, p0_state],
        )

    dynamic_checkbox.change(lambda v: v, inputs=[dynamic_checkbox], outputs=[dynamic_enabled_state])

    # Preset logic
    def _apply_preset(name: str):
        preset = {
            "dynamic_on": True,
            "beta": 1.40, "eps": 0.15, "use_global": True,
            "target_step": 0.30, "lambda_decay": 1.00, "alpha_cap": 1.00,
            "gate_m_norm": 0.10, "gate_min_dims": 1, "cooldown_k": 2,
            "passive_reg_alpha": 0.00, "passive_reg_use_decay": True, "global_drift": 0.01,
            "eta_base": 0.15, "eta_scale": 0.50, "eta_cap": 0.75,
            "guard_dist": 0.35, "guard_alpha_cap": 0.05,
        }
        if name == "Shape-Shifter":
            preset.update({
                "dynamic_on": True, "beta": 2.00, "target_step": 0.50, "lambda_decay": 0.30,
                "eta_base": 0.01, "eta_scale": 0.10, "eta_cap": 0.30,
                "guard_dist": 0.80, "guard_alpha_cap": 0.25,
            })
        elif name == "Situational Harmonizer":
            pass
        elif name == "Steady Identity":
            preset.update({"dynamic_on": False})
        return (
            gr.update(value=preset["dynamic_on"]),
            gr.update(value=round2(preset["beta"])),
            gr.update(value=round2(preset["eps"])),
            gr.update(value=preset["use_global"]),
            gr.update(value=round2(preset["target_step"])),
            gr.update(value=round2(preset["lambda_decay"])),
            gr.update(value=round2(preset["alpha_cap"])),
            gr.update(value=round2(preset["gate_m_norm"])),
            gr.update(value=int(preset["gate_min_dims"])),
            gr.update(value=int(preset["cooldown_k"])),
            gr.update(value=round2(preset["passive_reg_alpha"])),
            gr.update(value=preset["passive_reg_use_decay"]),
            gr.update(value=round2(preset["global_drift"])),
            gr.update(value=round2(preset["eta_base"])),
            gr.update(value=round2(preset["eta_scale"])),
            gr.update(value=round2(preset["eta_cap"])),
            gr.update(value=round2(preset["guard_dist"])),
            gr.update(value=round2(preset["guard_alpha_cap"])),
        )

    preset_drop.change(
        _apply_preset,
        inputs=[preset_drop],
        outputs=[
            dynamic_checkbox,
            pred_beta, pred_eps, pred_use_global,
            target_step, lambda_decay, alpha_cap,
            gate_m_norm, gate_min_dims, cooldown_k,
            passive_reg_alpha, passive_reg_use_decay, global_drift,
            eta_base, eta_scale, eta_cap, guard_dist, guard_alpha_cap,
        ],
    )

    # Initialize predictor + tracker with hyperparams & persona P0
    def init_session_and_models(
        session_id: str,
        persona_label: str,
        beta: float, eps: float, use_global: bool,
        target_step_v: float, lambda_decay_v: float, alpha_cap_v: float,
        gate_m: float, gate_dims: int, cooldown: int,
        passive_alpha: float, passive_use_decay: bool, drift: float,
        eta_base_v: float, eta_scale_v: float, eta_cap_v: float,
        guard_dist_v: float, guard_alpha_cap_v: float,
        O: float, C: float, E: float, A: float, N: float,
        base_task: str,
        dynamic_on: bool,
    ):
        pid = _persona_label_to_id(persona_label)
        P0 = {"O":float(O), "C":float(C), "E":float(E), "A":float(A), "N":float(N)}

        log_json("init_hparams",
                 session_id=session_id,
                 persona_id=pid,
                 predictor={"beta":beta,"eps":eps,"use_global":use_global},
                 tracker={
                    "target_step":target_step_v,"lambda_decay":lambda_decay_v,"alpha_cap":alpha_cap_v,
                    "gate_m":gate_m,"gate_dims":gate_dims,"cooldown":cooldown,
                    "passive_alpha":passive_alpha,"passive_use_decay":passive_use_decay,"drift":drift,
                    "eta_base":eta_base_v,"eta_scale":eta_scale_v,"eta_cap":eta_cap_v,
                    "guard_dist":guard_dist_v,"guard_alpha_cap":guard_alpha_cap_v,
                 },
                 P0=dict_round2(P0),
                 base_task=base_task,
                 dynamic_on=dynamic_on)

        predictor = HeuristicMotivePredictor(
            llm=client, beta=float(beta), use_global_factor_weight=bool(use_global), eps=float(eps)
        )
        tracker = PersonaStateTracker(
            P0=P0,
            predictor=predictor,
            target_step=float(target_step_v),
            lambda_decay=float(lambda_decay_v),
            alpha_cap=float(alpha_cap_v),
            gate_m_norm=float(gate_m),
            gate_min_dims=int(gate_dims),
            cooldown_k=int(cooldown),
            passive_reg_alpha=float(passive_alpha),
            passive_reg_use_decay=bool(passive_use_decay),
            global_drift=float(drift),
            eta_base=float(eta_base_v),
            eta_scale=float(eta_scale_v),
            eta_cap=float(eta_cap_v),
            guard_dist=float(guard_dist_v),
            guard_alpha_cap=float(guard_alpha_cap_v),
        )

        traj = [P0]
        df = df_round2(traj)
        img = render_traj_img(traj)
        preview = _make_preview(persona_label, base_task, O, C, E, A, N)
        return (
            pid, predictor, tracker, traj, df, gr.update(value=img),
            [], gr.update(value=preview),
            dict_round2(P0),
            dynamic_on,
        )

    init_btn.click(
        init_session_and_models,
        inputs=[
            session_state, persona_drop,
            pred_beta, pred_eps, pred_use_global,
            target_step, lambda_decay, alpha_cap,
            gate_m_norm, gate_min_dims, cooldown_k,
            passive_reg_alpha, passive_reg_use_decay, global_drift,
            eta_base, eta_scale, eta_cap, guard_dist, guard_alpha_cap,
            O_slider, C_slider, E_slider, A_slider, N_slider,
            system_box,
            dynamic_checkbox,
        ],
        outputs=[
            current_persona_id, predictor_state, tracker_state, traj_state, traj_df, traj_img,
            history_state, dyn_prompt_preview,
            p0_state,
            dynamic_enabled_state,
        ],
    )

    # Submit flow
    def user_submit(user_msg: str, history: History):
        if user_msg is None:
            user_msg = ""
        history = (history or []) + [(user_msg, "")]
        messages = history_to_messages(history)
        return gr.update(value=""), history, messages

    def bot_respond(
        history: History,
        persona_label: str,
        base_task: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        session_id: str,
        predictor_obj: HeuristicMotivePredictor,
        tracker_obj: PersonaStateTracker,
        traj: list,
        p0: Dict[str, float],
        dynamic_on: bool,
    ):
        if not history:
            yield [], history, traj, None, None
            return
        if tracker_obj is None:
            msg = "Please click Initialize Session first."
            prior = history[:-1]
            cur = prior + [(history[-1][0], msg)]
            yield history_to_messages(cur), cur, traj, None, None
            return

        user_msg, _ = history[-1]
        prior = history[:-1]

        if dynamic_on:
            tracker_context = []
            for u, a in prior:
                if u: tracker_context.append({"role":"user", "content":u})
                if a: tracker_context.append({"role":"assistant", "content":a})
            tracker_context.append({"role":"user", "content":user_msg})
            _ = tracker_obj.step(tracker_context)
            cur_pt = tracker_obj.get_current_state()
        else:
            cur_pt = getattr(tracker_obj, "P0", p0)

        pid = _persona_label_to_id(persona_label)
        # Build dynamic system prompt with rounded display values
        cur_pt_disp = dict_round2(cur_pt)
        sys_dyn = generate_persona_system_prompt(
            persona_id=pid,
            Pt=cur_pt_disp,
            include_base_task_line=True,
            include_big5_details=True,
        )
        if SYSTEM_PROMPT and base_task and base_task != SYSTEM_PROMPT:
            sys_dyn = sys_dyn.replace(SYSTEM_PROMPT, base_task)

        partial = ""
        for chunk in stream_chat(
            message=user_msg,
            history=prior,
            system_prompt=sys_dyn,
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens else None,
            session_id=session_id,
        ):
            partial = chunk
            cur = prior + [(user_msg, partial)]
            yield history_to_messages(cur), cur, traj, gr.update(), gr.update()

        # Update trajectory (compute uses full precision; display rounds)
        if dynamic_on:
            traj = traj + [cur_pt]
        else:
            traj = traj + [p0]

        df = df_round2(traj)
        final_hist = prior + [(user_msg, partial)]
        img = render_traj_img(traj)
        yield history_to_messages(final_hist), final_hist, traj, df, gr.update(value=img)

    msg_box.submit(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[
            history_state, persona_drop, system_box, model_drop, temperature_slider, max_tokens_box, session_state,
            predictor_state, tracker_state, traj_state,
            p0_state,
            dynamic_enabled_state,
        ],
        outputs=[chatbot, history_state, traj_state, traj_df, traj_img],
    )

    send_btn.click(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[
            history_state, persona_drop, system_box, model_drop, temperature_slider, max_tokens_box, session_state,
            predictor_state, tracker_state, traj_state,
            p0_state,
            dynamic_enabled_state,
        ],
        outputs=[chatbot, history_state, traj_state, traj_df, traj_img],
    )

    def clear_chat():
        return [], [], []

    clear_btn.click(clear_chat, inputs=None, outputs=[history_state, chatbot, msg_box])

if __name__ == "__main__":
    demo.queue(max_size=64).launch(server_name="0.0.0.0", server_port=27861, share=False)
