# -*- coding: utf-8 -*-
"""
FluxPersona Simulator — Chat.py Style Alignment (Dialogue Generation + Streaming) v3 (with collapsible control panels)

Run:
    pip install "gradio==4.*" matplotlib pandas loguru
    python simulate_persona_gui_collapsible.py
"""
from __future__ import annotations

import time, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from simulate_personas import Agent as SPA_Agent
from simulate_personas import PromptedUserAgent as SPA_User
from simulate_personas import DIMENSIONS as SPA_DIMS
from simulate_personas import sample_persona_lines_from_dataset
from state_tracker import PersonaStateTracker
from predictor import HeuristicMotivePredictor
from prompt import SYSTEM_PROMPT as BASE_TASK_LINE, generate_persona_system_prompt
from matplotlib.ticker import FormatStrFormatter
DIMENSIONS = list(SPA_DIMS)

def _bars(pt: Dict[str, float]):
    if not pt: return None
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=160)
    dims = list(DIMENSIONS)
    vals = [float(pt.get(k, 0.0)) for k in dims]
    ax.barh(dims, vals)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for y, v in enumerate(vals):
        ax.text(min(0.98, v)+0.005, y, f"{v:.2f}", va="center")
    fig.tight_layout()
    return fig

def _traj_img(tr):
    if not tr: return None
    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=160)
    ts = list(range(len(tr)))
    for k in DIMENSIONS:
        ax.plot(ts, [row[k] for row in tr], label=k)
    ax.set_ylim(0,1)
    ax.set_xlabel("Turn (user-only)")
    ax.set_ylabel("Trait value")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=len(DIMENSIONS))
    fig.tight_layout()
    return fig

@dataclass
class Session:
    temperature: float = 0.7
    persona_id: str = "01"
    dynamic_on: bool = True
    user_scenario: str = "stable"
    user_emotion_mode: Optional[str] = None
    user_first: str = "Guess what? I finally asked Alex out and he said yes!"
    user_persona_lines_k: int = 0
    agent: SPA_Agent | None = None
    user: SPA_User | None = None
    msgs: List[Dict[str,str]] = field(default_factory=list)
    turn: int = 0
    traj: List[Dict[str,float]] = field(default_factory=list)

    def _patch_system_prompt(self, base_task_line: str):
        orig = self.agent.system_prompt
        def patched(history):
            text = orig(history)
            try:
                return text.replace(BASE_TASK_LINE, base_task_line)
            except Exception:
                return text
        self.agent.system_prompt = patched

    def init(self,
             persona_id: str,
             dynamic_on: bool,
             base_task_line: str,
             pred_beta: float, pred_eps: float, pred_use_global: bool,
             target_step: float, lambda_decay: float, alpha_cap: float,
             gate_m_norm: float, gate_min_dims: int, cooldown_k: int,
             passive_reg_alpha: float, passive_reg_use_decay: bool, global_drift: float,
             scenario: str, emotion_mode: Optional[str], first_line: str, persona_k: int,
            ):
        self.persona_id = persona_id
        self.dynamic_on = dynamic_on
        self.user_scenario = scenario
        self.user_emotion_mode = emotion_mode
        self.user_first = first_line
        self.user_persona_lines_k = int(max(0, persona_k or 0))

        self.agent = SPA_Agent(name="Assistant", dynamic=bool(dynamic_on), persona_id=persona_id)
        lines = []
        if self.user_persona_lines_k > 0:
            try:
                lines = sample_persona_lines_from_dataset(self.user_persona_lines_k)
            except Exception:
                lines = []
        self.user  = SPA_User(name="User", scenario=scenario, total_turns=99,
                              persona_lines=lines, first_message_override=first_line)
        if emotion_mode:
            self.user.emotion_mode = emotion_mode

        if dynamic_on:
            predictor = HeuristicMotivePredictor(
                self.agent.predictor.llm if getattr(self.agent, "predictor", None) else None,
                beta=float(pred_beta), use_global_factor_weight=bool(pred_use_global), eps=float(pred_eps)
            )
            P0 = dict(self.agent.P0)
            try:
                self.agent.state_tracker = PersonaStateTracker(
                    P0=P0,
                    predictor=predictor,
                    target_step=float(target_step),
                    lambda_decay=float(lambda_decay),
                    alpha_cap=float(alpha_cap),
                    gate_m_norm=float(gate_m_norm),
                    gate_min_dims=int(gate_min_dims),
                    cooldown_k=int(cooldown_k),
                    passive_reg_alpha=float(passive_reg_alpha),
                    passive_reg_use_decay=bool(passive_reg_use_decay),
                    global_drift=float(global_drift),
                )
            except TypeError:
                self.agent.state_tracker = PersonaStateTracker(P0=P0, predictor=predictor)

        self._patch_system_prompt(base_task_line)

        self.msgs = []
        first = self.user.respond([], temperature=self.temperature)
        self.msgs.append({"role":"user", "content": first})
        self.turn = 1
        self.traj = [self.agent.get_current_state()]

    def step_stream(self, chunk_tokens: int = 6, delay: float = 0.04):
        assert self.agent and self.user
        if self.turn >= 1:
            u = self.user.respond(self.msgs.copy(), temperature=self.temperature)
            self.msgs.append({"role":"user", "content": u})
            self.turn += 1
            yield self.msgs
        if self.dynamic_on:
            self.agent.update_persona(self.msgs.copy())
        cur_pt = self.agent.get_current_state()
        full = self.agent.respond(self.msgs.copy(), temperature=self.temperature)
        self.msgs.append({"role":"assistant", "content": ""})
        yield self.msgs
        buf = []
        words = full.split()
        for i,w in enumerate(words,1):
            buf.append(w)
            if i % chunk_tokens == 0 or i==len(words):
                self.msgs[-1]["content"] = " ".join(buf)
                yield self.msgs
                time.sleep(delay)
        self.msgs[-1]["content"] = " ".join(words)
        self.traj.append({k: float(cur_pt[k]) for k in DIMENSIONS})
        yield self.msgs

STATE = Session()

with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate")) as demo:
    gr.Markdown("## FluxPersona — Automatic Dialogue Generation + Streaming\n")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Control Panel (collapsible)", open=True):
                persona = gr.Dropdown([f"{i:0>2}" for i in range(1,9)], value="01", label="Assistant Persona (ID from prompt.AGENTS)")
                base_task = gr.Textbox(label="Base Task Line (SYSTEM_PROMPT)", value=BASE_TASK_LINE, lines=3)
                dynamic_on = gr.Checkbox(value=True, label="Enable dynamic persona (use tracker updates)")
                temperature = gr.Slider(0.0, 2.0, value=STATE.temperature, step=0.05, label="temperature")
                preset = gr.Dropdown(label="Preset", choices=["Shape-Shifter", "Situational Harmonizer", "Steady Identity", "Custom"], value="Situational Harmonizer")

                with gr.Accordion("Predictor", open=True):
                    gr.Markdown("**Predictor**")
                    pred_beta = gr.Slider(0.8, 4.0, value=2.0, step=0.1, label="beta")
                    pred_eps = gr.Slider(0.0, 0.5, value=0.15, step=0.01, label="eps")
                    pred_use_global = gr.Checkbox(value=True, label="use_global_factor_weight")

                with gr.Accordion("Tracker", open=True):
                    gr.Markdown("**Tracker**")
                    target_step = gr.Slider(0.01, 0.50, value=0.30, step=0.01, label="target_step")
                    lambda_decay = gr.Slider(0.05, 0.95, value=0.80, step=0.01, label="lambda_decay")
                    alpha_cap = gr.Slider(0.05, 1.50, value=1.0, step=0.05, label="alpha_cap")
                    gate_m_norm = gr.Slider(0.0, 0.9, value=0.10, step=0.01, label="gate_m_norm")
                    gate_min_dims = gr.Slider(1, 5, value=1, step=1, label="gate_min_dims")
                    cooldown_k = gr.Slider(0, 5, value=1, step=1, label="cooldown_k")
                    passive_reg_alpha = gr.Slider(0.0, 0.2, value=0.002, step=0.001, label="passive_reg_alpha")
                    passive_reg_use_decay = gr.Checkbox(value=True, label="passive_reg_use_decay")
                    global_drift = gr.Slider(0.0, 0.05, value=0.001, step=0.001, label="global_drift")

                with gr.Accordion("User Persona / Scenario", open=True):
                    gr.Markdown("**User Persona / Scenario**")
                    scenario = gr.Dropdown(["stable", "shifting"], value="stable", label="scenario (topic switch policy)")
                    emotion = gr.Dropdown([None, "sadness", "anxiety", "joy", "neutral"], value=None, label="persistent emotion mode")
                    first_line = gr.Textbox(value=STATE.user_first, label="first user line", lines=2)
                    k_lines = gr.Slider(0, 6, value=0, step=1, label="sample persona lines (HF)")

            init_btn = gr.Button("Initialize / Reset", variant="primary")
            with gr.Row():
                btn_step = gr.Button("Step (stream)")
                steps = gr.Number(value=3, precision=0, label="Run N")
                btn_run = gr.Button("Run")

        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Dialogue", height=560, type="messages", bubble_full_width=False)

        with gr.Column(scale=1):
            traj_df = gr.Dataframe(pd.DataFrame(), label="Trajectory (per user turn)", interactive=False)
            bars = gr.Plot(label="Current persona (bars)")
            line = gr.Plot(label="Persona Trajectories")

    def apply_preset(name: str):
        preset = {
            "dynamic_on": True,
            "beta": 2.0, "eps": 0.15, "use_global": True,
            "target_step": 0.30, "lambda_decay": 0.30, "alpha_cap": 1.0,
            "gate_m_norm": 0.10, "gate_min_dims": 1, "cooldown_k": 1,
            "passive_reg_alpha": 0.002, "passive_reg_use_decay": True, "global_drift": 0.001,
        }
        if name == "Shape-Shifter":
            preset.update({"dynamic_on": True, "lambda_decay": 0.20})
        elif name == "Steady Identity":
            preset.update({"dynamic_on": False, "lambda_decay": 0.90, "target_step": 0.10})
        return (
            gr.update(value=preset["dynamic_on"]),
            gr.update(value=preset["beta"]), gr.update(value=preset["eps"]), gr.update(value=preset["use_global"]),
            gr.update(value=preset["target_step"]), gr.update(value=preset["lambda_decay"]), gr.update(value=preset["alpha_cap"]),
            gr.update(value=preset["gate_m_norm"]), gr.update(value=preset["gate_min_dims"]), gr.update(value=preset["cooldown_k"]),
            gr.update(value=preset["passive_reg_alpha"]), gr.update(value=preset["passive_reg_use_decay"]), gr.update(value=preset["global_drift"]),
        )

    preset.change(
        apply_preset,
        inputs=[preset],
        outputs=[dynamic_on, pred_beta, pred_eps, pred_use_global,
                 target_step, lambda_decay, alpha_cap,
                 gate_m_norm, gate_min_dims, cooldown_k,
                 passive_reg_alpha, passive_reg_use_decay, global_drift]
    )

    def ui_init(pid, dyn, base_task_line, temp_v,
                b, e, ug, ts, ld, ac, gm, gmd, ck, pra, prd, gd,
                scen, emo, first, k):
        STATE.temperature = float(temp_v)
        STATE.init(pid, bool(dyn), base_task_line,
                   b, e, bool(ug), ts, ld, ac, gm, gmd, ck, pra, bool(prd), gd,
                   scen, emo, first, int(k))
        bars_fig = _bars(STATE.traj[0]) if STATE.traj else None
        line_fig = _traj_img(STATE.traj) if STATE.traj else None
        return STATE.msgs, pd.DataFrame(STATE.traj).round(2), gr.update(value=bars_fig), gr.update(value=line_fig)

    init_btn.click(
        ui_init,
        inputs=[persona, dynamic_on, base_task, temperature,
                pred_beta, pred_eps, pred_use_global,
                target_step, lambda_decay, alpha_cap, gate_m_norm, gate_min_dims, cooldown_k,
                passive_reg_alpha, passive_reg_use_decay, global_drift,
                scenario, emotion, first_line, k_lines],
        outputs=[chat, traj_df, bars, line]
    )

    def _step_once():
        for out in STATE.step_stream():
            bars_fig = _bars(STATE.agent.get_current_state()) if STATE.agent else None
            line_fig = _traj_img(STATE.traj)
            yield out, pd.DataFrame(STATE.traj).round(2), gr.update(value=bars_fig), gr.update(value=line_fig)

    btn_step.click(_step_once, inputs=None, outputs=[chat, traj_df, bars, line])

    def _run_n(n:int):
        n = int(max(1, n or 1))
        for _ in range(n):
            for out in STATE.step_stream():
                bars_fig = _bars(STATE.agent.get_current_state()) if STATE.agent else None
                line_fig = _traj_img(STATE.traj)
                yield out, pd.DataFrame(STATE.traj).round(2), gr.update(value=bars_fig), gr.update(value=line_fig)

    btn_run.click(_run_n, inputs=[steps], outputs=[chat, traj_df, bars, line])

if __name__ == "__main__":
    demo.queue().launch()