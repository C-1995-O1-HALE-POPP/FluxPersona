# FluxPersona: Dynamic Personality Regulation for Contextually Adaptive Role-Playing Conversational Agents

This repository implements two main systems of the *FluxPersona* framework:
* **simulate\_dialogue**: Implements the multi-agent dialogue simulation platform described in **§5.1 Simulative Evaluation** of the paper, used for large-scale evaluation of dynamic personality modulation.
* **chat**: Implements the **FluxPersona Chatbox** described in **§6.1 User Evaluation** of the paper, supporting natural interactions between humans and AI agents to assess user perception and preferences on personality modulation strength.
These two systems demonstrate that FluxPersona achieves **context-dependent dynamic personality adjustment** in multi-turn dialogues, improving adaptability and perceived human-likeness while maintaining personality consistency.
---
## 1 Installation & Setup

```bash
git clone https://github.com/C-1995-O1-HALE-POPP/big5tragectory.git
cd big5tragectory

conda create -n big5tragectory python=3.13 -y
conda activate big5tragectory
pip install -r requirements.txt

export OPENAI_API_KEY="your_api_key_here"
export OPENAI_API_URL="your_api_url_here"
```
---
## 2 Demos
### 2.1 `simulate_dialogue.py` (Paper §5.1 Simulative Evaluation)
**Purpose**: Generate offline multi-agent dialogue datasets to compare **static** vs **dynamic** personality behaviors.
**Example**
```bash
python simulate_dialogue.py \
  --num_dialogues 100 \
  --turns_per_dialogue 10 \
  --use_fluxpersona true \
  --preset harmonizer \
  --save_logs true
```
**Outputs**
* `logs/`: dialogue logs and trajectory JSON
* `plots/`: personality trajectories (O/C/E/A/N)

---

### 2.2 `simulate_dialogue_gui.py` (Paper §5.1 Simulative Evaluation)
**Purpose**: GUI-based simulator for **quick parameter tuning and visualizing trajectories**.
Has the same functionality as `simulate_dialogue.py`, but with a graphical interface.
**Example**
```bash
python simulate_dialogue_gui.py
# open http://localhost:7861
```

---

### 2.3 `chat.py` (Paper §6.1 User Evaluation)

**Purpose**: Launch a Gradio Chatbox to interact with the model, observing personality dynamics during real-time conversation.
Supports personality presets, real-time trajectory visualization, and adjustable hyperparameters.

**Example**

```bash
python chat.py
# open http://localhost:7860
```

**Features**

* Select personality style (Shape / Harmonizer / Steady / Custom)
* Real-time OCEAN trajectory plotting
* Export logs, dialogues, and trajectory graphs

---

## 3 Personality Modulation Implementation

Implemented in `state_tracker.py` and `predictor.py`.
`PersonaStateTracker` accepts these hyperparameters to control personality state adjustment.
Five sequential modules: **Trigger → Retrieval → Inference → Modulation → Generation**.
Each user turn triggers this pipeline, combining heuristic evidence and state regression to achieve both **adaptivity + consistency**.

---
### 3.1 HeuristicMotivePredictor

Encapsulates Retrieval → Inference → Modulation. Performs two-stage heuristic inference.

| Param                      | Meaning                          | Recommended |
| -------------------------- | -------------------------------- | ----------- |
| `beta`                     | Motivation scaling factor        | 1.0–2.0     |
| `use_global_factor_weight` | Use global factor weights        | True/False  |
| `eps`                      | Threshold to ignore weak motives | 0.05–0.2    |

Example:

```python
HeuristicMotivePredictor(
	llmClient(), 
	beta=1.3, 
	use_global_factor_weight=True, 
	eps=0.15
)
```
---
### 3.2 PersonaStateTracker

Determines whether a personality update should occur, and if so, computes the new target state using a regression-based modulation algorithm.

| Param                   | Meaning                            | Default | Recommended |
| ----------------------- | ---------------------------------- | ------- | ----------- |
| `target_step`           | Step size per turn                 | 0.10    | 0.08–0.30   |
| `lambda_decay`          | Decay factor λ                     | 0.85    | 0.7–0.9     |
| `alpha_cap`             | Max fusion weight α                | 0.75    | 0.3–1.0     |
| `gate_m_norm`           | Gating threshold (motivation)      | 0.25    | 0.1–0.3     |
| `gate_min_dims`         | Min number of triggered dims       | 1       | 1–3         |
| `cooldown_k`            | Cooldown turns                     | 1       | 1–3         |
| `passive_reg_alpha`     | Passive regression when gate=false | 0.06    | 0.01–0.08   |
| `passive_reg_use_decay` | Time decay for passive regression  | True    | True/False  |
| `eta_base`              | Base elastic pull strength         | 0.15    | 0.0–0.3     |
| `eta_scale`             | Elastic pull vs distance ratio     | 0.50    | 0.0–1.0     |
| `eta_cap`               | Max elastic pull                   | 0.75    | 0.2–0.8     |
| `guard_dist`            | Guardrail threshold                | 0.35    | 0.3–0.8     |
| `guard_alpha_cap`       | Max α under guard                  | 0.25    | 0.05–0.3    |
| `global_drift`          | Global micro-regression per turn   | 0.02    | 0.001–0.03  |

Example:
```python
PersonaStateTracker(
    P0=..., predictor=...,
    target_step=0.10,
    lambda_decay=0.85, alpha_cap=0.75,
    gate_m_norm=0.25, gate_min_dims=1, cooldown_k=1,
    passive_reg_alpha=0.06, passive_reg_use_decay=True,
    eta_base=0.15, eta_scale=0.50, eta_cap=0.75,
    guard_dist=0.35, guard_alpha_cap=0.25,
    global_drift=0.02,
    eps_update=1e-9
)
```
---

## 4 Preset Styles

### 4.1 Shape-Shifter

High sensitivity, rapid adaptation

```python
{
    "dynamic_on": True,
    "beta": 2.0, "eps": 0.15, "use_global": True,
    "target_step": 0.50, "lambda_decay": 0.30, "alpha_cap": 1.0,
    "gate_m_norm": 0.10, "gate_min_dims": 1, "cooldown_k": 2,
    "passive_reg_alpha": 0.002, "passive_reg_use_decay": True, "global_drift": 0.01,
    "eta_base": 0.01, "eta_scale": 0.10, "eta_cap": 0.30,
    "guard_dist": 0.80, "guard_alpha_cap": 0.05,
}
```

* Larger `target_step` (0.50) → bigger single-step shifts
* Lower `eta_base/eta_scale` and higher `guard_dist` → weaker pullback
* Very small `passive_reg_alpha` → minimal passive regression

---

### 4.2 Situational Harmonizer (default)

Balanced sensitivity and stability

```python
{
    "dynamic_on": True,
    "beta": 1.4, "eps": 0.15, "use_global": True,
    "target_step": 0.30, "lambda_decay": 0.30, "alpha_cap": 1.0,
    "gate_m_norm": 0.10, "gate_min_dims": 1, "cooldown_k": 2,
    "passive_reg_alpha": 0.002, "passive_reg_use_decay": True, "global_drift": 0.01,
    "eta_base": 0.15, "eta_scale": 0.50, "eta_cap": 0.75,
    "guard_dist": 0.35, "guard_alpha_cap": 0.25,
}
```

* Moderate `target_step` (0.30) with `alpha_cap=1.0` → flexible but bounded
* Uses standard pullback (`eta_base=0.15, eta_scale=0.50`)
* Default preset in `_apply_preset`

---

### 4.3 Steady Identity

Personality fixed at initial P₀ (no modulation)

```python
{
    "dynamic_on": True,
    "beta": 0.2, "eps": 0.15,
    "target_step": 0.10, "lambda_decay": 10.0, "alpha_cap": 1.0,
    "passive_reg_alpha": 0.1, "global_drift": 0.01,
    "eta_base": 0.5, "eta_scale": 0.8, "eta_cap": 0.75,
    "guard_dist": 0.2, "guard_alpha_cap": 0.5,
    "min_update_threshold": 0.10, "cooldown_k": 5
}
```

* Effectively disables modulation by setting `dynamic_on=False`
* Strong regression (`lambda_decay=10.0`) and large pullback
* Serves as static baseline in experiments

---

## 📖 Citation

TBD
