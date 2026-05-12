
import gradio as gr
import base64
from model import RiskModel
from explainer import SHAPExplainer
from config import RISK_TIERS, COST_BANDS

print("Training model...")
risk_model = RiskModel()
X_train    = risk_model.train()
explainer  = SHAPExplainer(risk_model.model, X_train)
print("✅ Ready.")

# ── Logo → base64 ─────────────────────────────────────────────────────────────
with open("logo_transparent.png", "rb") as f:
    LOGO_B64 = base64.b64encode(f.read()).decode()
LOGO_SRC = f"data:image/png;base64,{LOGO_B64}"

# ── Validation ────────────────────────────────────────────────────────────────
def validate_inputs(gender, cancer_type, stage, metastasis, bmi, age, performance_status):
    errors   = []
    warnings = []

    if gender == "Female" and cancer_type == "Prostate":
        errors.append("Prostate cancer cannot occur in females. Please verify the cancer type.")

    if gender == "Male" and cancer_type == "Breast":
        warnings.append(
            "Male breast cancer is rare (~1% of cases) but significantly more aggressive. "
            "Risk score has been adjusted upward accordingly."
        )

    if stage in [1, 2] and metastasis:
        warnings.append(
            "Metastasis at Stage 1–2 is clinically uncommon. "
            "Please confirm diagnosis — this combination affects the risk calculation."
        )

    if bmi < 16.0:
        warnings.append(
            "BMI below 16 indicates severe undernutrition (cachexia). "
            "This is associated with very advanced disease and may underestimate true risk."
        )

    if age < 30 and performance_status >= 3:
        warnings.append(
            "High functional impairment (PS ≥ 3) in a patient under 30 is unusual. "
            "Please verify the performance status score."
        )

    return errors, warnings

# ── Main function ─────────────────────────────────────────────────────────────
def analyze(age, stage, bmi, prior_surgeries, comorbidities,
            performance_status, smoking, diabetes, family_history,
            metastasis, cancer_type, treatment_type, gender):
    try:
        errors, warnings = validate_inputs(
            gender, cancer_type, int(stage), metastasis,
            bmi, age, int(performance_status)
        )

        if errors:
            error_html = "".join([
                f"""
                <div style='background:#2d1515;border:1px solid #ef4444;
                    border-radius:12px;padding:16px 20px;margin:8px 0;
                    display:flex;gap:14px;align-items:flex-start'>
                  <span style='font-size:22px'>🚫</span>
                  <div>
                    <div style='color:#ef4444;font-weight:700;font-size:14px;
                                margin-bottom:4px'>Input Error</div>
                    <div style='color:#fca5a5;font-size:13px;line-height:1.6'>{e}</div>
                  </div>
                </div>"""
                for e in errors
            ])
            return (
                f"<div style='animation:fadeIn 0.4s ease'>{error_html}</div>",
                "", None, ""
            )

        warnings_html = ""
        if warnings:
            warnings_html = "".join([
                f"""
                <div style='background:#1f1a0a;border:1px solid #f7941d55;
                    border-radius:12px;padding:14px 18px;margin:6px 0;
                    display:flex;gap:12px;align-items:flex-start'>
                  <span style='font-size:18px'>⚠️</span>
                  <div style='color:#fcd34d;font-size:13px;line-height:1.6'>{w}</div>
                </div>"""
                for w in warnings
            ])

        raw = dict(
            age=age, stage=int(stage), bmi=bmi,
            prior_surgeries=int(prior_surgeries),
            comorbidities=int(comorbidities),
            performance_status=int(performance_status),
            smoking=smoking, diabetes=diabetes,
            family_history=family_history, metastasis=metastasis,
            cancer_type=cancer_type,
            treatment_type=treatment_type.replace(" Care", ""),
            gender=gender
        )

        tier, proba = risk_model.predict(raw)
        tier_info   = RISK_TIERS[tier]
        cost_band   = COST_BANDS[tier]
        X_enc       = risk_model.encode_input(raw)
        shap_vals   = explainer.explain(X_enc)
        chart_path  = explainer.plot_shap(shap_vals, X_enc, tier)
        factors     = explainer.top_factors(shap_vals, X_enc, tier)

        messages = {
            0: ("Low risk profile detected.",
                "Standard policy terms are likely applicable."),
            1: ("Moderate risk profile detected.",
                "Some conditions may require adjusted premiums."),
            2: ("High risk profile detected.",
                "Careful review of coverage terms is recommended."),
            3: ("Critical risk profile detected.",
                "Specialist review strongly advised before issuing a policy."),
        }
        msg1, msg2 = messages[tier]

        badge = f"""
        <div style='animation:fadeIn 0.5s ease'>
          {warnings_html}
          <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                      border:2px solid {tier_info['color']};border-radius:20px;
                      padding:32px 24px;text-align:center;
                      margin-top:{"12px" if warnings_html else "0"};
                      box-shadow:0 0 40px {tier_info['color']}44'>
            <div style='font-size:56px;margin-bottom:10px'>{tier_info['badge']}</div>
            <div style='font-size:28px;font-weight:800;color:{tier_info['color']};
                        letter-spacing:2px;margin-bottom:10px'>
              {tier_info['label'].upper()}
            </div>
            <div style='height:2px;background:linear-gradient(90deg,transparent,
                        {tier_info['color']},transparent);margin:12px 0'></div>
            <div style='color:#e2e8f0;font-size:15px;margin-bottom:4px'>{msg1}</div>
            <div style='color:#94a3b8;font-size:13px;margin-bottom:20px'>{msg2}</div>
            <div style='background:#0a0f1e;border-radius:12px;padding:14px'>
              <div style='color:#64748b;font-size:11px;text-transform:uppercase;
                          letter-spacing:3px'>Estimated Annual Cost</div>
              <div style='color:#f7941d;font-size:22px;font-weight:800;
                          margin-top:6px'>{cost_band}</div>
            </div>
          </div>
        </div>"""

        conf_html = "<div style='margin-top:4px'>"
        for i, (p, info) in enumerate(zip(proba, RISK_TIERS.values())):
            w_pct = int(p * 100)
            conf_html += f"""
            <div style='margin:10px 0;animation:slideIn 0.4s ease {i*0.1}s both'>
              <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
                <span style='color:#e2e8f0;font-size:14px'>{info['badge']} {info['label']}</span>
                <span style='color:white;font-weight:700;font-size:14px'>{p*100:.0f}%</span>
              </div>
              <div style='background:#1e293b;border-radius:8px;height:14px;overflow:hidden'>
                <div style='background:linear-gradient(90deg,{info['color']}aa,{info['color']});
                            width:{w_pct}%;height:100%;border-radius:8px'></div>
              </div>
            </div>"""
        conf_html += "</div>"

        factors_html = "<div>"
        for idx, f in enumerate(factors):
            arrow = "▲" if f["direction"] == "increases" else "▼"
            color = "#f87171" if f["direction"] == "increases" else "#4ade80"
            label = "raises risk" if f["direction"] == "increases" else "lowers risk"
            factors_html += f"""
            <div style='display:flex;align-items:center;gap:12px;
                        padding:12px 16px;margin:6px 0;
                        background:linear-gradient(135deg,#1e293b,#1a2540);
                        border-radius:12px;border-left:4px solid {color};
                        animation:slideIn 0.4s ease {idx*0.1}s both'>
              <span style='color:{color};font-size:22px;font-weight:900'>{arrow}</span>
              <span style='color:#e2e8f0;font-size:14px;flex:1;font-weight:500'>
                {f['feature']}</span>
              <span style='color:{color};font-size:11px;font-weight:700;
                           background:{color}22;padding:3px 10px;
                           border-radius:20px;border:1px solid {color}44'>
                {label}</span>
            </div>"""
        factors_html += "</div>"

        return badge, conf_html, chart_path, factors_html

    except Exception as e:
        err = f"""
        <div style='color:#f87171;padding:16px 20px;background:#2d1515;
                    border-radius:12px;border:1px solid #ef444444'>
          ❌ Unexpected error: {str(e)}
        </div>"""
        return err, "", None, ""


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
@keyframes fadeIn {
  from { opacity:0; transform:translateY(12px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes slideIn {
  from { opacity:0; transform:translateX(-15px); }
  to   { opacity:1; transform:translateX(0); }
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 0 0 #f7941d44; }
  50%      { box-shadow: 0 0 0 10px #f7941d00; }
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0f1e !important;
    color: #e2e8f0 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

.gr-panel, .gr-box, .gradio-container .block {
    background: #111827 !important;
    border: 1px solid #1f2d45 !important;
    border-radius: 16px !important;
}

/* labels */
label, .gr-form label {
    color: #e2e8f0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* sliders */
input[type=range] { accent-color: #f7941d !important; }

/* checkboxes */
input[type=checkbox] {
    accent-color: #f7941d !important;
    width:17px !important;
    height:17px !important;
}

/* ── Radio buttons ── */
.gr-radio-group, .radio-group {
    display: flex !important;
    gap: 12px !important;
    flex-direction: row !important;
}

.gr-radio-group label, .radio-group label {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    padding: 12px 16px !important;
    background: #1e293b !important;
    border: 2px solid #334155 !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.gr-radio-group label:hover, .radio-group label:hover {
    border-color: #f7941d !important;
    color: #f7941d !important;
    background: #f7941d11 !important;
}

.gr-radio-group input[type=radio]:checked + span,
.radio-group input[type=radio]:checked + span {
    color: #f7941d !important;
}

.gr-radio-group label:has(input:checked),
.radio-group label:has(input:checked) {
    border-color: #f7941d !important;
    background: linear-gradient(135deg,#f7941d22,#f7941d11) !important;
    color: #f7941d !important;
    box-shadow: 0 0 12px #f7941d33 !important;
}

input[type=radio] {
    accent-color: #f7941d !important;
    width: 16px !important;
    height: 16px !important;
}

/* ── Dropdowns ── */
.gr-dropdown, select,
.gradio-container .wrap,
.gradio-container .wrap-inner,
.gradio-container input.svelte-1ed2p3z,
.gradio-container .svelte-1ed2p3z {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* dropdown selected value text */
.gradio-container .multiselect,
.gradio-container ul.options,
.gradio-container .item {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}

/* dropdown arrow + input text */
.gradio-container input {
    color: #e2e8f0 !important;
    background: transparent !important;
}

/* dropdown open list */
.gradio-container ul.options {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

.gradio-container ul.options li {
    color: #e2e8f0 !important;
    padding: 10px 14px !important;
}

.gradio-container ul.options li:hover {
    background: #f7941d22 !important;
    color: #f7941d !important;
}

/* ── Analyze button ── */
.gr-button-primary {
    background: linear-gradient(135deg, #f7941d, #e07b0a) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border-radius: 12px !important;
    padding: 16px !important;
    width: 100% !important;
    animation: pulse 2s infinite !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.gr-button-primary:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 10px 35px #f7941d77 !important;
    animation: none !important;
}

/* section titles */
.section-title {
    color: #f7941d !important;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 18px 0 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1f2d45;
}

footer { display: none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:

    # Header
    gr.HTML(f"""
    <div style='display:flex;align-items:center;gap:18px;
                padding:20px 28px;
                background:linear-gradient(135deg,#111827,#1a2035);
                border-bottom:3px solid #f7941d;
                border-radius:16px;margin-bottom:20px;
                animation:fadeIn 0.6s ease'>
      <img src='{LOGO_SRC}'
           style='height:50px;width:auto;object-fit:contain;
                  mix-blend-mode:screen;
                  filter:brightness(1.1) drop-shadow(0 0 8px #f7941d88)'>
      <div style='flex:1'>
        <div style='font-size:22px;font-weight:800;color:white;letter-spacing:0.5px'>
          AMC <span style='color:#f7941d'>Risk</span> Intelligence Engine
        </div>
        <div style='font-size:12px;color:#64748b;margin-top:3px'>
          Aseel Medical Care · Red Sea · Health Insurance Risk Scoring
        </div>
      </div>
      <div style='background:linear-gradient(135deg,#f7941d22,#f7941d11);
                  border:1px solid #f7941d55;border-radius:10px;
                  padding:10px 16px;text-align:center'>
        <div style='color:#f7941d;font-size:11px;font-weight:800;letter-spacing:1px'>
          ⚡ POWERED BY XAI</div>
        <div style='color:#64748b;font-size:10px;margin-top:2px'>Explainable AI</div>
      </div>
    </div>
    """)

    # Steps
    gr.HTML("""
    <div style='display:flex;gap:12px;margin-bottom:20px;animation:fadeIn 0.8s ease'>
      <div style='flex:1;background:linear-gradient(135deg,#1e293b,#1a2540);
                  border-radius:14px;padding:18px;text-align:center;
                  border-top:3px solid #f7941d;transition:transform 0.2s;cursor:default'
           onmouseover="this.style.transform='translateY(-4px)'"
           onmouseout="this.style.transform='translateY(0)'">
        <div style='font-size:28px;margin-bottom:8px'>📋</div>
        <div style='color:#f7941d;font-size:11px;font-weight:700;letter-spacing:2px;
                    margin-bottom:6px'>STEP 1</div>
        <div style='color:white;font-weight:600;font-size:14px'>Enter Patient Details</div>
        <div style='color:#64748b;font-size:12px;margin-top:4px'>
          Age, cancer type, stage & conditions</div>
      </div>
      <div style='display:flex;align-items:center;color:#f7941d;font-size:22px'>→</div>
      <div style='flex:1;background:linear-gradient(135deg,#1e293b,#1a2540);
                  border-radius:14px;padding:18px;text-align:center;
                  border-top:3px solid #f7941d;transition:transform 0.2s;cursor:default'
           onmouseover="this.style.transform='translateY(-4px)'"
           onmouseout="this.style.transform='translateY(0)'">
        <div style='font-size:28px;margin-bottom:8px'>🤖</div>
        <div style='color:#f7941d;font-size:11px;font-weight:700;letter-spacing:2px;
                    margin-bottom:6px'>STEP 2</div>
        <div style='color:white;font-weight:600;font-size:14px'>Click Analyze</div>
        <div style='color:#64748b;font-size:12px;margin-top:4px'>
          AI processes data in seconds</div>
      </div>
      <div style='display:flex;align-items:center;color:#f7941d;font-size:22px'>→</div>
      <div style='flex:1;background:linear-gradient(135deg,#1e293b,#1a2540);
                  border-radius:14px;padding:18px;text-align:center;
                  border-top:3px solid #f7941d;transition:transform 0.2s;cursor:default'
           onmouseover="this.style.transform='translateY(-4px)'"
           onmouseout="this.style.transform='translateY(0)'">
        <div style='font-size:28px;margin-bottom:8px'>📊</div>
        <div style='color:#f7941d;font-size:11px;font-weight:700;letter-spacing:2px;
                    margin-bottom:6px'>STEP 3</div>
        <div style='color:white;font-weight:600;font-size:14px'>Get Your Report</div>
        <div style='color:#64748b;font-size:12px;margin-top:4px'>
          Risk level, cost & full explanation</div>
      </div>
    </div>
    """)

    # Main layout
    with gr.Row(equal_height=False):

        # ── LEFT — Inputs ─────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            gr.HTML("<div class='section-title'>🧾 Patient Information</div>")

            gender = gr.Dropdown(
                ["Female", "Male"],
                value="Female",
                label="Patient Gender"
            )

            cancer_type = gr.Dropdown(
                ["Breast","Lung","Colon","Liver",
                 "Leukemia","Lymphoma","Prostate","Thyroid"],
                value="Breast",
                label="Type of Cancer"
            )

            treatment_type = gr.Dropdown(
                ["Surgery","Chemotherapy","Radiation",
                 "Immunotherapy","Palliative Care"],
                value="Surgery",
                label="Current or Planned Treatment"
            )

            stage = gr.Slider(1, 4, value=2, step=1,
                              label="Cancer Stage  ( 1 = Early  →  4 = Advanced )")
            age   = gr.Slider(18, 90, value=45, step=1,
                              label="Patient Age")
            bmi   = gr.Slider(15.0, 45.0, value=25.0, step=0.1,
                              label="BMI  ( Body Mass Index )")

            gr.HTML("<div class='section-title'>🏥 Medical History</div>")

            perf = gr.Slider(0, 4, value=1, step=1,
                             label="Daily Activity Level  ( 0 = Fully active  →  4 = Bedridden )")
            prior_surg = gr.Slider(0, 5, value=0, step=1,
                                   label="Number of Previous Surgeries")
            comorbid   = gr.Slider(0, 6, value=1, step=1,
                                   label="Other Health Conditions  ( heart disease, kidney issues… )")

            gr.HTML("<div class='section-title'>⚠️ Risk Factors</div>")

            with gr.Row():
                smoking    = gr.Checkbox(label="🚬  Smoker / Ex-smoker")
                diabetes   = gr.Checkbox(label="🩸  Has Diabetes")
            with gr.Row():
                fam_hist   = gr.Checkbox(label="🧬  Family History of Cancer")
                metastasis = gr.Checkbox(label="⚠️  Cancer Has Spread")

            gr.HTML("<div style='margin-top:20px'></div>")
            btn = gr.Button("🔍  Analyze Risk Profile", variant="primary", size="lg")

        # ── RIGHT — Outputs ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            gr.HTML("<div class='section-title'>📊 Risk Assessment Result</div>")
            badge_out = gr.HTML("""
            <div style='background:linear-gradient(135deg,#111827,#1a2035);
                        border:2px dashed #2d3748;border-radius:20px;
                        padding:56px 24px;text-align:center;color:#475569'>
              <div style='font-size:40px;margin-bottom:12px'>🔬</div>
              <div style='font-size:14px'>
                Fill in the patient data on the left<br>
                then click <strong style='color:#f7941d'>Analyze Risk Profile</strong>
              </div>
            </div>""")

            gr.HTML("<div class='section-title'>📈 Confidence per Risk Level</div>")
            conf_out = gr.HTML()

            gr.HTML("<div class='section-title'>🔬 What's Driving This Result?</div>")
            chart_out = gr.Image(show_label=False, container=False)

            gr.HTML("<div class='section-title'>🧩 Key Factors Explained</div>")
            factors_out = gr.HTML()

    btn.click(
        fn=analyze,
        inputs=[age, stage, bmi, prior_surg, comorbid, perf,
                smoking, diabetes, fam_hist, metastasis,
                cancer_type, treatment_type, gender],
        outputs=[badge_out, conf_out, chart_out, factors_out]
    )

    gr.HTML("""
    <div style='margin-top:24px;padding:16px 20px;
                background:linear-gradient(135deg,#111827,#1a2035);
                border-radius:12px;border:1px solid #1f2d45;
                display:flex;align-items:flex-start;gap:14px;
                animation:fadeIn 1s ease'>
      <span style='font-size:24px'>⚕️</span>
      <span style='color:#475569;font-size:12px;line-height:1.7'>
        <strong style='color:#64748b'>Disclaimer:</strong>
        This tool is designed to assist insurance underwriting teams with risk assessment only.
        It does not provide medical diagnosis or treatment advice.
        All outputs must be reviewed by a qualified professional before any financial or
        clinical decision is made. — <em>AMC Red Sea · AI Research Division</em>
      </span>
    </div>
    """)
import os

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=10000,
        share=True,      # أهم سطر
        debug=False,
    )
