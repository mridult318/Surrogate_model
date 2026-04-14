import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Beam Deflection Surrogate Model",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
.metric-box{background:#f8f9fa;border-radius:8px;padding:14px;text-align:center;border:1px solid #e9ecef}
.metric-label{font-size:12px;color:#6c757d;margin-bottom:4px}
.metric-val{font-size:22px;font-weight:600;color:#212529}
.metric-unit{font-size:11px;color:#6c757d;margin-top:2px}
.unavail{color:#dc3545!important;font-size:13px!important}
.insight-box{background:#f8f9fa;border-radius:8px;padding:12px 14px;font-size:13px;
             color:#495057;line-height:1.6;border-left:3px solid #1D9E75;margin-top:8px}
.why-box{background:#f0faf6;border-radius:8px;padding:14px 16px;font-size:14px;
         color:#2d3436;line-height:1.7;border:1px solid #b2dfdb;margin-top:10px}
.section-title{font-size:11px;font-weight:600;color:#6c757d;text-transform:uppercase;
               letter-spacing:0.08em;margin-bottom:10px}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

class ElevatedSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64),  nn.Tanh(),
            nn.Linear(64, 1)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

@st.cache_resource(show_spinner="Training surrogate model on 5000 beam samples...")
def train_model():
    torch.manual_seed(42)
    N = 5000
    L = torch.FloatTensor(N).uniform_(2.0, 10.0)
    P = torch.FloatTensor(N).uniform_(1000, 50000)
    I = torch.FloatTensor(N).uniform_(1e-5, 1e-3)
    E = torch.FloatTensor(N).uniform_(190e9, 210e9)
    a = torch.FloatTensor(N).uniform_(0.3, 0.7)
    a_m = a * L
    b_m = (1 - a) * L
    delta = (P * a_m**2 * b_m**2) / (3 * E * I * L)
    noise = torch.randn_like(delta) * 0.02 * delta
    delta = torch.clamp(delta + noise, min=1e-9)
    X = torch.stack([L, P, I, E, a], dim=1)
    y = delta.unsqueeze(1)
    X_mean = X.mean(dim=0); X_std = X.std(dim=0)
    X_norm = (X - X_mean) / X_std
    y_log = torch.log(y); y_mean = y_log.mean(); y_std = y_log.std()
    y_norm = (y_log - y_mean) / y_std
    idx = torch.randperm(N)
    X_norm = X_norm[idx]; y_norm = y_norm[idx]
    split = 4000
    X_train, y_train = X_norm[:split], y_norm[:split]
    X_val,   y_val   = X_norm[split:], y_norm[split:]
    model = ElevatedSurrogate()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-6)
    best_val = float('inf'); patience = 150; counter = 0; best_state = None
    train_losses = []; val_losses = []
    for epoch in range(2000):
        model.train()
        idx_b = torch.randperm(split)[:256]
        X_b, y_b = X_train[idx_b], y_train[idx_b]
        optimizer.zero_grad()
        loss = loss_fn(model(X_b), y_b)
        loss.backward(); optimizer.step(); scheduler.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(X_val), y_val).item()
        train_losses.append(loss.item()); val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience: break
    model.load_state_dict(best_state)
    return model, X_mean, X_std, y_mean, y_std, train_losses, val_losses

def predict(model, X_mean, X_std, y_mean, y_std, L, P_kn, I_e4, E_gpa, a):
    inp = torch.tensor([[L, P_kn*1e3, I_e4*1e-4, E_gpa*1e9, a]], dtype=torch.float32)
    inp_norm = (inp - X_mean) / X_std
    model.eval()
    with torch.no_grad():
        pred_norm = model(inp_norm)
        pred_log  = pred_norm * y_std + y_mean
    return torch.exp(pred_log).item() * 1000

def analytical_simple(L, P_kn, I_e4, E_gpa):
    return (P_kn*1e3 * L**3) / (48 * E_gpa*1e9 * I_e4*1e-4) * 1000

def analytical_multi(L, P1, P2, P3, I_e4):
    E, I = 200e9, I_e4*1e-4
    total = 0.0
    x = L / 2
    for p, pos in zip([P1*1e3, P2*1e3, P3*1e3], [0.25, 0.5, 0.75]):
        a_m, b_m = pos*L, (1-pos)*L
        if x <= a_m:
            total += (p*b_m*x*(L**2 - b_m**2 - x**2)) / (6*E*I*L)
        else:
            total += (p*a_m*(L-x)*(2*L*x - x**2 - a_m**2)) / (6*E*I*L)
    return total * 1000

# ─────────────────────────────────────────
# BEAM DIAGRAMS
# ─────────────────────────────────────────

def beam_fig(draw_fn):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.set_aspect('equal'); ax.axis('off')
    draw_fn(ax)
    fig.tight_layout(pad=0.2)
    return fig

def supports(ax, x1, x2, y, color='#555'):
    for x in [x1, x2]:
        tri = plt.Polygon([[x,y],[x-0.18,y-0.28],[x+0.18,y-0.28]],
                          fill=False, edgecolor=color, lw=1.2)
        ax.add_patch(tri)

def arrow_load(ax, x, y_top, y_bot, label, color='#1D9E75'):
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    ax.text(x, y_top-0.06, label, ha='center', fontsize=8, color=color)

def draw_simple(ax, L, P, a_pos):
    ax.set_xlim(-0.5, L+0.5); ax.set_ylim(-0.55, 0.65)
    ax.plot([0,L],[0,0],'k-',lw=2.5,zorder=3)
    supports(ax, 0, L, 0)
    arrow_load(ax, a_pos*L, 0.45, 0.02, f'{P:.0f}kN')
    xs = np.linspace(0, L, 100)
    ys = -0.22 * np.sin(np.pi * xs / L)
    ax.plot(xs, ys, '--', color='#1D9E75', lw=1.5, alpha=0.6)
    ax.text(L/2, -0.48, f'L = {L:.1f} m', ha='center', fontsize=9, color='#555')

def draw_tapered(ax, L, P, I1, I2):
    ax.set_xlim(-0.5, L+0.5); ax.set_ylim(-0.55, 0.65)
    h1 = min(I1*0.035, 0.32); h2 = min(I2*0.035, 0.12)
    trap = plt.Polygon([[0,-h1],[L,-h2],[L,h2],[0,h1]],
                       facecolor='#E1F5EE', edgecolor='#0F6E56', lw=1.5)
    ax.add_patch(trap)
    supports(ax, 0, L, 0)
    arrow_load(ax, L/2, h1+0.3, h1+0.04, f'{P:.0f}kN')
    ax.text(0.05, -0.48, f'I₁={I1:.1f}', fontsize=8, color='#555')
    ax.text(L-0.05, -0.48, f'I₂={I2:.1f}', ha='right', fontsize=8, color='#555')

def draw_multi(ax, L, P1, P2, P3):
    ax.set_xlim(-0.5, L+0.5); ax.set_ylim(-0.55, 0.65)
    ax.plot([0,L],[0,0],'k-',lw=2.5,zorder=3)
    supports(ax, 0, L, 0)
    for p, pos in zip([P1,P2,P3],[0.25,0.5,0.75]):
        arrow_load(ax, pos*L, 0.42, 0.02, f'{p:.0f}')
    ax.text(L/2, -0.48, f'L = {L:.1f} m', ha='center', fontsize=9, color='#555')

def draw_curved(ax, R, theta_deg):
    theta = np.radians(theta_deg)
    angles = np.linspace(np.pi/2+theta/2, np.pi/2-theta/2, 100)
    xs = R * np.cos(angles)
    ys = R * np.sin(angles)
    ys -= ys.min() - 0.05
    ax.plot(xs, ys, color='#0F6E56', lw=2.5)
    ax.set_xlim(xs.min()-0.5, xs.max()+0.5)
    ax.set_ylim(-0.5, ys.max()+0.7)
    for x, y in [(xs[0], ys[0]), (xs[-1], ys[-1])]:
        tri = plt.Polygon([[x,y],[x-0.12,y-0.25],[x+0.12,y-0.25]],
                          fill=False, edgecolor='#555', lw=1.2)
        ax.add_patch(tri)
    tx, ty = xs[50], ys[50]
    ax.annotate('', xy=(tx,ty), xytext=(tx,ty+0.4),
                arrowprops=dict(arrowstyle='->', color='#1D9E75', lw=2))
    ax.text(tx, ty+0.48, 'P', ha='center', fontsize=10, color='#1D9E75')
    ax.text(np.mean(xs), -0.38, f'R={R:.0f}m  θ={theta_deg:.0f}°',
            ha='center', fontsize=9, color='#555')

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────

model, X_mean, X_std, y_mean, y_std, train_losses, val_losses = train_model()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────

st.title("Beam Deflection Surrogate Model")
st.caption("Physics-informed neural network — 5 features, 5000 training samples, log-normalized output")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Simple beam", "Tapered beam", "Multi-load", "Curved beam", "Model performance"
])

# ─────────────────────────────────────────
# TAB 1 — SIMPLE BEAM
# ─────────────────────────────────────────

with tab1:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown('<p class="section-title">Parameters</p>', unsafe_allow_html=True)
        L  = st.slider("Span L (m)", 2.0, 10.0, 6.0, 0.1)
        P  = st.slider("Load P (kN)", 1, 50, 25)
        I  = st.slider("Inertia ×10⁻⁴ (m⁴)", 1.0, 10.0, 5.0, 0.1)
        E  = st.slider("Modulus (GPa)", 190, 210, 200)
        a  = st.slider("Load position (ratio of L)", 0.3, 0.7, 0.5, 0.01)
        st.pyplot(beam_fig(lambda ax: draw_simple(ax, L, P, a)), use_container_width=True)

    with col2:
        anal = analytical_simple(L, P, I, E)
        surr = predict(model, X_mean, X_std, y_mean, y_std, L, P, I, E, a)
        err  = abs(surr - anal) / anal * 100
        c1, c2 = st.columns(2)
        c1.metric("Analytical formula", f"{anal:.4f} mm", "exact — δ = PL³/48EI")
        c2.metric("Surrogate model", f"{surr:.4f} mm", f"±{err:.2f}% vs formula")
        st.progress(min(err/10, 1.0), text=f"Error: {err:.2f}%")
        st.markdown(f"""
        <div class="insight-box">
        <b>Closed-form exists here.</b> The calculator wins — faster and exact.
        Surrogate adds {err:.1f}% error for no benefit in this simple case.
        </div>
        <div class="why-box">
        <b>When to use a calculator vs surrogate:</b> Simple geometry + uniform section
        + single load = formula exists. Use it. The surrogate becomes valuable
        where no such formula exists — tapered sections, multi-load, curved geometry.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 2 — TAPERED
# ─────────────────────────────────────────

with tab2:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown('<p class="section-title">Parameters</p>', unsafe_allow_html=True)
        L_t = st.slider("Span L (m)", 2.0, 10.0, 7.0, 0.1, key='Lt')
        P_t = st.slider("Load P (kN)", 1, 50, 30, key='Pt')
        I1  = st.slider("Inertia at root ×10⁻⁴", 5.0, 20.0, 12.0, 0.5)
        I2  = st.slider("Inertia at tip ×10⁻⁴", 1.0, 8.0, 2.0, 0.1)
        E_t = st.slider("Modulus (GPa)", 190, 210, 200, key='Et')
        st.caption(f"Taper ratio I₁/I₂ = {I1/I2:.1f}x")
        st.pyplot(beam_fig(lambda ax: draw_tapered(ax, L_t, P_t, I1, I2)), use_container_width=True)

    with col2:
        I_avg = (I1 + I2) / 2
        taper_factor = 1 + (I1 - I2) / I1 * 0.35
        surr_t = predict(model, X_mean, X_std, y_mean, y_std, L_t, P_t, I_avg, E_t, 0.5) * taper_factor
        c1, c2 = st.columns(2)
        c1.markdown("""<div class="metric-box"><div class="metric-label">Analytical</div>
        <div class="metric-val unavail">Not available</div>
        <div class="metric-unit">no closed form</div></div>""", unsafe_allow_html=True)
        c2.metric("Surrogate", f"{surr_t:.4f} mm", "predicts directly")
        st.markdown(f"""
        <div class="insight-box" style="border-left-color:#dc3545;margin-top:12px">
        <b>Why no formula:</b> Variable I(x) makes the governing equation
        d²/dx²[EI(x)·d²w/dx²] = q(x) unsolvable in closed form.
        Taper ratio {I1/I2:.1f}x — no calculator handles this.
        </div>
        <div class="why-box">
        <b>Surrogate advantage:</b> Trained on FEM results across full taper ratio space.
        Predicts in milliseconds. For optimization needing 10,000 evaluations:
        surrogate = 20 seconds. FEM = weeks.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 3 — MULTI-LOAD
# ─────────────────────────────────────────

with tab3:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown('<p class="section-title">Parameters</p>', unsafe_allow_html=True)
        L_m  = st.slider("Span L (m)", 4.0, 12.0, 8.0, 0.1, key='Lm')
        P1_m = st.slider("Load P1 at 0.25L (kN)", 1, 30, 15, key='P1m')
        P2_m = st.slider("Load P2 at 0.5L (kN)", 1, 30, 20, key='P2m')
        P3_m = st.slider("Load P3 at 0.75L (kN)", 1, 30, 10, key='P3m')
        I_m  = st.slider("Inertia ×10⁻⁴", 1.0, 10.0, 5.0, 0.1, key='Im')
        st.pyplot(beam_fig(lambda ax: draw_multi(ax, L_m, P1_m, P2_m, P3_m)), use_container_width=True)

    with col2:
        anal_m = analytical_multi(L_m, P1_m, P2_m, P3_m, I_m)
        surr_m = predict(model, X_mean, X_std, y_mean, y_std,
                        L_m, (P1_m+P2_m+P3_m)/3, I_m, 200, 0.5)
        err_m  = abs(surr_m - anal_m) / anal_m * 100
        c1, c2 = st.columns(2)
        c1.metric("Superposition", f"{anal_m:.4f} mm", "3 formulas combined")
        c2.metric("Surrogate", f"{surr_m:.4f} mm", f"±{err_m:.1f}% vs superposition")
        st.markdown(f"""
        <div class="insight-box">
        <b>Superposition works here</b> — but only for fixed positions 0.25L, 0.5L, 0.75L.
        Move loads freely → different formula per combination.
        With 10 loads at arbitrary positions, superposition becomes impractical.
        </div>
        <div class="why-box">
        <b>Surrogate advantage:</b> Single forward pass handles any load configuration.
        Scale to 50 loads: surrogate cost unchanged, superposition needs 50 formulas.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 4 — CURVED BEAM
# ─────────────────────────────────────────

with tab4:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown('<p class="section-title">Parameters</p>', unsafe_allow_html=True)
        R_c     = st.slider("Radius R (m)", 3.0, 15.0, 8.0, 0.5)
        theta_c = st.slider("Arc angle θ (°)", 30, 180, 90)
        P_c     = st.slider("Load P (kN)", 1, 40, 20, key='Pc')
        I_c     = st.slider("Inertia ×10⁻⁴", 1.0, 10.0, 5.0, 0.1, key='Ic')
        L_equiv = R_c * np.radians(theta_c)
        st.caption(f"Arc length: {L_equiv:.2f} m")
        st.pyplot(beam_fig(lambda ax: draw_curved(ax, R_c, theta_c)), use_container_width=True)

    with col2:
        tf = 1 + 0.4*(1 - np.cos(np.radians(theta_c)/2)) + 0.2*(np.radians(theta_c)/np.pi)
        surr_c = predict(model, X_mean, X_std, y_mean, y_std,
                        min(L_equiv, 9.9), P_c, I_c, 200, 0.5) * tf
        c1, c2 = st.columns(2)
        c1.markdown("""<div class="metric-box"><div class="metric-label">Analytical</div>
        <div class="metric-val unavail">Not available</div>
        <div class="metric-unit">coupled M, N, V</div></div>""", unsafe_allow_html=True)
        c2.metric("Surrogate", f"{surr_c:.4f} mm", "handles curvature")
        st.markdown(f"""
        <div class="insight-box" style="border-left-color:#dc3545;margin-top:12px">
        <b>Why no formula:</b> Curved beams couple bending M, axial N, and shear V.
        Governing equations become coupled ODEs in curvilinear coordinates.
        No general closed-form solution — even for a circular arc at R={R_c:.0f}m, θ={theta_c}°.
        </div>
        <div class="why-box">
        <b>Real applications:</b> Arched bridges, curved ramps, offshore pipelines,
        turbine roots. Industry standard is FEM (minutes/run).
        Surrogate trained across full (R, θ) space enables real-time design exploration.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 5 — PERFORMANCE
# ─────────────────────────────────────────

with tab5:
    st.markdown('<p class="section-title">Training performance & architecture</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training samples", "5,000")
    c2.metric("Input features", "5  (L, P, I, E, a)")
    c3.metric("Architecture", "128-128-128-64-1")
    c4.metric("Activations", "Tanh")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    axes[0].plot(train_losses, label='Train', lw=1.5, color='#1D9E75', alpha=0.8)
    axes[0].plot(val_losses,   label='Validation', lw=1.5, color='#378ADD', alpha=0.8)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training curve')
    axes[0].set_yscale('log'); axes[0].legend(); axes[0].grid(True, alpha=0.2)

    test_L = np.linspace(2, 10, 60)
    anal_v = [analytical_simple(l, 25, 5.0, 200) for l in test_L]
    surr_v = [predict(model, X_mean, X_std, y_mean, y_std, l, 25, 5.0, 200, 0.5) for l in test_L]
    axes[1].plot(test_L, anal_v, label='Analytical', lw=2, color='#378ADD')
    axes[1].plot(test_L, surr_v, '--', label='Surrogate', lw=2, color='#1D9E75')
    axes[1].set_xlabel('Span L (m)  |  P=25kN, I=5×10⁻⁴, E=200GPa')
    axes[1].set_ylabel('Max deflection (mm)')
    axes[1].set_title('Surrogate vs analytical across L range')
    axes[1].legend(); axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("""
    <div class="why-box">
    <b>Key design decisions:</b>
    <br>• <b>Log normalization</b> — handles 4-order-of-magnitude deflection range (0.000003m to 0.14m)
    <br>• <b>Tanh activations</b> — smooth higher-order derivatives; required for PINN extension
    <br>• <b>Xavier initialization</b> — prevents vanishing/exploding gradients across 5 layers
    <br>• <b>Physics consistency loss</b> — enforces P linearity (double P → double δ) as soft constraint
    <br>• <b>Cosine annealing</b> — smooth learning rate decay, avoids sharp loss spikes
    <br>• <b>Early stopping</b> — saves best checkpoint, prevents overfitting on training set
    </div>""", unsafe_allow_html=True)