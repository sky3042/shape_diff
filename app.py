import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import streamlit as st

# poisson_binomial_pmf 関数
def poisson_binomial_pmf(p_list):
    n = len(p_list)
    probs = np.zeros(n + 1)
    probs[0] = 1.0
    for p in p_list:
        probs[1:] = probs[1:] * (1 - p) + probs[:-1] * p
        probs[0] *= (1 - p)
    return probs

# 分散値を直接指定して確率を生成
def create_probs_with_target_variance(n, target_var, mean=0.5):
    max_var = mean * (1 - mean)
    if target_var > max_var:
        target_var = max_var
        print(f"Warning: Target variance too high. Clamping to {target_var:.3f}")

    d = np.sqrt(target_var)
    p_a = max(0, mean - d)
    p_b = min(1, mean + d)

    probs = np.full(n, p_a)
    num_b = n // 2
    probs[:num_b] = p_b

    np.random.shuffle(probs)
    return probs

# グラフ描画
def plot_comparison_with_target_var(n, target_var, ax, mean=0.5):
    p_i = create_probs_with_target_variance(n, target_var, mean)
    p_bar = np.mean(p_i)
    actual_var = np.var(p_i)
    diff_variance_theory = n * actual_var

    k = np.arange(n + 1)
    pmf_pb = poisson_binomial_pmf(p_i)
    pmf_b = binom.pmf(k, n, p_bar)

    v_pb = np.sum(p_i * (1 - p_i))
    v_b = n * p_bar * (1 - p_bar)

    ax.plot(k, pmf_pb, color='blue', linewidth=2, alpha=0.8, label=f'Poisson Binomial ($V_{{PB}}={v_pb:.2f}$)')
    ax.plot(k, pmf_b, color='red', linewidth=2, alpha=0.8, linestyle='--', label=f'Binomial ($V_B={v_b:.2f}$)')

    title_text = (
        f'n={n}, target Var(p_i)={target_var:.3f}, actual Var(p_i)={actual_var:.3f}\n'
        f'$V_B - V_{{PB}} = n \\cdot Var(p_i) \\approx {diff_variance_theory:.2f}$'
    )
    ax.set_title(title_text, fontsize=12)
    ax.set_xlabel("Number of successes (k)")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

# Streamlit アプリ本体
st.title("Poisson Binomial vs Binomial (Target Var(p_i))")

n = st.slider('Number of trials (n)', min_value=10, max_value=500, value=100, step=10)
target_var = st.slider('Target Var(p_i)', min_value=0.0, max_value=0.25, value=0.02, step=0.005)

fig, ax = plt.subplots(figsize=(10, 6))
plot_comparison_with_target_var(n, target_var, ax)
st.pyplot(fig)
