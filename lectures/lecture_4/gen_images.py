"""Generate illustrations for lecture 1 presentation."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
})

# ── 1. 1D binary environment with agent ──────────────────────────────────────
def plot_1d_env():
    N = 20
    rng = np.random.default_rng(42)
    W = rng.integers(0, 2, N)
    agent_pos = 7
    M = 3  # perception window half-width

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-1.2, 2.2)
    ax.axis("off")

    # draw cells
    for i, bit in enumerate(W):
        color = "#2c3e50" if bit == 1 else "#ecf0f1"
        edgecolor = "#7f8c8d"
        rect = mpatches.FancyBboxPatch(
            (i - 0.45, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=edgecolor, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(i, 0.5, str(bit), ha="center", va="center",
                color="white" if bit == 1 else "#2c3e50", fontsize=13, fontweight="bold")
        ax.text(i, -0.25, str(i), ha="center", va="center", fontsize=9, color="#95a5a6")

    # perception window highlight
    for offset in range(-M, M + 1):
        idx = (agent_pos + offset) % N
        rect = mpatches.FancyBboxPatch(
            (idx - 0.45, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor="none", edgecolor="#e74c3c", linewidth=2.5,
            zorder=3
        )
        ax.add_patch(rect)

    # agent triangle
    ax.annotate("", xy=(agent_pos, 1.05), xytext=(agent_pos, 1.65),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2.5))
    ax.text(agent_pos, 1.85, "Агент\n$p=7$", ha="center", va="bottom",
            fontsize=13, color="#e74c3c", fontweight="bold")

    # ring topology arrows
    ax.annotate("", xy=(N - 0.55, -0.7), xytext=(0.55, -0.7),
                arrowprops=dict(arrowstyle="<->", color="#3498db", lw=1.8,
                                connectionstyle="arc3,rad=-0.4"))
    ax.text(N / 2, -1.1, "кільцева топологія ($N=20$)", ha="center", va="center",
            fontsize=11, color="#3498db", fontstyle="italic")

    # perception label
    ax.text(agent_pos, -0.55, f"сприйняття $M={2*M+1}$", ha="center", va="center",
            fontsize=10, color="#e74c3c")

    ax.set_title("1D бінарне дискретне середовище з агентом", pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/1d_env_agent.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved 1d_env_agent.png")


# ── 2. Rule 30 cellular automaton spacetime diagram ──────────────────────────
def apply_rule(row, rule_number=30):
    n = len(row)
    rule = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)
    new_row = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        l = row[(i - 1) % n]
        c = row[i]
        r = row[(i + 1) % n]
        idx = (l << 2) | (c << 1) | r
        new_row[i] = rule[idx]
    return new_row


def plot_rule30():
    N, T = 61, 40
    spacetime = np.zeros((T, N), dtype=np.uint8)
    spacetime[0, N // 2] = 1

    for t in range(1, T):
        spacetime[t] = apply_rule(spacetime[t - 1], rule_number=30)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(spacetime, cmap="binary", interpolation="nearest", aspect="auto")
    ax.set_xlabel("Позиція $i$", fontsize=13)
    ax.set_ylabel("Час $t$", fontsize=13)
    ax.set_title("Клітинний автомат — Правило 30\n(динамічне середовище $W_t$)", pad=12)
    ax.tick_params(labelsize=11)

    # agent path example — random walk
    rng = np.random.default_rng(7)
    pos = N // 2
    path_x, path_y = [pos], [0]
    for t in range(1, T):
        pos = (pos + rng.choice([-1, 0, 1])) % N
        path_x.append(pos)
        path_y.append(t)
    ax.plot(path_x, path_y, color="#e74c3c", linewidth=2, label="траєкторія агента")
    ax.scatter(path_x[0], path_y[0], color="#e74c3c", s=80, zorder=5)
    ax.legend(loc="upper right", fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{OUT}/rule30_spacetime.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved rule30_spacetime.png")


# ── 3. Agent-environment cycle diagram ───────────────────────────────────────
def plot_agent_env_cycle():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(cx, cy, w, h, label, sublabel="", color="#3498db"):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.92
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (0.22 if sublabel else 0), label,
                ha="center", va="center", fontsize=14, color="white", fontweight="bold")
        if sublabel:
            ax.text(cx, cy - 0.35, sublabel, ha="center", va="center",
                    fontsize=10, color="white", alpha=0.85)

    def arrow(x1, y1, x2, y2, label="", color="#2c3e50"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
        mx, my = (x1+x2)/2, (y1+y2)/2
        if label:
            ax.text(mx + 0.15, my, label, ha="left", va="center",
                    fontsize=12, color=color, fontstyle="italic")

    # boxes
    box(2.5, 4.5, 3.5, 1.2, "Середовище", "$W \\in \\{0,1\\}^N$", color="#2980b9")
    box(7.5, 4.5, 3.5, 1.2, "Агент", "$p \\in \\{0,\\ldots,N\\!-\\!1\\}$", color="#c0392b")

    box(5, 1.8, 2.8, 1.0, "Модель світу", "$\\hat{W}$", color="#27ae60")
    box(2, 1.8, 2.2, 1.0, "Цілі", "", color="#8e44ad")
    box(8, 1.8, 2.2, 1.0, "Стратегія", "$\\pi$", color="#d35400")

    # arrows env ↔ agent
    arrow(4.27, 4.5, 5.73, 4.5, label="$o_t$ (сприйняття)", color="#16a085")
    arrow(5.73, 4.2, 4.27, 4.2, label="$a_t$ (дія)      ", color="#e74c3c")

    # internal arrows
    ax.annotate("", xy=(5, 2.3), xytext=(7, 3.9),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.8,
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(8, 3.9), xytext=(8, 2.3),
                arrowprops=dict(arrowstyle="-|>", color="#d35400", lw=1.8))
    ax.annotate("", xy=(3.1, 2.3), xytext=(7, 3.9),
                arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=1.8,
                                connectionstyle="arc3,rad=0.35"))

    ax.set_title("Цикл агент – середовище", fontsize=16, pad=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/agent_env_cycle.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved agent_env_cycle.png")


# ── 4. Approaches to AGI – simple diagram ────────────────────────────────────
def plot_agi_approaches():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # center
    cx, cy = 6, 3.5
    center = mpatches.FancyBboxPatch((cx-1.4, cy-0.6), 2.8, 1.2,
                                      boxstyle="round,pad=0.15",
                                      facecolor="#2c3e50", edgecolor="white", lw=2)
    ax.add_patch(center)
    ax.text(cx, cy, "ЗШІ", ha="center", va="center", fontsize=18,
            color="white", fontweight="bold")

    branches = [
        (2.0, 5.8, "Біологічний", "Нейронні мережі\nнейроморфні чіпи", "#2980b9"),
        (2.0, 1.2, "Символічний", "Логіка, правила\nПролог, експертні системи", "#8e44ad"),
        (10.0, 5.8, "На основі даних", "Глибоке навчання\nТрансформери", "#16a085"),
        (10.0, 1.2, "Гібридний ✓", "Neuro-symbolic\n(цей курс)", "#c0392b"),
    ]

    for bx, by, title, sub, color in branches:
        rect = mpatches.FancyBboxPatch((bx-1.7, by-0.75), 3.4, 1.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="white", lw=1.8, alpha=0.9)
        ax.add_patch(rect)
        ax.text(bx, by + 0.22, title, ha="center", va="center",
                fontsize=13, color="white", fontweight="bold")
        ax.text(bx, by - 0.32, sub, ha="center", va="center",
                fontsize=9.5, color="white", alpha=0.88)
        # arrow from branch to center
        dx = cx - bx
        dy = cy - by
        length = (dx**2 + dy**2)**0.5
        ax.annotate("", xy=(bx + dx*0.37, by + dy*0.37),
                    xytext=(bx + dx*0.08, by + dy*0.08),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    ax.set_title("Підходи до створення ЗШІ", fontsize=16, pad=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/agi_approaches.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved agi_approaches.png")


if __name__ == "__main__":
    plot_1d_env()
    plot_rule30()
    plot_agent_env_cycle()
    plot_agi_approaches()
    print("All images generated.")
