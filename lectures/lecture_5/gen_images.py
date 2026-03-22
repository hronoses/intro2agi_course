"""Generate illustrations for Lecture 5: 2D Binary Discrete Environment."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
})


# ── 1. Toroidal topology ─────────────────────────────────────────────────────
def plot_toroidal_topology():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: flat grid with arrows showing wrapping
    ax = axes[0]
    ax.set_xlim(-1.5, 8.5)
    ax.set_ylim(-1.5, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Плоска сітка з тороїдальним загортанням", fontsize=13, pad=10)

    N = 7
    for y in range(N):
        for x in range(N):
            color = "#ecf0f1" if (x + y) % 2 == 0 else "#d5dbdb"
            rect = mpatches.Rectangle((x, N - 1 - y), 0.95, 0.95,
                                       facecolor=color, edgecolor="#95a5a6", lw=0.8)
            ax.add_patch(rect)

    # Wrapping arrows: right edge → left edge
    for y_idx in [1, 3, 5]:
        y = N - 1 - y_idx + 0.47
        ax.annotate("", xy=(-0.3, y), xytext=(N + 0.3, y),
                     arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2,
                                     connectionstyle="arc3,rad=-0.3"))
    # Top → bottom
    for x_idx in [1, 3, 5]:
        x = x_idx + 0.47
        ax.annotate("", xy=(x, -0.3), xytext=(x, N + 0.3),
                     arrowprops=dict(arrowstyle="-|>", color="#3498db", lw=2,
                                     connectionstyle="arc3,rad=0.3"))

    ax.text(N / 2 + 0.5, -1.2, "← → з'єднані (вісь X)", ha="center",
            fontsize=10, color="#e74c3c", fontstyle="italic")
    ax.text(-1.2, N / 2, "↑ ↓ з'єднані\n(вісь Y)", ha="center", va="center",
            fontsize=10, color="#3498db", fontstyle="italic", rotation=90)

    # Right: conceptual torus (2D schematic)
    ax2 = axes[1]
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Результат: тор (схематично)", fontsize=13, pad=10)

    # Draw torus as two concentric ellipses with connecting lines
    theta = np.linspace(0, 2 * np.pi, 100)
    # Outer ellipse
    ax2.plot(3 * np.cos(theta), 2 * np.sin(theta), color="#2980b9", lw=2.5)
    # Inner ellipse (hole)
    ax2.plot(1.2 * np.cos(theta), 0.8 * np.sin(theta), color="#2980b9", lw=2, ls="--")
    # Arrows showing wrap-around
    ax2.annotate("", xy=(-2.2, 1.6), xytext=(2.2, 1.6),
                 arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=2))
    ax2.text(0, 2.1, "горизонтальне загортання", ha="center", fontsize=10, color="#e74c3c")
    ax2.annotate("", xy=(3.2, -0.8), xytext=(3.2, 0.8),
                 arrowprops=dict(arrowstyle="<->", color="#3498db", lw=2))
    ax2.text(3.7, 0, "вертикальне", ha="left", fontsize=10, color="#3498db", rotation=90)
    ax2.text(0, -2.8, "Протилежні краї з'єднані", ha="center",
             fontsize=11, color="#555", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(f"{OUT}/toroidal_topology.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved toroidal_topology.png")


# ── 2. Game of Life patterns ─────────────────────────────────────────────────
def plot_gol_patterns():
    patterns = {
        "Block\n(still life, k=1)": [
            np.array([[1, 1], [1, 1]])
        ],
        "Beehive\n(still life, k=1)": [
            np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
        ],
        "Blinker\n(oscillator, k=2)": [
            np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        ],
        "Toad\n(oscillator, k=2)": [
            np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]]),
            np.array([[0, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 0]]),
        ],
        "Glider\n(spaceship)": [
            np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]),
            np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]]),  # simplified
        ],
    }

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (name, frames) in zip(axes, patterns.items()):
        n_frames = len(frames)
        total_w = sum(f.shape[1] for f in frames) + (n_frames - 1)
        max_h = max(f.shape[0] for f in frames)

        ax.set_xlim(-0.5, total_w - 0.5)
        ax.set_ylim(-0.5, max_h - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=11, pad=8)

        offset_x = 0
        for fi, frame in enumerate(frames):
            h, w = frame.shape
            for y in range(h):
                for x in range(w):
                    color = "#2c3e50" if frame[y, x] else "#ecf0f1"
                    rect = mpatches.Rectangle(
                        (offset_x + x - 0.45, max_h - 1 - y - 0.45), 0.9, 0.9,
                        facecolor=color, edgecolor="#95a5a6", lw=1)
                    ax.add_patch(rect)
            if fi < n_frames - 1:
                mid_y = max_h / 2 - 0.5
                ax.annotate("→", xy=(offset_x + w + 0.1, mid_y),
                            fontsize=14, ha="center", va="center", color="#e74c3c")
            offset_x += w + 1

    fig.suptitle("Патерни Game of Life", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/gol_patterns.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved gol_patterns.png")


# ── 3. Perception window ─────────────────────────────────────────────────────
def plot_perception_window():
    W, H = 15, 15
    S = 3
    ax_pos, ay_pos = 7, 7

    rng = np.random.default_rng(42)
    world = rng.integers(0, 2, (H, W)).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    for y in range(H):
        for x in range(W):
            in_perc = abs(x - ax_pos) <= S and abs(y - ay_pos) <= S
            if x == ax_pos and y == ay_pos:
                color = "#E64A19"
            elif in_perc:
                color = "#90CAF9" if world[y, x] else "#1a3a5c"
            else:
                color = "#e0e0e0" if world[y, x] else "#2c2c2c"
            rect = mpatches.Rectangle((x - 0.48, H - 1 - y - 0.48), 0.96, 0.96,
                                       facecolor=color, edgecolor="#555", lw=0.5)
            ax.add_patch(rect)

    # Agent label
    ax.text(ax_pos, H - 1 - ay_pos, "A", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")

    # Perception rectangle
    rect = mpatches.Rectangle(
        (ax_pos - S - 0.5, H - 1 - (ay_pos + S) - 0.5),
        2 * S + 1, 2 * S + 1,
        fill=False, edgecolor="#E64A19", lw=2.5, linestyle="--")
    ax.add_patch(rect)

    ax.text(ax_pos, H - 1 - (ay_pos + S) - 0.8,
            f"Вікно сприйняття (2S+1)² = {(2*S+1)**2}",
            ha="center", va="top", fontsize=12, color="#E64A19")

    ax.set_title(f"Агент на 2D сітці {W}×{H}, S={S}", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/perception_window.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved perception_window.png")


# ── 4. Koch-Itti saliency model ──────────────────────────────────────────────
def plot_koch_itti():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(cx, cy, w, h, label, sublabel="", color="#3498db"):
        rect = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        dy = 0.18 if sublabel else 0
        ax.text(cx, cy + dy, label, ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
        if sublabel:
            ax.text(cx, cy - 0.3, sublabel, ha="center", va="center",
                    fontsize=9, color="white", alpha=0.8)

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))

    # Input
    box(1.5, 3.5, 2.2, 1, "Вхід", "зображення", "#7f8c8d")

    # Feature maps
    features = [("Колір", 5.5), ("Орієнтація", 3.5), ("Рух", 1.5)]
    for label, y_pos in features:
        box(5, y_pos, 2, 0.8, label, "", "#2980b9")
        arrow(2.6, 3.5, 3.95, y_pos)

    # Normalization
    for y_pos in [5.5, 3.5, 1.5]:
        box(8.5, y_pos, 2, 0.8, "Норм.", "конкуренція", "#8e44ad")
        arrow(6.05, y_pos, 7.45, y_pos)

    # Combination
    box(11.5, 3.5, 2.2, 1.2, "Карта\nсалієнтності", "S(x,y)", "#e74c3c")
    for y_pos in [5.5, 3.5, 1.5]:
        arrow(9.55, y_pos, 10.35, 3.5)

    # Output
    ax.text(11.5, 1.8, "argmax S → точка фіксації",
            ha="center", fontsize=10, color="#e74c3c", fontstyle="italic")

    ax.set_title("Модель Коха–Ітті (спрощена схема)", fontsize=15, pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/koch_itti.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved koch_itti.png")


# ── 5. What/Where pathways ───────────────────────────────────────────────────
def plot_what_where():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(cx, cy, w, h, label, sublabel="", color="#3498db"):
        rect = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        dy = 0.2 if sublabel else 0
        ax.text(cx, cy + dy, label, ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
        if sublabel:
            ax.text(cx, cy - 0.3, sublabel, ha="center", va="center",
                    fontsize=9, color="white", alpha=0.85)

    def arrow(x1, y1, x2, y2, label="", color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx - 0.1, my + 0.3, label, fontsize=10,
                    ha="center", va="bottom", color=color, fontstyle="italic")

    # Eye → V1
    box(1.5, 3.5, 2, 1, "Сітківка", "вхід", "#7f8c8d")
    box(5, 3.5, 1.5, 1, "V1", "первинна\nкора", "#34495e")
    arrow(2.55, 3.5, 4.2, 3.5, color="#555")

    # Ventral (What)
    box(8.5, 5.5, 2.2, 1, "IT кора", "форма, текстура", "#27ae60")
    box(12, 5.5, 2, 1, "ЩО?", "ознака  fᵢ", "#1B5E20")
    arrow(5.8, 3.8, 7.35, 5.3, "вентральний", "#27ae60")
    arrow(9.65, 5.5, 10.95, 5.5, color="#27ae60")

    # Dorsal (Where)
    box(8.5, 1.5, 2.5, 1, "Тім'яна кора", "простір, рух", "#2980b9")
    box(12, 1.5, 2, 1, "ДЕ?", "позиція  pᵢ", "#1565C0")
    arrow(5.8, 3.2, 7.2, 1.7, "дорсальний", "#2980b9")
    arrow(9.8, 1.5, 10.95, 1.5, color="#2980b9")

    ax.set_title("Два зорових шляхи: «Що?» та «Де?»", fontsize=15, pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/what_where.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved what_where.png")


# ── 6. Raster vs Object model ────────────────────────────────────────────────
def plot_raster_vs_object():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: raster map
    ax = axes[0]
    ax.set_title("Растрова карта", fontsize=14, pad=10)
    N = 12
    rng = np.random.default_rng(42)
    grid = np.zeros((N, N), dtype=np.uint8)
    # Place some patterns
    grid[2:4, 2:4] = 1  # block
    grid[6, 4:7] = 1    # blinker
    grid[2:4, 8:10] = 1 # block2
    grid[9, 1:4] = 1    # blinker2

    for y in range(N):
        for x in range(N):
            known = y < 9  # partially known
            if not known:
                color = "#aaa"
            elif grid[y, x]:
                color = "#2c3e50"
            else:
                color = "#ecf0f1"
            rect = mpatches.Rectangle((x, N - 1 - y), 0.95, 0.95,
                                       facecolor=color, edgecolor="#95a5a6", lw=0.5)
            ax.add_patch(rect)
            if not known:
                ax.text(x + 0.47, N - 1 - y + 0.47, "?", ha="center", va="center",
                        fontsize=8, color="#555")
    ax.set_xlim(-0.3, N + 0.3)
    ax.set_ylim(-0.3, N + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(N / 2, -0.8, f"Розмір: W × H = {N}×{N} = {N*N} біт",
            ha="center", fontsize=11, color="#555")

    # Right: object model
    ax2 = axes[1]
    ax2.set_title("Об'єктна модель", fontsize=14, pad=10)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    objects = [
        ("Клас 1: Block (k=1)", "#2980b9",
         "■■\n■■", "× 2 шт.", "(2,2), (8,2)"),
        ("Клас 2: Blinker (k=2)", "#e74c3c",
         "▪▪▪ ↔ ▪\n      ▪\n      ▪", "× 2 шт.", "(5,6), (2,9)"),
    ]

    y_pos = 8.5
    for title, color, pattern, count, positions in objects:
        ax2.text(0.5, y_pos, title, fontsize=12, fontweight="bold", color=color, va="top")
        y_pos -= 0.6
        ax2.text(1, y_pos, f"Ознака: {pattern}", fontsize=10, va="top",
                 family="monospace", color="#333")
        y_pos -= 1.2
        ax2.text(1, y_pos, f"Кількість: {count}", fontsize=10, va="top", color="#555")
        y_pos -= 0.5
        ax2.text(1, y_pos, f"Позиції: {positions}", fontsize=10, va="top", color="#555")
        y_pos -= 1.2

    ax2.text(5, 0.5, "Розмір: C ознак + K координат\n≪ W × H біт",
             ha="center", fontsize=11, color="#555")

    fig.suptitle("Растрова карта   vs   Об'єктна модель", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/raster_vs_object.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved raster_vs_object.png")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_toroidal_topology()
    plot_gol_patterns()
    plot_perception_window()
    plot_koch_itti()
    plot_what_where()
    plot_raster_vs_object()
    print(f"\nAll images saved to {OUT}/")
