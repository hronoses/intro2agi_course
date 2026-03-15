"""Generate chess, go and atari board illustrations using matplotlib."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans"})


# ── Chess board ───────────────────────────────────────────────────────────────
def plot_chess():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(8):
        for c in range(8):
            color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
            ax.add_patch(mpatches.Rectangle((c, r), 1, 1, facecolor=color))

    pieces = {
        # row, col, symbol, color
        (7, 0): ("♜", "black"), (7, 1): ("♞", "black"), (7, 2): ("♝", "black"),
        (7, 3): ("♛", "black"), (7, 4): ("♚", "black"), (7, 5): ("♝", "black"),
        (7, 6): ("♞", "black"), (7, 7): ("♜", "black"),
        (6, 0): ("♟", "black"), (6, 1): ("♟", "black"), (6, 2): ("♟", "black"),
        (6, 3): ("♟", "black"), (6, 4): ("♟", "black"), (6, 5): ("♟", "black"),
        (6, 6): ("♟", "black"), (6, 7): ("♟", "black"),
        (1, 0): ("♙", "white"), (1, 1): ("♙", "white"), (1, 2): ("♙", "white"),
        (1, 3): ("♙", "white"), (1, 4): ("♙", "white"), (1, 5): ("♙", "white"),
        (1, 6): ("♙", "white"), (1, 7): ("♙", "white"),
        (0, 0): ("♖", "white"), (0, 1): ("♘", "white"), (0, 2): ("♗", "white"),
        (0, 3): ("♕", "white"), (0, 4): ("♔", "white"), (0, 5): ("♗", "white"),
        (0, 6): ("♘", "white"), (0, 7): ("♖", "white"),
    }
    for (r, c), (sym, col) in pieces.items():
        ax.text(c + 0.5, r + 0.5, sym, ha="center", va="center",
                fontsize=22, color=col,
                path_effects=[
                    __import__("matplotlib.patheffects", fromlist=["withStroke"])
                    .withStroke(linewidth=1.5, foreground="gray" if col == "white" else "white")
                ])

    ax.set_title("Шахи", fontsize=14, pad=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/chess.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved chess.png")


# ── Go board ──────────────────────────────────────────────────────────────────
def plot_go():
    N = 9  # 9x9 board for clarity
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#dcb167")
    ax.axis("off")

    # grid lines
    for i in range(N):
        ax.plot([0, N - 1], [i, i], color="#7a5c2e", lw=0.9)
        ax.plot([i, i], [0, N - 1], color="#7a5c2e", lw=0.9)

    # star points
    for sx, sy in [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]:
        ax.plot(sx, sy, "o", color="#7a5c2e", markersize=5)

    # some stones
    rng = np.random.default_rng(3)
    black_stones = [(1,1),(2,3),(3,2),(4,5),(5,4),(6,7),(7,6),(3,6),(5,2)]
    white_stones = [(1,5),(2,6),(4,3),(5,6),(6,3),(7,4),(3,4),(6,5),(2,1)]

    for x, y in black_stones:
        circle = plt.Circle((x, y), 0.4, color="#1a1a1a", zorder=3)
        ax.add_patch(circle)
    for x, y in white_stones:
        circle = plt.Circle((x, y), 0.4, color="#f5f5f5", zorder=3)
        edge = plt.Circle((x, y), 0.4, color="#888", fill=False, lw=1, zorder=4)
        ax.add_patch(circle)
        ax.add_patch(edge)

    ax.set_title("Го", fontsize=14, pad=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/go.png", dpi=130, bbox_inches="tight", facecolor="#dcb167")
    plt.close(fig)
    print("Saved go.png")


# ── Atari Breakout-style screenshot ───────────────────────────────────────────
def plot_atari():
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 210)
    ax.set_facecolor("#000000")
    ax.axis("off")

    colors = ["#ff0000", "#ff6600", "#ffff00", "#00cc00", "#00ccff", "#9900cc"]
    for row_i, color in enumerate(colors):
        for col in range(14):
            x = 4 + col * 11
            y = 180 - row_i * 10
            ax.add_patch(mpatches.Rectangle((x, y), 9, 7, facecolor=color))

    # paddle
    ax.add_patch(mpatches.Rectangle((55, 20), 40, 6, facecolor="#aaaaaa"))
    # ball
    ax.add_patch(mpatches.Rectangle((78, 80), 4, 4, facecolor="white"))
    # score area
    ax.text(80, 200, "SCORE  042", ha="center", va="center",
            fontsize=9, color="white", family="monospace")

    ax.set_title("Atari Breakout", fontsize=14, color="white", pad=6,
                 backgroundcolor="black")
    fig.tight_layout()
    fig.savefig(f"{OUT}/atari.png", dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print("Saved atari.png")


if __name__ == "__main__":
    plot_chess()
    plot_go()
    plot_atari()
    print("All game images generated.")
