import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SCRIPTS_DIR = os.path.join(
    BASE_DIR,
    "scripts"
)


# ============================================================
# Run Game
# ============================================================

def run_game(script_name):

    script_path = os.path.join(
        SCRIPTS_DIR,
        script_name
    )

    if not os.path.exists(script_path):

        messagebox.showerror(
            "Error",
            f"Game file not found:\n\n{script_path}"
        )

        return

    # Hide main window
    root.withdraw()

    try:

        # Run selected game
        subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            f"Could not start the game:\n\n{e}"
        )

    finally:

        # Show Home Page again
        root.deiconify()


# ============================================================
# Human vs Computer
# ============================================================

def human_vs_computer():

    run_game(
        "human_vs_computer.py"
    )


# ============================================================
# Human vs Human
# ============================================================

def human_vs_human():

    run_game(
        "human_vs_human.py"
    )


# ============================================================
# Exit
# ============================================================

def exit_app():

    root.destroy()


# ============================================================
# Main Window
# ============================================================

root = tk.Tk()

root.title(
    "Rock Paper Scissors AI"
)

root.geometry(
    "700x500"
)

root.resizable(
    False,
    False
)


# ============================================================
# Title
# ============================================================

title_label = tk.Label(
    root,
    text="ROCK PAPER SCISSORS",
    font=("Arial", 30, "bold")
)

title_label.pack(
    pady=(60, 10)
)


# ============================================================
# Subtitle
# ============================================================

subtitle_label = tk.Label(
    root,
    text="AI Hand Gesture Recognition Game",
    font=("Arial", 14)
)

subtitle_label.pack(
    pady=(0, 40)
)


# ============================================================
# Human vs Computer Button
# ============================================================

computer_button = tk.Button(
    root,
    text="HUMAN VS COMPUTER",
    font=("Arial", 16, "bold"),
    width=25,
    height=2,
    command=human_vs_computer
)

computer_button.pack(
    pady=10
)


# ============================================================
# Human vs Human Button
# ============================================================

human_button = tk.Button(
    root,
    text="HUMAN VS HUMAN",
    font=("Arial", 16, "bold"),
    width=25,
    height=2,
    command=human_vs_human
)

human_button.pack(
    pady=10
)


# ============================================================
# Exit Button
# ============================================================

exit_button = tk.Button(
    root,
    text="EXIT",
    font=("Arial", 12),
    width=15,
    height=1,
    command=exit_app
)

exit_button.pack(
    pady=30
)


# ============================================================
# Footer
# ============================================================

footer_label = tk.Label(
    root,
    text="MediaPipe + Random Forest",
    font=("Arial", 10)
)

footer_label.pack(
    side="bottom",
    pady=15
)


# ============================================================
# Start Application
# ============================================================

root.mainloop()

