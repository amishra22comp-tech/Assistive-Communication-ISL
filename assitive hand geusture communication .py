import tkinter as tk
from tkinter import messagebox
import subprocess
import threading


# ============================================================
# ASSISTIVE COMMUNICATION APPLICATION
# FRONTEND ONLY
# ============================================================

root = tk.Tk()

root.title("Assistive Communication Application")

root.geometry("1150x720")

root.minsize(950, 620)

root.configure(bg="#F5F5F7")


# ============================================================
# COLORS
# ============================================================

BG = "#F5F5F7"
WHITE = "#FFFFFF"
BLACK = "#171717"
DARK = "#292929"
GRAY = "#777777"
LIGHT_GRAY = "#E5E5E5"

PURPLE = "#6750A4"
LIGHT_PURPLE = "#EEE8FA"

GREEN = "#20A464"
RED = "#D93025"


# ============================================================
# APPLICATION DATA
# ============================================================

sentences = {

    "HELP": "I need help.",
    "WATER": "I want some water.",
    "FOOD": "I want to eat food.",
    "HOME": "I want to go home.",
    "THANK YOU": "Thank you for helping me.",
    "HELLO": "Hello, nice to meet you.",
    "YES": "Yes, I agree.",
    "NO": "No, I don't agree.",
    "CALL": "Please call my family.",
    "DOCTOR": "I need to see a doctor.",
    "STOP": "Please stop.",
    "WAIT": "Please wait.",
    "SORRY": "I am sorry.",
    "PLEASE": "Please help me.",
    "HAPPY": "I am feeling happy.",
    "SAD": "I am feeling sad."

}


current_sentence = "Show your sign to the camera."

history = []


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak_text(text):

    if not text:
        return

    def speak():

        try:

            safe_text = text.replace(
                "'",
                "''"
            )

            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object "
                "System.Speech.Synthesis.SpeechSynthesizer; "
                "$speaker.Rate = 0; "
                "$speaker.Volume = 100; "
                "$speaker.Speak('"
                + safe_text +
                "')"
            )

            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    command
                ],
                creationflags=subprocess.CREATE_NO_WINDOW
            )

        except Exception as e:

            print("Speech error:", e)


    threading.Thread(
        target=speak,
        daemon=True
    ).start()


# ============================================================
# SHOW SIGN
# ============================================================

def show_sign(sign):

    global current_sentence

    current_sentence = sentences.get(
        sign,
        "Sentence not available."
    )

    detected_sign_label.config(
        text=sign
    )

    sentence_label.config(
        text=current_sentence
    )

    status_label.config(
        text="✓ Sign detected"
    )

    camera_status.config(
        text="● HAND DETECTED",
        fg=GREEN
    )

    add_history(
        sign,
        current_sentence
    )


# ============================================================
# SPEAK
# ============================================================

def speak_current():

    speak_text(
        current_sentence
    )


# ============================================================
# CLEAR
# ============================================================

def clear_result():

    global current_sentence

    current_sentence = (
        "Show your sign to the camera."
    )

    detected_sign_label.config(
        text="No sign detected"
    )

    sentence_label.config(
        text=current_sentence
    )

    status_label.config(
        text="Ready"
    )

    camera_status.config(
        text="● CAMERA READY",
        fg=GRAY
    )


# ============================================================
# CAMERA FRONTEND CONTROLS
# ============================================================

def start_camera():

    camera_status.config(
        text="● CAMERA ACTIVE",
        fg=GREEN
    )

    camera_display.config(
        text=(
            "📷\n\n"
            "LIVE CAMERA\n\n"
            "Camera feed will be connected\n"
            "to the backend."
        ),
        fg="#CCCCCC"
    )

    status_label.config(
        text="Camera active • Waiting for sign..."
    )


def stop_camera():

    camera_status.config(
        text="● CAMERA OFF",
        fg=GRAY
    )

    camera_display.config(
        text=(
            "📷\n\n"
            "CAMERA PREVIEW\n\n"
            "Click START CAMERA"
        ),
        fg="#AAAAAA"
    )

    status_label.config(
        text="Camera stopped"
    )


# ============================================================
# HISTORY
# ============================================================

def add_history(
    sign,
    sentence
):

    history.insert(
        0,
        (sign, sentence)
    )

    update_history()


def update_history():

    for widget in history_inner.winfo_children():

        widget.destroy()


    for sign, sentence in history[:5]:

        row = tk.Frame(
            history_inner,
            bg=WHITE
        )

        row.pack(
            fill="x",
            pady=5
        )


        tk.Label(
            row,
            text="✋",
            bg=LIGHT_PURPLE,
            fg=PURPLE,
            font=("Arial", 11),
            width=3
        ).pack(
            side="left",
            padx=(0, 8)
        )


        text_box = tk.Frame(
            row,
            bg=WHITE
        )

        text_box.pack(
            side="left",
            fill="x",
            expand=True
        )


        tk.Label(
            text_box,
            text=sign,
            bg=WHITE,
            fg=BLACK,
            font=("Arial", 9, "bold")
        ).pack(
            anchor="w"
        )


        tk.Label(
            text_box,
            text=sentence,
            bg=WHITE,
            fg=GRAY,
            font=("Arial", 9)
        ).pack(
            anchor="w"
        )


        tk.Button(
            row,
            text="🔊",
            bg=WHITE,
            fg=PURPLE,
            relief="flat",
            borderwidth=0,
            command=lambda s=sentence:
                speak_text(s)
        ).pack(
            side="right"
        )


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=BG
)

header.pack(
    fill="x",
    padx=30,
    pady=(22, 12)
)


# Logo

tk.Label(
    header,
    text="✋",
    bg=PURPLE,
    fg=WHITE,
    font=("Arial", 24),
    width=3
).pack(
    side="left"
)


# Title

title_box = tk.Frame(
    header,
    bg=BG
)

title_box.pack(
    side="left",
    padx=15
)


tk.Label(
    title_box,
    text="Assistive Communication",
    bg=BG,
    fg=BLACK,
    font=("Arial", 22, "bold")
).pack(
    anchor="w"
)


tk.Label(
    title_box,
    text="Indian Sign Language → English → Voice",
    bg=BG,
    fg=GRAY,
    font=("Arial", 10)
).pack(
    anchor="w"
)


# Camera status

camera_status = tk.Label(
    header,
    text="● CAMERA READY",
    bg=BG,
    fg=GRAY,
    font=("Arial", 10, "bold")
)

camera_status.pack(
    side="right",
    pady=10
)


# ============================================================
# MAIN AREA
# ============================================================

main = tk.Frame(
    root,
    bg=BG
)

main.pack(
    fill="both",
    expand=True,
    padx=30
)


# ============================================================
# LEFT PANEL
# ============================================================

left = tk.Frame(
    main,
    bg=BG
)

left.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 12)
)


# ============================================================
# CAMERA CARD
# ============================================================

camera_card = tk.Frame(
    left,
    bg=BLACK,
    height=410
)

camera_card.pack(
    fill="both",
    expand=True
)

camera_card.pack_propagate(
    False
)


camera_display = tk.Label(
    camera_card,
    text=(
        "📷\n\n"
        "CAMERA PREVIEW\n\n"
        "Click START CAMERA"
    ),
    bg=BLACK,
    fg="#AAAAAA",
    font=("Arial", 16),
    justify="center"
)

camera_display.pack(
    fill="both",
    expand=True
)


# ============================================================
# CAMERA BUTTONS
# ============================================================

camera_buttons = tk.Frame(
    left,
    bg=BG
)

camera_buttons.pack(
    fill="x",
    pady=12
)


tk.Button(
    camera_buttons,
    text="📷 START CAMERA",
    bg=PURPLE,
    fg=WHITE,
    activebackground=PURPLE,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    font=("Arial", 10, "bold"),
    command=start_camera
).pack(
    side="left",
    fill="x",
    expand=True,
    ipady=10,
    padx=(0, 5)
)


tk.Button(
    camera_buttons,
    text="⏹ STOP CAMERA",
    bg=WHITE,
    fg=PURPLE,
    relief="solid",
    cursor="hand2",
    font=("Arial", 10, "bold"),
    command=stop_camera
).pack(
    side="right",
    fill="x",
    expand=True,
    ipady=10,
    padx=(5, 0)
)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    left,
    text="Ready",
    bg=BG,
    fg=GRAY,
    font=("Arial", 9)
)

status_label.pack(
    pady=(0, 5)
)


# ============================================================
# RIGHT PANEL
# ============================================================

right = tk.Frame(
    main,
    bg=BG,
    width=390
)

right.pack(
    side="right",
    fill="both",
    padx=(12, 0)
)

right.pack_propagate(
    False
)


# ============================================================
# DETECTED SIGN
# ============================================================

tk.Label(
    right,
    text="Detected Sign",
    bg=BG,
    fg=GRAY,
    font=("Arial", 10, "bold")
).pack(
    anchor="w"
)


sign_card = tk.Frame(
    right,
    bg=WHITE,
    height=78
)

sign_card.pack(
    fill="x",
    pady=(7, 17)
)

sign_card.pack_propagate(
    False
)


tk.Label(
    sign_card,
    text="✋",
    bg=LIGHT_PURPLE,
    fg=PURPLE,
    font=("Arial", 22),
    width=3
).pack(
    side="left",
    fill="y"
)


detected_sign_label = tk.Label(
    sign_card,
    text="No sign detected",
    bg=WHITE,
    fg=BLACK,
    font=("Arial", 15, "bold")
)

detected_sign_label.pack(
    side="left",
    padx=15
)


# ============================================================
# ENGLISH SENTENCE
# ============================================================

tk.Label(
    right,
    text="English Sentence",
    bg=BG,
    fg=GRAY,
    font=("Arial", 10, "bold")
).pack(
    anchor="w"
)


sentence_card = tk.Frame(
    right,
    bg=PURPLE,
    height=145
)

sentence_card.pack(
    fill="x",
    pady=(7, 15)
)

sentence_card.pack_propagate(
    False
)


sentence_label = tk.Label(
    sentence_card,
    text=current_sentence,
    bg=PURPLE,
    fg=WHITE,
    font=("Arial", 16, "bold"),
    wraplength=330,
    justify="left"
)

sentence_label.pack(
    expand=True,
    padx=20
)


# ============================================================
# ACTION BUTTONS
# ============================================================

actions = tk.Frame(
    right,
    bg=BG
)

actions.pack(
    fill="x"
)


tk.Button(
    actions,
    text="🔊 SPEAK",
    bg=WHITE,
    fg=PURPLE,
    relief="solid",
    cursor="hand2",
    font=("Arial", 10, "bold"),
    command=speak_current
).pack(
    side="left",
    fill="x",
    expand=True,
    ipady=10,
    padx=(0, 5)
)


tk.Button(
    actions,
    text="CLEAR",
    bg=PURPLE,
    fg=WHITE,
    relief="flat",
    cursor="hand2",
    font=("Arial", 10, "bold"),
    command=clear_result
).pack(
    side="right",
    fill="x",
    expand=True,
    ipady=10,
    padx=(5, 0)
)


# ============================================================
# TEST TRANSLATION
# ============================================================

tk.Label(
    right,
    text="Test Translation",
    bg=BG,
    fg=BLACK,
    font=("Arial", 13, "bold")
).pack(
    anchor="w",
    pady=(18, 8)
)


test_frame = tk.Frame(
    right,
    bg=BG
)

test_frame.pack(
    fill="x"
)


test_signs = [
    "HELP",
    "WATER",
    "FOOD",
    "HOME",
    "THANK YOU",
    "HELLO",
    "YES",
    "NO"
]


for i, sign in enumerate(test_signs):

    tk.Button(
        test_frame,
        text=sign,
        bg=WHITE,
        fg=PURPLE,
        relief="solid",
        cursor="hand2",
        font=("Arial", 8, "bold"),
        command=lambda s=sign:
            show_sign(s)
    ).grid(
        row=i // 2,
        column=i % 2,
        sticky="ew",
        padx=3,
        pady=3,
        ipady=5
    )


test_frame.columnconfigure(
    0,
    weight=1
)

test_frame.columnconfigure(
    1,
    weight=1
)


# ============================================================
# HISTORY
# ============================================================

tk.Label(
    right,
    text="Recent Conversations",
    bg=BG,
    fg=BLACK,
    font=("Arial", 13, "bold")
).pack(
    anchor="w",
    pady=(17, 7)
)


history_card = tk.Frame(
    right,
    bg=WHITE
)

history_card.pack(
    fill="both",
    expand=True
)


history_inner = tk.Frame(
    history_card,
    bg=WHITE
)

history_inner.pack(
    fill="both",
    expand=True,
    padx=12,
    pady=10
)


# ============================================================
# FOOTER
# ============================================================

tk.Label(
    root,
    text="Assistive Communication • ISL Recognition System",
    bg=BG,
    fg=GRAY,
    font=("Arial", 8)
).pack(
    pady=8
)


# ============================================================
# CLOSE
# ============================================================

root.protocol(
    "WM_DELETE_WINDOW",
    root.destroy
)


# ============================================================
# RUN
# ============================================================

root.mainloop()