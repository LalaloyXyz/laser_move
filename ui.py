from typing import List, Optional
import tkinter as tk
import serial
import time

select = int(input("Select UI mode (1: head, 2: body): "))

class PositionUI:
    def __init__(self, title: str = "Positions",
                 port: str = "/dev/ttyACM0", baud: int = 115200) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="#2b2b2b")

        self.pos_text = tk.Text(
            self.root,
            height=20,
            width=30,
            font=("Consolas", 12),
            bg="#2b2b2b",
            fg="#ffffff",
            insertbackground="white",
            bd=0,
            highlightthickness=0
        )
        self.pos_text.pack(padx=5, pady=5)

        self.selected_id: Optional[int] = None
        self.isTarget: int = 0

        if select == 1:
            self.mode: str = "head"
        elif select == 2:
            self.mode: str = "body"

        # --- Setup Serial ---
        try:
            self.serial = serial.Serial(port, baud, timeout=1)
            print(f"[OK] Serial connected → {port} at {baud} baud")
        except Exception as e:
            print(f"[ERR] Serial failed: {e}")
            self.serial = None

    def set_selected(self, track_id: Optional[int]) -> None:
        self.selected_id = track_id

    def update_positions(self, tracks, frame_shape) -> None:
        self.pos_text.delete("1.0", tk.END)
        h, w = frame_shape[:2]

        target_track = None
        other_tracks: List = []

        # --- Pick target ---
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            if self.selected_id is not None and t.track_id == self.selected_id:
                target_track = t
            else:
                other_tracks.append(t)

        if target_track:
            # --- Calculate position based on mode ---
            l, t_, r, b = map(int, target_track.to_ltrb())
            cx = (l + r) // 2
            if self.mode == "head":
                # Estimate head position: top 1/4 of the person's height
                person_height = b - t_
                y_point = t_ + (person_height // 4)
            else:
                y_point = (t_ + b) // 2

            nx = (cx / w) * 2 - 1
            ny = -((y_point / h) * 2 - 1)

            self.pos_text.insert(
                tk.END,
                f"ID {target_track.track_id} ({self.mode}): x={nx:.2f}, y={ny:.2f}\n",
                "target"
            )

            # --- Target flag ---
            self.isTarget = 0 if target_track.track_id == 0 else 1

            # --- Send with updated coords ---
            if self.serial and self.serial.is_open:
                msg = f"{nx:.2f},{ny:.2f},{self.isTarget}\n"
                try:
                    self.serial.write(msg.encode())
                    print(f"[TX] {msg.strip()}")
                except Exception as e:
                    print(f"[ERR] Serial write: {e}")
        else:
            # --- No target detected ---
            self.isTarget = 0
            if self.serial and self.serial.is_open:
                msg = f"0.00,0.00,{self.isTarget}\n"
                try:
                    self.serial.write(msg.encode())
                    print(f"[TX] {msg.strip()}")
                except Exception as e:
                    print(f"[ERR] Serial write: {e}")

        # --- Show other tracks ---
        for t in other_tracks:
            l, t_, r, b = map(int, t.to_ltrb())
            cx = (l + r) // 2
            if self.mode == "head":
                # Estimate head position: top 1/4 of the person's height
                person_height = b - t_
                y_point = t_ + (person_height // 4)
            else:
                y_point = (t_ + b) // 2
            nx = (cx / w) * 1.5 - 1
            ny = -((y_point / h) * 2 - 1)
            self.pos_text.insert(
                tk.END, f"ID {t.track_id}: x={nx:.2f}, y={ny:.2f}\n"
            )

        # Highlight selected track in bright green
        self.pos_text.tag_config("target", foreground="#00ff00")
        self.root.update_idletasks()

    def toggle_mode(self) -> None:
        self.mode = "body" if self.mode == "head" else "head"
        print(f"[MODE] {self.mode}")

    def destroy(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.root.destroy()
