import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from tkinter import ttk

def launch_video_labeler(simulation, pairs_path, out_paths, verbose=False):
    """
    Launch the video labeler app.
    
    Args:
        simulation: Simulation object that will be used for generating pairs
        pairs_path: Path to the pairs CSV file
        out_paths: Dictionary containing output paths
        verbose: Whether to print verbose output
    
    Note: executes in the main thread.
    """
    root = tk.Tk()
    app = VideoLabelerApp(root, simulation, pairs_path, out_paths, verbose)
    root.protocol("WM_DELETE_WINDOW", app.save_and_exit)
    root.mainloop()

class VideoLabelerApp:
    def __init__(self, master, simulation, pairs_path, out_paths, verbose=False):
        self.master = master
        self.simulation = simulation
        self.pairs_path = pairs_path
        self.out_paths = out_paths
        self.verbose = verbose
        self.pairs_df = pd.read_csv(pairs_path, dtype=str)
        self.unranked_pairs = self.pairs_df[self.pairs_df['winner'].isnull()]
        self.unranked_pairs = self.unranked_pairs.sample(frac=1).reset_index(drop=True)  # Shuffle pairs
        self.current_pair_index = 0
        self.after_id = None

        self.master.title("Video Labeler")
        self.create_widgets()
        self.bind_keys()
        self.show_pair()

    def create_widgets(self):
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack()

        self.left_video_label = tk.Label(self.video_frame)
        self.left_video_label.pack(side="left")

        self.right_video_label = tk.Label(self.video_frame)
        self.right_video_label.pack(side="right")

        # Create a new frame for progress bars
        self.progress_frame = tk.Frame(self.master)
        self.progress_frame.pack(pady=5)

        # Add progress bars to the new frame
        self.left_progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=200, mode="determinate")
        self.left_progress.pack(side="left", padx=5)

        self.right_progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=200, mode="determinate")
        self.right_progress.pack(side="right", padx=5)

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()

        self.left_button = tk.Button(self.button_frame, text="Left Wins", command=self.left_wins)
        self.left_button.pack(side="left")

        self.right_button = tk.Button(self.button_frame, text="Right Wins", command=self.right_wins)
        self.right_button.pack(side="right")

        # Add quit button frame at the bottom
        self.quit_frame = tk.Frame(self.master)
        self.quit_frame.pack(pady=10, fill=tk.X)
        
        self.quit_button = tk.Button(self.quit_frame, text="Quit", command=self.save_and_exit)
        self.quit_button.pack(side="right", padx=10)

    def bind_keys(self):
        self.master.bind('<Left>', lambda event: self.left_wins())
        self.master.bind('<Right>', lambda event: self.right_wins())

    def show_pair(self):
        if self.current_pair_index < len(self.unranked_pairs):
            pair = self.unranked_pairs.iloc[self.current_pair_index]
            self.left_video_path = os.path.join(self.out_paths['videos'], f"{pair['param1']}.mp4")
            self.right_video_path = os.path.join(self.out_paths['videos'], f"{pair['param2']}.mp4")
            
            # Check if video files exist
            if not os.path.exists(self.left_video_path) or not os.path.exists(self.right_video_path):
                messagebox.showerror("Error", f"Video files not found ({self.left_video_path}, {self.right_video_path}).")
                return

            # Display loading message
            self.left_video_label.config(text="Loading left video...")
            self.right_video_label.config(text="Loading right video...")

            if self.verbose:
                print(f"Loading videos: {self.left_video_path}, {self.right_video_path}")
            self.play_videos()
        else:
            self.prompt_generate_new_pairs()

    def prompt_generate_new_pairs(self):
        if messagebox.askyesno("Generate New Pairs", "No more unranked pairs. Would you like to generate new pairs?"):
            self.generate_new_pairs()

    def generate_new_pairs(self):
        self.simulation.generate_pairs(3, self.out_paths, self.pairs_path, verbose=True)
        self.pairs_df = pd.read_csv(self.pairs_path, dtype=str)
        self.unranked_pairs = self.pairs_df[self.pairs_df['winner'].isnull()]
        self.unranked_pairs = self.unranked_pairs.sample(frac=1).reset_index(drop=True)  # Shuffle pairs
        self.current_pair_index = 0
        self.show_pair()

    def play_videos(self):
        self.left_cap = cv2.VideoCapture(self.left_video_path)
        self.right_cap = cv2.VideoCapture(self.right_video_path)
        self.update_frames()

    def update_frames(self):
        ret_left, frame_left = self.left_cap.read()
        ret_right, frame_right = self.right_cap.read()

        if not ret_left:
            if self.verbose:
                print("Replaying left video.")
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_left, frame_left = self.left_cap.read()

        if not ret_right:
            if self.verbose:
                print("Replaying right video.")
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_right, frame_right = self.right_cap.read()

        if ret_left and ret_right:
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)

            img_left = ImageTk.PhotoImage(Image.fromarray(frame_left))
            img_right = ImageTk.PhotoImage(Image.fromarray(frame_right))

            self.left_video_label.config(image=img_left, text="")
            self.left_video_label.image = img_left

            self.right_video_label.config(image=img_right, text="")
            self.right_video_label.image = img_right

            # Update progress bars
            left_pos = self.left_cap.get(cv2.CAP_PROP_POS_FRAMES)
            left_total = self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.left_progress['value'] = (left_pos / left_total) * 100 if left_total > 0 else 0

            right_pos = self.right_cap.get(cv2.CAP_PROP_POS_FRAMES)
            right_total = self.right_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.right_progress['value'] = (right_pos / right_total) * 100 if right_total > 0 else 0

            if self.after_id is not None:
                self.master.after_cancel(self.after_id)
            self.after_id = self.master.after(30, self.update_frames)
        else:
            if self.verbose:
                print("Finished playing videos or error in reading frames.")
            self.left_cap.release()
            self.right_cap.release()

    def left_wins(self):
        self.record_winner(self.unranked_pairs.iloc[self.current_pair_index]['param1'])

    def right_wins(self):
        self.record_winner(self.unranked_pairs.iloc[self.current_pair_index]['param2'])

    def record_winner(self, winner):
        # Release video resources
        self.left_cap.release()
        self.right_cap.release()

        self.pairs_df.loc[self.unranked_pairs.index[self.current_pair_index], 'winner'] = winner
        self.current_pair_index += 1
        self.show_pair()

    def save_and_exit(self):
        # Release video resources if they exist
        if hasattr(self, 'left_cap'):
            self.left_cap.release()
        if hasattr(self, 'right_cap'):
            self.right_cap.release()
        
        # Save the current state
        self.pairs_df.to_csv(self.pairs_path, index=False)
        
        # Cancel any pending after callbacks
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
        
        # Destroy the window and quit
        self.master.destroy()
        self.master.quit()