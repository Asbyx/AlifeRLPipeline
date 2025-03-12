import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from tkinter import ttk
import numpy as np
import threading
from rlhfalife.utils import Simulator

def launch_video_labeler(simulator: Simulator, pairs_path: str, out_paths: dict, verbose: bool = False) -> None:
    """
    Launch the video labeler app.
    
    Args:
        simulator: Simulator object that will be used for generating pairs
        pairs_path: Path to the pairs CSV file
        out_paths: Dictionary containing output paths
        verbose: Whether to print verbose output
    
    Note: executes in the main thread.
    """
    root = tk.Tk()
    app = VideoLabelerApp(root, simulator, pairs_path, out_paths, verbose)
    root.protocol("WM_DELETE_WINDOW", app.save_and_exit)
    root.mainloop()

class VideoLabelerApp:
    def __init__(self, master: tk.Tk, simulator: Simulator, pairs_path: str, out_paths: dict, verbose: bool = False) -> None:
        """
        Initialize the video labeler app

        Args:
            master: The master window
            simulator: The simulator to use
            pairs_path: The path to the pairs CSV file
            out_paths: The paths to the outputs
            verbose: Whether to print verbose output
        """
        self.simulator = simulator
        self.pairs_path = pairs_path
        self.out_paths = out_paths
        self.verbose = verbose
        
        self.master = master
        self.load_pairs()
        self.after_id = None

        self.master.title("Video Labeler")
        self.create_widgets()
        self.bind_keys()
        self.master.lift()
        self.master.focus_force()
        self.master.attributes('-topmost', True)
        self.master.attributes('-topmost', False)
        self.update_progress_percentage()
        self.show_pair()

    def load_pairs(self):
        """Load pairs from CSV file, filter unranked pairs, and shuffle them."""
        # Load all pairs from CSV
        self.pairs_df = pd.read_csv(self.pairs_path, dtype=str)
        
        # Filter unranked pairs (those with null winner)
        self.unranked_pairs = self.pairs_df[self.pairs_df['winner'].isnull()].copy()
        
        # If there are no unranked pairs, create new ones
        if len(self.unranked_pairs) == 0:
            unique_param1 = self.pairs_df['param1'].unique() if 'param1' in self.pairs_df.columns else []
            unique_param2 = self.pairs_df['param2'].unique() if 'param2' in self.pairs_df.columns else []
            
            if len(unique_param1) > 0 and len(unique_param2) > 0:
                param1_values = np.tile(unique_param1, len(unique_param2))
                param2_values = np.repeat(unique_param2, len(unique_param1))
                
                # Shuffle param2 values while maintaining the structure
                param2_indices = np.arange(len(param2_values))
                np.random.shuffle(param2_indices)
                shuffled_param2 = param2_values[param2_indices]
                
                new_pairs = pd.DataFrame({
                    'param1': param1_values,
                    'param2': shuffled_param2,
                    'winner': np.nan
                })
                
                # get rid of pairs that are the same
                new_pairs = new_pairs[new_pairs['param1'] != new_pairs['param2']]
                new_pairs = new_pairs.drop_duplicates(subset=['param1', 'param2'])
                
                # Append new pairs to the existing pairs_df
                self.pairs_df = pd.concat([self.pairs_df, new_pairs], ignore_index=True)
                
                # Update unranked_pairs to include the new pairs
                self.unranked_pairs = self.pairs_df[self.pairs_df['winner'].isnull()].copy()
        
        # Shuffle the unranked pairs for randomized presentation
        self.unranked_pairs = self.unranked_pairs.sample(frac=1).reset_index(drop=True)
        self.current_pair_index = 0

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
        
        # Add progress percentage label
        self.progress_label = tk.Label(self.quit_frame, text="0% Ranked")
        self.progress_label.pack(side="left", padx=10)
        
        self.quit_button = tk.Button(self.quit_frame, text="Quit", command=self.save_and_exit)
        self.quit_button.pack(side="right", padx=10)

        # Create a new frame for keybindings
        self.keybindings_frame = tk.Frame(self.master)
        self.keybindings_frame.pack(side="left", padx=10, pady=10)

        # Add a label to display keybindings
        self.keybindings_label = tk.Label(self.keybindings_frame, text="Keybindings:\nLeft Arrow: Left Wins\nRight Arrow: Right Wins\nSpace: Restart Videos\nBackspace: Previous Pair", justify="left")
        self.keybindings_label.pack()

    def bind_keys(self):
        self.master.bind('<Left>', lambda event: self.left_wins())
        self.master.bind('<Right>', lambda event: self.right_wins())
        self.master.bind('<space>', lambda event: self.restart_videos())
        self.master.bind('<BackSpace>', lambda event: self.previous_pair())

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
        if messagebox.askyesno("Generate New Pairs", "No more unranked pairs. Would you like to generate new simulators?"):
            num_pairs = simpledialog.askinteger("Input", "How many simulators do you want to generate?", minvalue=1)
            if num_pairs is not None:
                self.generate_new_pairs(num_pairs)
        else:
            self.save_and_exit()

    def generate_new_pairs(self, num_pairs):
        """Generate new pairs of simulators"""
        self.pairs_df.to_csv(self.pairs_path, index=False)

        loading_screen = LoadingScreen(self.master)
        loading_screen.run_generation_process(
            self.simulator, 
            num_pairs, 
            self.out_paths, 
            self.pairs_path, 
            self.verbose,
            on_complete=self.on_generation_complete,
            on_error=self.on_generation_error
        )

    def on_generation_complete(self, loading_screen):
        """Called when generation is complete"""
        loading_screen.close()
        self.load_pairs()
        self.update_progress_percentage()
        self.show_pair()
        
    def on_generation_error(self, loading_screen, error_message):
        """Handle errors during generation"""
        loading_screen.close()
        messagebox.showerror("Error", f"An error occurred during pair generation:\n{error_message}")

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
        self.record_winner(str(self.unranked_pairs.iloc[self.current_pair_index]['param1']))

    def right_wins(self):
        self.record_winner(str(self.unranked_pairs.iloc[self.current_pair_index]['param2']))

    def record_winner(self, winner):
        # Release video resources
        self.left_cap.release()
        self.right_cap.release()
        
        # Get the current pair
        current_pair = self.unranked_pairs.iloc[self.current_pair_index]
        
        # Find the corresponding row in pairs_df that matches the current pair
        mask = (self.pairs_df['param1'] == current_pair['param1']) & \
               (self.pairs_df['param2'] == current_pair['param2']) & \
               (self.pairs_df['winner'].isnull())
        
        # Update the winner in pairs_df
        self.pairs_df.loc[mask, 'winner'] = winner
        
        # Move to the next pair
        self.current_pair_index += 1
        self.update_progress_percentage()
        self.show_pair()
        
    def update_progress_percentage(self):
        """Update the progress percentage label based on ranked pairs."""
        total_pairs = len(self.pairs_df)
        ranked_pairs = len(self.pairs_df[self.pairs_df['winner'].notnull()])
        percentage = int((ranked_pairs / total_pairs) * 100) if total_pairs > 0 else 0
        self.progress_label.config(text=f"{percentage}% Ranked")
        
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

    def restart_videos(self):
        if hasattr(self, 'left_cap') and hasattr(self, 'right_cap'):
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.update_frames()

    def previous_pair(self):
        if self.current_pair_index > 0:
            self.current_pair_index -= 1
            self.update_progress_percentage()
            self.show_pair()

class LoadingScreen:
    def __init__(self, master, title="Generating Pairs"):
        self.window = tk.Toplevel(master)
        self.window.title(title)
        self.window.geometry("300x200")  # Slightly taller to accommodate the animation
        self.window.resizable(False, False)
        
        # Make it modal
        self.window.transient(master)
        self.window.grab_set()
        
        # Center the window
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Create a frame for the content
        self.frame = ttk.Frame(self.window, padding=20)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a simple animation (spinning dots)
        self.animation_frame = ttk.Frame(self.frame)
        self.animation_frame.pack(pady=(0, 10))
        
        self.dots = []
        for i in range(5):
            dot = ttk.Label(self.animation_frame, text="●", font=("Arial", 14))
            dot.grid(row=0, column=i, padx=3)
            self.dots.append(dot)
        
        # Add a label for the status
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var)
        self.status_label.pack(pady=(0, 10))
        
        # Add a progress bar
        self.progress = ttk.Progressbar(self.frame, orient=tk.HORIZONTAL, length=250, mode="indeterminate")
        self.progress.pack(pady=10)
        self.progress.start(10)
        
        self.total_steps = None
        self.current_step = 0
        self.master = master
        
        # Start the animation
        self.animation_index = 0
        self.animate()
        
    def animate(self):
        """Animate the dots"""
        if not hasattr(self, 'window') or not self.window.winfo_exists():
            return  # Stop animation if window is closed
            
        # Reset all dots to default color
        for dot in self.dots:
            dot.configure(foreground="gray")
            
        # Highlight the current dot
        self.dots[self.animation_index].configure(foreground="blue")
        
        # Move to the next dot
        self.animation_index = (self.animation_index + 1) % len(self.dots)
        
        # Schedule the next animation frame
        self.animation_id = self.window.after(200, self.animate)
        
    def set_determinate_mode(self, total_steps):
        """Switch to determinate mode with a known number of steps"""
        self.total_steps = total_steps
        self.current_step = 0
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=total_steps, value=0)
        
    def increment_progress(self):
        """Increment the progress bar by one step"""
        if self.total_steps is not None:
            self.current_step += 1
            self.progress.configure(value=self.current_step)
            self.window.update()
        
    def update_status(self, message):
        """Update the status message in the loading screen"""
        self.status_var.set(message)
        self.window.update()
        
    def close(self):
        """Close the loading screen"""
        if hasattr(self, 'animation_id'):
            self.window.after_cancel(self.animation_id)
        self.window.grab_release()
        self.window.destroy()
        
    def run_generation_process(self, simulator, num_pairs, out_paths, pairs_path, verbose=False, on_complete=None, on_error=None):
        """
        Run the generation process in a separate thread with progress updates
        
        Args:
            simulator: The simulator object
            num_pairs: Number of pairs to generate
            out_paths: Dictionary of output paths
            pairs_path: Path to the pairs CSV file
            verbose: Whether to print verbose output
            on_complete: Callback function to call when generation completes successfully
            on_error: Callback function to call when an error occurs
        """
        # Set to determinate mode with known number of steps
        # We have approximately 7 main steps in the generation process
        self.set_determinate_mode(10)
        
        # Define a callback function to update the loading screen
        def update_callback(message):
            self.update_status(message)
            self.increment_progress()
            
        # Run the generation in a separate thread
        def run_generation():
            try:
                success = simulator.generate_pairs(
                    num_pairs, 
                    out_paths, 
                    pairs_path, 
                    verbose=verbose,
                    progress_callback=update_callback
                )
                
                # Update the UI in the main thread
                if success:
                    if on_complete:
                        self.master.after(0, lambda: on_complete(self))
                else:
                    self.master.after(0, lambda: self.close())
            except Exception as e:
                # Handle any exceptions
                if on_error:
                    self.master.after(0, lambda: on_error(self, str(e)))
        
        # Start the generation thread
        threading.Thread(target=run_generation, daemon=True).start()