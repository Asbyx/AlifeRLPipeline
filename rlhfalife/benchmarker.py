import os
import tkinter as tk
from tkinter import messagebox
from rlhfalife.utils import Generator, Rewarder, Simulator
import threading
import cv2
from PIL import Image, ImageTk
import shutil
import numpy as np

class BenchmarkApp:
    def __init__(self, master: tk.Tk, simulator: Simulator, generator: Generator, rewarder: Rewarder, out_paths: dict) -> None:
        """
        Initialize the benchmarker

        Args:
            master: The master window
            simulator: The simulator to use
            generator: The generator to use
            rewarder: The rewarder to use
            out_paths: The paths to the outputs
        """
        self.master = master
        self.simulator = simulator
        self.generator = generator
        self.rewarder = rewarder
        self.out_paths = out_paths
        self.scores = []
        self.videos = []
        self.params = []
        self.current_index = 0
        self.after_id = None

        self.master.title("Benchmarking App")
        self.create_widgets()
        self.run_benchmark()
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.lift()
        self.master.focus_force()
        self.master.attributes('-topmost', True)
        self.master.attributes('-topmost', False)

        self.master.bind('<space>', lambda event: self.restart_video())
        self.master.bind('<Left>', lambda event: self.show_previous())
        self.master.bind('<Right>', lambda event: self.show_next())

    def create_widgets(self):
        self.status_label = tk.Label(self.master, text="Initializing...")
        self.status_label.pack(pady=10)

        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack()

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.score_label = tk.Label(self.master, text="Score: ")
        self.score_label.pack(pady=5)

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=10)

        self.save_button = tk.Button(self.button_frame, text="Save Video & Params", command=self.save_video)
        self.save_button.pack(side="left", padx=5)

        self.prev_button = tk.Button(self.button_frame, text="Previous", command=self.show_previous)
        self.prev_button.pack(side="left", padx=5)

        self.next_button = tk.Button(self.button_frame, text="Next", command=self.show_next)
        self.next_button.pack(side="left", padx=5)

        self.restart_button = tk.Button(self.button_frame, text="Restart", command=self.restart_video)
        self.restart_button.pack(side="left", padx=5)

    def run_benchmark(self):
        # Start the benchmarking process in a new thread
        threading.Thread(target=self.benchmark_process).start()

    def benchmark_process(self):
        self.update_status("Generating parameters...")
        self.params = self.generator.generate(10)

        # check if two params are the same
        if any(np.array_equal(self.params[i], self.params[j]) for i in range(len(self.params)) for j in range(i+1, len(self.params))):
            print("\n" + "="*50)
            print("!!! WARNING !!!: Generator generated at least two identical parameters.")
            # filter out the identical parameters
            self.params = [self.params[i] for i in range(len(self.params)) if not any(np.array_equal(self.params[i], self.params[j]) for j in range(i+1, len(self.params)))]
            print(f"Unique parameters: {len(self.params)}, over 10 generated.")
            print("="*50 + "\n")

        self.update_status("Running simulators...")
        outputs = self.simulator.run(self.params)
        
        self.update_status("Scoring simulators...")
        self.scores = self.rewarder.rank(outputs)

        self.update_status("Saving videos...")
        self.videos = self.simulator.save_videos(self.generator.hash_params(self.params), outputs, self.out_paths['videos'])

        # Sort videos by score
        sorted_videos = sorted(zip(self.scores, self.videos, self.params), key=lambda x: x[0], reverse=True)
        self.scores, self.videos, self.params = zip(*sorted_videos)

        self.update_status("Videos (best to worst):")
        self.master.after(0, lambda: self.show_video(0))

    def update_status(self, message):
        self.master.after(0, lambda: self.status_label.config(text=message))

    def show_video(self, index):
        self.current_index = index
        self.cap = cv2.VideoCapture(self.videos[index])
        self.update_frame()
        self.score_label.config(text=f"Score: {self.scores[index]}")

    def update_frame(self):
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
            self.after_id = None

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to 300x300 to match the labeler.py dimensions
            frame = cv2.resize(frame, (300, 300))
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_label.config(image=img)
            self.video_label.image = img
            self.after_id = self.master.after(30, self.update_frame)  # Update every 30 ms for smooth playback
        else:
            self.cap.release()

    def show_next(self):
        if self.current_index < len(self.videos) - 1:
            self.cap.release()
            self.show_video(self.current_index + 1)

    def show_previous(self):
        if self.current_index > 0:
            self.cap.release()
            self.show_video(self.current_index - 1)

    def save_video(self):
        # Save the video (duplicate the file) and params
        shutil.copy(self.videos[self.current_index], self.out_paths['saved_simulations'])
        self.simulator.save_param(self.params[self.current_index], os.path.join(self.out_paths['saved_simulations'], str(self.generator.hash_params([self.params[self.current_index]])[0])))

        print(f"Video saved to {self.out_paths['saved_simulations']}.")
        self.update_status(f"Video and its parameters saved to {self.out_paths['saved_simulations']} !")

    def restart_video(self):
        self.cap.release()
        self.show_video(self.current_index)

    def on_close(self):
        # Release video resources
        if hasattr(self, 'cap'):
            self.cap.release()

        # Delete video files
        for video_path in self.videos:
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Error deleting {video_path}: {e}")
        print("Deleted all videos from the benchmark.")

        # Ask if user wants to save the generator and rewarder
        save_generator = messagebox.askyesno("Save Generator", "Do you want to save the generator?")
        save_rewarder = messagebox.askyesno("Save Rewarder", "Do you want to save the rewarder?")

        if save_generator:
            self.generator.save()   
        if save_rewarder:
            self.rewarder.save()

        # Destroy the window
        self.master.destroy()


def launch_benchmarker(simulator: Simulator, generator: Generator, rewarder: Rewarder, out_paths: dict) -> None:
    """
    Launch the benchmarker

    Args:
        simulator: The simulator to use
        generator: The generator to use
        rewarder: The rewarder to use
        out_paths: The paths to the outputs
    """
    root = tk.Tk()
    app = BenchmarkApp(root, simulator, generator, rewarder, out_paths)
    root.mainloop()
