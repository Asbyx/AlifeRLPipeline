import os
import tkinter as tk
from tkinter import ttk
from src.utils import Generator, Rewardor, Simulation
import threading
import cv2
from PIL import Image, ImageTk
import shutil

class BenchmarkApp:
    def __init__(self, master, simulation, generator, rewardor, out_paths):
        self.master = master
        self.simulation = simulation
        self.generator = generator
        self.rewardor = rewardor
        self.out_paths = out_paths
        self.scores = []
        self.videos = []
        self.params = []
        self.current_index = 0

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

        self.update_status("Running simulations...")
        outputs = self.simulation.run(self.params)
        self.update_status("Scoring simulations...")
        self.scores = [score.item() for score in self.rewardor.rank(outputs)]
        self.update_status("Sorting videos...")
        self.videos = self.simulation.save_videos(self.params, outputs, self.out_paths['videos'])

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
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_label.config(image=img)
            self.video_label.image = img
            self.master.after(30, self.update_frame)  # Update every 30 ms for smooth playback
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
        self.simulation.save_params([self.params[self.current_index]], self.out_paths['saved_simulations'])

        print(f"Video saved to {self.out_paths['saved_simulations']}")
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
        print("Deleted all videos")

        # Destroy the window
        self.master.destroy()


def launch_benchmarker(simulation, generator, rewardor, out_paths):
    root = tk.Tk()
    app = BenchmarkApp(root, simulation, generator, rewardor, out_paths)
    root.mainloop()
