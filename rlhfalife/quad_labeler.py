import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from tkinter import ttk
import numpy as np
import threading
import itertools
from rlhfalife.utils import Simulator
from rlhfalife.data_managers import DatasetManager, PairsManager

def launch_quad_labeler(simulator: Simulator, dataset_manager: DatasetManager, pairs_manager: PairsManager, verbose: bool = False, frame_size: tuple = (300, 300)) -> None:
    """
    Launch the quad labeler app that shows 4 videos for ranking.
    
    Args:
        simulator: Simulator object that will be used for generating quadruples
        dataset_manager: DatasetManager instance for storing simulation data
        pairs_manager: PairsManager instance for storing pairs
        verbose: Whether to print verbose output
        frame_size: Tuple of (width, height) for video frames
    
    Note: executes in the main thread.
    """
    root = tk.Tk()
    app = QuadLabelerApp(root, simulator, dataset_manager, pairs_manager, verbose, frame_size)
    root.protocol("WM_DELETE_WINDOW", app.save_and_exit)
    root.mainloop()

class DraggableVideo(tk.Frame):
    """A draggable video widget that can be reordered via drag and drop."""
    
    def __init__(self, parent, video_path, index, frame_size, hash_value, on_drag_start=None, on_drag_release=None, on_drag_motion=None):
        """
        Initialize a draggable video widget.
        
        Args:
            parent: The parent widget
            video_path: Path to the video file
            index: The index of this video (0-3)
            frame_size: Size of the video frame (width, height)
            hash_value: Hash value of the video for display
            on_drag_start: Callback function when drag starts
            on_drag_release: Callback function when drag ends
            on_drag_motion: Callback function when dragging
        """
        super().__init__(parent, relief=tk.RAISED, borderwidth=2)
        self.parent = parent
        self.index = index
        self.frame_size = frame_size
        self.video_path = video_path
        self.hash_value = hash_value
        self.on_drag_start = on_drag_start
        self.on_drag_release = on_drag_release
        self.on_drag_motion = on_drag_motion
        self.is_dragging = False
        
        # Create the video label
        self.video_label = tk.Label(self)
        self.video_label.pack(padx=5, pady=5)
        
        # Create hash label
        self.hash_label = tk.Label(self, text=f"Hash: {hash_value}", font=("Arial", 8), wraplength=frame_size[0]-10)
        self.hash_label.pack(pady=(0, 5))
        
        # Create ranking indicator
        self.rank_label = tk.Label(self, text=f"#{index+1}", font=("Arial", 14, "bold"))
        self.rank_label.pack(pady=5)
        
        # Open the video
        self.cap = cv2.VideoCapture(video_path)
        
        # Bind events for dragging
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<B1-Motion>", self.on_motion)
        self.video_label.bind("<ButtonPress-1>", self.on_press)
        self.video_label.bind("<ButtonRelease-1>", self.on_release)
        self.video_label.bind("<B1-Motion>", self.on_motion)
        self.rank_label.bind("<ButtonPress-1>", self.on_press)
        self.rank_label.bind("<ButtonRelease-1>", self.on_release)
        self.rank_label.bind("<B1-Motion>", self.on_motion)
        self.hash_label.bind("<ButtonPress-1>", self.on_press)
        self.hash_label.bind("<ButtonRelease-1>", self.on_release)
        self.hash_label.bind("<B1-Motion>", self.on_motion)
        
        # Track mouse offset for dragging
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # For drag and drop feedback
        self.orig_background = self.cget("background")
        
        # Start playing the video
        self.after_id = None
        self.update_frame()
    
    def update_frame(self):
        """Update the video frame."""
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        if ret:
            # Convert and resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            
            # Convert to PIL Image and then to PhotoImage
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            # Schedule next update
            self.after_id = self.after(33, self.update_frame)  # ~30 fps
    
    def update_rank(self, new_index):
        """Update the rank label to show new position."""
        self.index = new_index
        self.rank_label.config(text=f"#{new_index+1}")
    
    def on_press(self, event):
        """Handle mouse button press event."""
        self.is_dragging = True
        # Record the offset of the mouse click relative to the widget
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        if self.on_drag_start:
            self.on_drag_start(self)
        
        # Change appearance to indicate dragging
        self.config(relief=tk.SUNKEN, background="#ddddff")  # Highlight with light blue background
    
    def on_motion(self, event):
        """Handle mouse motion event."""
        if not self.is_dragging:
            return
            
        # Get mouse position relative to the screen
        x_root = event.x_root
        y_root = event.y_root
        
        # Call the motion callback if it exists
        if self.on_drag_motion:
            self.on_drag_motion(self, x_root, y_root)
    
    def on_release(self, event):
        """Handle mouse button release event."""
        if self.is_dragging:
            self.is_dragging = False
            # Reset appearance
            self.config(relief=tk.RAISED, background=self.orig_background)
            
            if self.on_drag_release:
                self.on_drag_release(self, event)
    
    def release_resources(self):
        """Release video resources."""
        if self.after_id:
            self.after_cancel(self.after_id)
        if self.cap and self.cap.isOpened():
            self.cap.release()

class QuadLabelerApp:
    def __init__(self, master: tk.Tk, simulator: Simulator, dataset_manager: DatasetManager, 
                 pairs_manager: PairsManager, verbose: bool = False, frame_size: tuple = (300, 300)) -> None:
        """
        Initialize the quad labeler app.
        
        Args:
            master: The master window
            simulator: The simulator to use
            dataset_manager: DatasetManager instance for storing simulation data
            pairs_manager: PairsManager instance for storing pairs
            verbose: Whether to print verbose output
            frame_size: Tuple of (width, height) for video frames
        """
        self.simulator = simulator
        self.dataset_manager = dataset_manager
        self.pairs_manager = pairs_manager
        self.verbose = verbose
        self.frame_size = frame_size
        
        self.master = master
        self.master.title("Quad Video Labeler")
        
        # Store currently displayed videos
        self.current_videos = []
        self.current_hashes = []
        self.video_widgets = []
        
        # Create widgets
        self.create_widgets()
        self.bind_keys()
        
        # Set window properties
        self.master.lift()
        self.master.focus_force()
        self.master.attributes('-topmost', True)
        self.master.attributes('-topmost', False)
        
        # Track dragging state
        self.drag_source = None
        self.drag_over_widget = None
        self.highlight_widget = None
        
        # Update UI and load videos
        self.update_progress_percentage()
        self.load_next_videos()
    
    def create_widgets(self):
        """Create the main UI widgets."""
        # Create title and instructions
        self.title_frame = tk.Frame(self.master)
        self.title_frame.pack(pady=10)
        
        title_label = tk.Label(self.title_frame, 
                              text="Drag videos to rank them in order (best to worst)",
                              font=("Arial", 14))
        title_label.pack()
        
        # Create main frame for videos
        self.videos_frame = tk.Frame(self.master)
        self.videos_frame.pack(padx=20, pady=10)
        
        # Create frame for action buttons
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=15)
        
        # Submit button
        self.submit_button = tk.Button(self.button_frame, text="Submit Ranking", 
                                      command=self.submit_ranking, padx=10, pady=5,
                                      font=("Arial", 12, "bold"))
        self.submit_button.pack(side=tk.LEFT, padx=5)
        
        # Add restart videos button
        self.restart_button = tk.Button(self.button_frame, text="Restart Videos", 
                                      command=self.restart_videos, padx=10, pady=5)
        self.restart_button.pack(side=tk.LEFT, padx=5)
        
        # Add generate new videos button
        self.generate_button = tk.Button(self.button_frame, text="Generate New Videos", 
                                        command=self.generate_new_videos_dialog, padx=10, pady=5)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        
        # Create bottom frame for status and quit
        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Add progress percentage label
        self.progress_label = tk.Label(self.bottom_frame, text="Progress: 0% (0/0)")
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        # Add quit button
        self.quit_button = tk.Button(self.bottom_frame, text="Quit", 
                                    command=self.save_and_exit, padx=5, pady=2)
        self.quit_button.pack(side=tk.RIGHT, padx=10)
        
        # Create frame for keybindings
        self.key_frame = tk.Frame(self.master)
        self.key_frame.pack(side=tk.BOTTOM, pady=5)
        
        # Add keybindings label
        key_text = "Keybindings:\nEnter: Submit Ranking\nSpace: Restart Videos\nEsc: Quit"
        self.key_label = tk.Label(self.key_frame, text=key_text, justify=tk.LEFT)
        self.key_label.pack()
    
    def bind_keys(self):
        """Bind keyboard shortcuts."""
        self.master.bind('<Return>', lambda event: self.submit_ranking())
        self.master.bind('<space>', lambda event: self.restart_videos())
        self.master.bind('<Escape>', lambda event: self.save_and_exit())
    
    def load_next_videos(self):
        """Load the next set of 4 videos."""
        # Get 4 random unranked hashes
        all_hashes = self.dataset_manager.get_all_hashes()
        
        if len(all_hashes) < 4:
            self.prompt_generate_new_videos()
            return
        
        # Try to find 4 hashes that haven't been ranked together before
        # This is a simple approach - more sophisticated selection possible
        sample_size = min(20, len(all_hashes))  # Limit search space for efficiency
        candidate_hashes = np.random.choice(all_hashes, sample_size, replace=False)
        
        # Choose 4 hashes that have the fewest pairs already ranked between them
        best_hashes = self.find_best_hashes(candidate_hashes)
        
        if not best_hashes or len(best_hashes) < 4:
            # If we couldn't find a good set, just take 4 random ones
            self.current_hashes = np.random.choice(all_hashes, 4, replace=False)
        else:
            self.current_hashes = best_hashes
        
        # Clear any existing video widgets
        self.clear_video_widgets()
        
        # Get the video paths
        video_paths = self.dataset_manager.get_video_paths(self.current_hashes)
        
        # Check if all videos exist
        if None in video_paths or not all(video_paths):
            messagebox.showerror("Error", "One or more video files not found")
            self.load_next_videos()
            return
        
        # Randomly permute the videos for initial display
        indices = np.random.permutation(4)
        self.current_hashes = [self.current_hashes[i] for i in indices]
        video_paths = [video_paths[i] for i in indices]
        
        # Create the video widgets
        self.create_video_widgets(video_paths)
    
    def find_best_hashes(self, candidate_hashes):
        """Find 4 hashes with minimal existing ranked pairs between them."""
        if len(candidate_hashes) < 4:
            return None
        
        # Get all possible sets of 4 hashes
        hash_combinations = list(itertools.combinations(candidate_hashes, 4))
        
        if not hash_combinations:
            return None
        
        # Choose a random subset to check (for efficiency)
        subset_size = min(10, len(hash_combinations))
        # Fix: np.random.choice can't handle list of tuples directly
        # Use random indices instead
        random_indices = np.random.choice(len(hash_combinations), subset_size, replace=False)
        hash_combinations_subset = [hash_combinations[i] for i in random_indices]
        
        best_combination = None
        min_ranked_pairs = float('inf')
        
        for hashes in hash_combinations_subset:
            # Count how many pairs among these 4 are already ranked
            pairs = list(itertools.combinations(hashes, 2))
            ranked_count = 0
            
            for h1, h2 in pairs:
                # Check if this pair exists in ranked pairs
                pair_exists = self.pairs_manager._get_ranked_pairs()[
                    ((self.pairs_manager._get_ranked_pairs()['hash1'] == h1) & 
                     (self.pairs_manager._get_ranked_pairs()['hash2'] == h2)) |
                    ((self.pairs_manager._get_ranked_pairs()['hash1'] == h2) & 
                     (self.pairs_manager._get_ranked_pairs()['hash2'] == h1))
                ].shape[0] > 0
                
                if pair_exists:
                    ranked_count += 1
            
            if ranked_count < min_ranked_pairs:
                min_ranked_pairs = ranked_count
                best_combination = hashes
                
                # If we found a combination with no ranked pairs, return immediately
                if min_ranked_pairs == 0:
                    break
        
        return best_combination
    
    def create_video_widgets(self, video_paths):
        """Create the draggable video widgets."""
        # Side by side layout for all 4 videos
        for i, path in enumerate(video_paths):
            # Create the draggable video widget
            video_widget = DraggableVideo(
                self.videos_frame, 
                path, 
                i, 
                self.frame_size,
                hash_value=self.current_hashes[i],  # Pass the hash value
                on_drag_start=self.start_drag,
                on_drag_release=self.end_drag,
                on_drag_motion=self.on_drag_motion
            )
            video_widget.pack(side=tk.LEFT, padx=5, pady=10)
            
            # Store the widget
            self.video_widgets.append(video_widget)
    
    def clear_video_widgets(self):
        """Clear all video widgets and release resources."""
        for widget in self.video_widgets:
            widget.release_resources()
            widget.destroy()
        
        self.video_widgets = []
    
    def start_drag(self, source_widget):
        """Handle the start of a drag operation."""
        self.drag_source = source_widget
        self.highlight_widget = None
    
    def on_drag_motion(self, source_widget, x, y):
        """Handle drag motion to provide visual feedback."""
        # Find which widget is under the mouse cursor
        target_widget = self.find_widget_at_position(x, y)
        
        # If we've moved to a different widget, update highlighting
        if target_widget != self.highlight_widget and target_widget != source_widget:
            # Remove previous highlighting
            if self.highlight_widget and self.highlight_widget != self.drag_source:
                self.highlight_widget.config(relief=tk.RAISED, background=self.highlight_widget.orig_background)
            
            # Add highlighting to new widget
            if target_widget and target_widget != self.drag_source:
                target_widget.config(relief=tk.RIDGE, background="#ddffdd")  # Light green highlight
                
            self.highlight_widget = target_widget
    
    def find_widget_at_position(self, x, y):
        """Find which video widget is at the given screen position."""
        for widget in self.video_widgets:
            widget_x = widget.winfo_rootx()
            widget_y = widget.winfo_rooty()
            widget_width = widget.winfo_width()
            widget_height = widget.winfo_height()
            
            if (widget_x <= x <= widget_x + widget_width and 
                widget_y <= y <= widget_y + widget_height):
                return widget
        
        return None
    
    def end_drag(self, source_widget, event):
        """Handle the end of a drag operation."""
        # Find which widget we're over
        x, y = event.x_root, event.y_root
        target_widget = self.find_widget_at_position(x, y)
        
        # Clean up any highlighting
        if self.highlight_widget and self.highlight_widget != self.drag_source:
            self.highlight_widget.config(relief=tk.RAISED, background=self.highlight_widget.orig_background)
        
        self.highlight_widget = None
        
        if target_widget and target_widget != source_widget:
            # Swap the widgets in the display
            self.swap_videos(source_widget, target_widget)
    
    def swap_videos(self, widget1, widget2):
        """Swap two video widgets in the display."""
        # Get their current indices
        index1 = self.video_widgets.index(widget1)
        index2 = self.video_widgets.index(widget2)
        
        # Swap the widgets in the list
        self.video_widgets[index1], self.video_widgets[index2] = self.video_widgets[index2], self.video_widgets[index1]
        
        # Swap the hashes
        self.current_hashes[index1], self.current_hashes[index2] = self.current_hashes[index2], self.current_hashes[index1]
        
        # Update their visual rank indicators
        widget1.update_rank(index2)
        widget2.update_rank(index1)
        
        # Rearrange in the display - unpack all then repack in order
        for widget in self.video_widgets:
            widget.pack_forget()
        
        for widget in self.video_widgets:
            widget.pack(side=tk.LEFT, padx=5, pady=10)
    
    def submit_ranking(self):
        """Process the current video ranking and generate pairs."""
        if len(self.current_hashes) != 4:
            messagebox.showerror("Error", "Need exactly 4 videos to rank")
            return
        
        # Generate all pairs and record the winners based on ranking
        # For each pair (i,j), the winner is the one with lower index
        print(self.current_hashes)
        for i in range(4):
            hash_i = self.current_hashes[i]
            for j in range(i+1, 4):
                hash_j = self.current_hashes[j]
                self.pairs_manager.set_winner(hash_i, hash_j, 0)  # 0 means the first hash wins
        
        # Load the next set of videos
        self.load_next_videos()
        self.update_progress_percentage()
    
    def restart_videos(self):
        """Restart all videos from the beginning."""
        for widget in self.video_widgets:
            if widget.cap and widget.cap.isOpened():
                widget.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def prompt_generate_new_videos(self):
        """Prompt user to generate new videos if not enough available."""
        num_videos = simpledialog.askinteger("Generate Videos", 
                                           "Not enough videos available. How many new videos to generate?", 
                                           minvalue=4)
        if num_videos:
            self.generate_new_videos(num_videos)
    
    def generate_new_videos_dialog(self):
        """Prompt for generating new videos."""
        num_videos = simpledialog.askinteger("Generate Videos", 
                                           "How many new videos would you like to generate?", 
                                           minvalue=4, initialvalue=4)
        if num_videos:
            self.generate_new_videos(num_videos)
    
    def generate_new_videos(self, num_videos):
        """Generate new video simulations."""
        loading_screen = LoadingScreen(self.master)
        loading_screen.run_generation_process(
            self.simulator, 
            num_videos, 
            self.dataset_manager, 
            self.pairs_manager,
            self.verbose,
            on_complete=lambda: self.on_generation_complete(loading_screen),
            on_error=lambda: self.on_generation_error(loading_screen)
        )
    
    def on_generation_complete(self, loading_screen):
        """Called when generation is complete."""
        loading_screen.close()
        self.update_progress_percentage()
        self.load_next_videos()
    
    def on_generation_error(self, loading_screen):
        """Handle errors during generation."""
        loading_screen.close()
        # Error dialog is handled in the loading screen
    
    def update_progress_percentage(self):
        """Update the progress percentage label."""
        nb_pairs = self.pairs_manager.get_nb_pairs()
        nb_ranked_pairs = self.pairs_manager.get_nb_ranked_pairs()
        self.progress_label.config(
            text=f"Progress: {((nb_ranked_pairs) / nb_pairs) * 100 if nb_pairs > 0 else 0:.1f}% "
                 f"({nb_ranked_pairs}/{nb_pairs}). Number of simulations: {len(self.dataset_manager)}"
        )
    
    def save_and_exit(self):
        """Save data and exit the application."""
        # Release video resources
        self.clear_video_widgets()
        
        # Save managers
        self.pairs_manager.save()
        self.dataset_manager.save()
        
        # Destroy window and quit
        self.master.destroy()
        self.master.quit()

class LoadingScreen:
    def __init__(self, master, title="Generating Videos"):
        self.window = tk.Toplevel(master)
        self.window.title(title)
        self.window.geometry("300x200")
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
        
    def update_status(self, message):
        """Update the status message."""
        self.status_var.set(message)
        self.window.update()
        
    def close(self):
        """Close the loading screen."""
        if hasattr(self, 'animation_id'):
            self.window.after_cancel(self.animation_id)
        self.window.grab_release()
        self.window.destroy()
        
    def run_generation_process(self, simulator, num_pairs, dataset_manager, pairs_manager, 
                              verbose=False, on_complete=None, on_error=None):
        """
        Run the generation process in a separate thread.
        """
        self.update_status("Initializing...")
        
        def update_callback(message):
            self.update_status(message)
            return True  # Continue processing
        
        def run_generation():
            try:
                simulator.generate_pairs(
                    num_pairs, 
                    dataset_manager, 
                    pairs_manager, 
                    verbose=verbose, 
                    progress_callback=update_callback
                )
                if on_complete:
                    self.master.after(0, on_complete)
            except Exception as e:
                import traceback
                import sys
                
                # Get the full traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                
                # Get simulator state information
                simulator_info = f"Simulator Configuration:\n"
                simulator_info += f"- Type: {type(simulator).__name__}\n"
                for attr in dir(simulator):
                    if not attr.startswith('_'):  # Only show public attributes
                        try:
                            value = getattr(simulator, attr)
                            if not callable(value):  # Skip methods
                                simulator_info += f"- {attr}: {value}\n"
                        except Exception:
                            pass  # Skip attributes that can't be accessed
                
                # Get dataset and pairs manager state
                dataset_info = f"\nDataset Manager State:\n"
                dataset_info += f"- Total simulations: {len(dataset_manager)}\n"
                pairs_info = f"\nPairs Manager State:\n"
                pairs_info += f"- Total pairs: {pairs_manager.get_nb_pairs()}\n"
                pairs_info += f"- Ranked pairs: {pairs_manager.get_nb_ranked_pairs()}\n"
                
                # Combine all debug information
                debug_info = (
                    f"Error Details:\n"
                    f"{'='*50}\n"
                    f"{str(e)}\n\n"
                    f"Full Traceback:\n"
                    f"{'='*50}\n"
                    f"{tb_str}\n\n"
                    f"State Information:\n"
                    f"{'='*50}\n"
                    f"{simulator_info}\n"
                    f"{dataset_info}\n"
                    f"{pairs_info}"
                )
                
                if on_error:
                    self.master.after(0, lambda: self.show_error_dialog(debug_info))
                    self.master.after(0, lambda: on_error())
        
        threading.Thread(target=run_generation).start()
    
    def show_error_dialog(self, debug_info):
        """Show a detailed error dialog with debug information."""
        error_window = tk.Toplevel(self.master)
        error_window.title("Error Details")
        error_window.geometry("800x600")
        
        # Make it modal
        error_window.transient(self.master)
        error_window.grab_set()
        
        # Add a frame for the content
        frame = ttk.Frame(error_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a label at the top
        ttk.Label(frame, text="An error occurred during video generation. Details below:", 
                 wraplength=780).pack(pady=(0, 10))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_widget.yview)
        
        # Insert the debug information
        text_widget.insert(tk.END, debug_info)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Add buttons at the bottom
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        
        # Copy button
        def copy_to_clipboard():
            error_window.clipboard_clear()
            error_window.clipboard_append(debug_info)
            error_window.update()
        
        ttk.Button(button_frame, text="Copy to Clipboard", 
                  command=copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        
        # Close button that will exit the application when clicked
        def close_and_exit():
            error_window.destroy()
            self.master.quit()
            self.master.destroy()
            
        ttk.Button(button_frame, text="Close", 
                  command=close_and_exit).pack(side=tk.LEFT, padx=5)
