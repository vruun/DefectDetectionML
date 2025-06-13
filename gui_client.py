import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import grpc
import sys
import os
import threading
from io import BytesIO

# Add the ml_grpc_project/server directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.join(current_dir, "ml_grpc_project", "server")
sys.path.append(server_dir)

try:
    from ml_grpc_project.server import app_pb2
    from ml_grpc_project.server import app_pb2_grpc
except ImportError as e:
    print(f"Error importing proto files: {e}")
    sys.exit(1)

# =====================================================
# CUSTOMIZATION SECTION - EDIT THESE VALUES TO CHANGE APPEARANCE
# =====================================================

# WINDOW SETTINGS
WINDOW_TITLE = "Industrial Defect Detection System"  # Title bar text
WINDOW_SIZE = "900x800"  # "widthxheight" - overall window size
WINDOW_BG_COLOR = "#161127"  # Background color around the main content

# COLORS
COLORS = {
    'primary': "#8e57b3",        # "Test Component" button background, status text color
    'success': "#3fde81",        # "Result: OK" text color when test passes
    'danger': "#ea4057",         # "Train Model" button background, "Result: DEFECT" text color
    'warning': "#8e57b3",        # "Browse" button background
    'bg_main': "#1F1835",        # Main content area background (white sections)
    'bg_secondary': '#1F1835',   # Section backgrounds (Input, Status, Result, Image sections)
    'text_primary': "#ffffff",   # Main text color (section titles, labels like "Component Name:")
    'text_secondary': "#ffffff", # Hint text color (like "(e.g., nuts, bolts, discs)")
    'border': "#4526ce",         # Border color around entry boxes and frames
    'entry_bg': "#38136b",       # Background color inside text entry boxes
    'button_hover': "#6b4de4",   # Button color when you hover over them
}

# FONTS
FONTS = {
    'title': ("Arial", 18, "bold"),     # Main window title at the top
    'heading': ("Arial", 12, "bold"),   # Section titles ("Input Configuration", "Status", etc.)
    'normal': ("Arial", 10),            # Labels like "Component Name:", "Test Image Path:"
    'entry': ("Arial", 11),             # Text you type inside the entry boxes
    'button': ("Arial", 10, "bold"),    # Text inside all buttons ("Test Component", "Browse", etc.)
    'result': ("Arial", 14, "bold"),    # "Result: OK" or "Result: DEFECT" text
    'status': ("Arial", 9),             # Status messages and hint text
}

# SIZING
SIZES = {
    'entry_width': 50,           # Width of text entry boxes (higher = wider boxes)
    'entry_height': 35,          # Height of text entry boxes
    'button_padx': 20,           # Button width padding (makes buttons wider)
    'button_pady': 8,            # Button height padding (makes buttons taller)
    'frame_padding': 15,         # Space inside each section frame
    'border_width': 2,           # Thickness of borders around frames and entry boxes
    'corner_radius': 10,         # How rounded the entry box corners are (higher = more round)
}

# SPACING
SPACING = {
    'section_gap': 20,           # Space between main sections (Input, Status, Result, etc.)
    'element_gap': 10,           # Space between elements within a section
    'small_gap': 5,              # Small spaces between labels and entry boxes
    'frame_padding': 15,         # Padding inside frames
}

# =====================================================
# END CUSTOMIZATION SECTION
# =====================================================

class DefectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=WINDOW_BG_COLOR)
        
        # Variables
        self.component_var = tk.StringVar()
        self.image_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.result_var = tk.StringVar()
        
        # gRPC connection
        self.channel = None
        self.stub = None
        
        self.setup_ui()
        self.connect_to_server()
        
        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            if hasattr(self, 'main_canvas'):
                self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
    
    def create_rounded_entry(self, parent, textvariable, width=None, height=None):
        """Create a rounded entry widget using Canvas"""
        if width is None:
            width = SIZES['entry_width'] * 8
        if height is None:
            height = SIZES['entry_height']
            
        canvas = tk.Canvas(parent, width=width, height=height, 
                          highlightthickness=0, bg=COLORS['bg_main'])
        
        # Draw rounded rectangle background
        def draw_rounded_rect(x1, y1, x2, y2, radius=SIZES['corner_radius']):
            points = []
            for x, y in [(x1, y1 + radius), (x1, y1), (x1 + radius, y1),
                         (x2 - radius, y1), (x2, y1), (x2, y1 + radius),
                         (x2, y2 - radius), (x2, y2), (x2 - radius, y2),
                         (x1 + radius, y2), (x1, y2), (x1, y2 - radius)]:
                points.extend([x, y])
            return canvas.create_polygon(points, smooth=True, 
                                       fill=COLORS['entry_bg'], 
                                       outline=COLORS['border'], 
                                       width=SIZES['border_width'])
        
        # Create rounded background
        draw_rounded_rect(2, 2, width-2, height-2)
        
        # Create entry widget on top
        entry = tk.Entry(canvas, textvariable=textvariable, 
                        font=FONTS['entry'], border=0, highlightthickness=0,
                        bg=COLORS['entry_bg'], fg=COLORS['text_primary'],
                        insertbackground=COLORS['text_primary'])
        canvas.create_window(width//2, height//2, window=entry, width=width-20)
        
        return canvas, entry
    
    def create_styled_button(self, parent, text, command, bg_color=None, state="normal"):
        """Create a styled button with custom colors"""
        if bg_color is None:
            bg_color = COLORS['primary']
        
        button = tk.Button(parent, text=text, command=command,
                          font=FONTS['button'],
                          bg=bg_color,
                          fg='white',
                          relief='flat',
                          bd=0,
                          padx=SIZES['button_padx'], 
                          pady=SIZES['button_pady'],
                          cursor='hand2',
                          activebackground=COLORS['button_hover'],
                          activeforeground='white',
                          state=state)
        return button
    
    def create_styled_frame(self, parent, title=None, bg_color=None):
        """Create a styled frame with optional title"""
        if bg_color is None:
            bg_color = COLORS['bg_secondary']
        
        frame = tk.Frame(parent, bg=bg_color, relief='ridge', 
                        bd=SIZES['border_width'])
        
        if title:
            title_label = tk.Label(frame, text=title, 
                                  font=FONTS['heading'], 
                                  bg=bg_color, fg=COLORS['text_primary'])
            title_label.pack(pady=(SPACING['element_gap'], SPACING['small_gap']))
        
        return frame
        
    def setup_ui(self):
        # Create main canvas and scrollbar for entire page
        self.main_canvas = tk.Canvas(self.root, bg=COLORS['bg_main'])
        main_scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas, bg=COLORS['bg_main'])
        
        # Pack scrollbar first, then canvas fills remaining space
        main_scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        # Make scrollable frame fill the canvas width
        def configure_scroll_region(event):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
            # Make the scrollable frame fill the canvas width
            canvas_width = event.width
            self.main_canvas.itemconfig(self.main_canvas.find_all()[0], width=canvas_width)
        
        self.main_canvas.bind('<Configure>', configure_scroll_region)
        
        # Main container
        main_container = tk.Frame(self.scrollable_frame, bg=COLORS['bg_main'])
        main_container.pack(fill='both', expand=True, padx=SPACING['section_gap'], 
                        pady=SPACING['section_gap'])
        
        # Title
        title_label = tk.Label(main_container, text=WINDOW_TITLE, 
                            font=FONTS['title'], 
                            bg=COLORS['bg_main'], fg=COLORS['text_primary'])
        title_label.pack(pady=(0, SPACING['section_gap']))
        
        # Input Section Frame
        input_frame = self.create_styled_frame(main_container, "Input Configuration")
        input_frame.pack(fill='x', pady=(0, SPACING['section_gap']))
        
        # Component input
        comp_container = tk.Frame(input_frame, bg=COLORS['bg_secondary'])
        comp_container.pack(fill='x', padx=SPACING['frame_padding'], 
                        pady=SPACING['element_gap'])
        
        tk.Label(comp_container, text="Component Name:", 
                font=FONTS['normal'], bg=COLORS['bg_secondary'], 
                fg=COLORS['text_primary']).pack(anchor='w')
        
        self.component_canvas, self.component_entry = self.create_rounded_entry(
            comp_container, self.component_var)
        self.component_canvas.pack(fill='x', pady=(SPACING['small_gap'], 0))
        
        tk.Label(comp_container, text="(e.g., nuts, bolts, discs)", 
                font=FONTS['status'], bg=COLORS['bg_secondary'], 
                fg=COLORS['text_secondary']).pack(anchor='w')
        
        # Image path input
        img_container = tk.Frame(input_frame, bg=COLORS['bg_secondary'])
        img_container.pack(fill='x', padx=SPACING['frame_padding'], 
                        pady=SPACING['element_gap'])
        
        img_label_frame = tk.Frame(img_container, bg=COLORS['bg_secondary'])
        img_label_frame.pack(fill='x')
        
        tk.Label(img_label_frame, text="Test Image Path:", 
                font=FONTS['normal'], bg=COLORS['bg_secondary'], 
                fg=COLORS['text_primary']).pack(side='left')
        
        self.browse_button = self.create_styled_button(img_label_frame, "Browse", 
                                                    self.browse_image, COLORS['warning'])
        self.browse_button.pack(side='right')
        
        self.image_canvas, self.image_entry = self.create_rounded_entry(
            img_container, self.image_path_var)
        self.image_canvas.pack(fill='x', pady=(SPACING['small_gap'], 0))
        
        # Control Buttons Frame
        button_frame = tk.Frame(main_container, bg=COLORS['bg_main'])
        button_frame.pack(pady=SPACING['section_gap'])
        
        self.test_button = self.create_styled_button(button_frame, "Test Component", 
                                                    self.test_component, COLORS['primary'])
        self.test_button.pack(side='left', padx=(0, SPACING['element_gap']))
        
        self.train_button = self.create_styled_button(button_frame, "Train Model", 
                                                    self.train_model, COLORS['danger'], 
                                                    state="disabled")
        self.train_button.pack(side='left')
        
        # Status Section
        status_frame = self.create_styled_frame(main_container, "Status")
        status_frame.pack(fill='x', pady=(0, SPACING['section_gap']))
        
        status_container = tk.Frame(status_frame, bg=COLORS['bg_secondary'])
        status_container.pack(fill='x', padx=SPACING['frame_padding'], 
                            pady=SPACING['element_gap'])
        
        tk.Label(status_container, text="Current Status:", 
                font=FONTS['normal'], bg=COLORS['bg_secondary'], 
                fg=COLORS['text_primary']).pack(side='left')
        
        self.status_label = tk.Label(status_container, textvariable=self.status_var, 
                                    font=FONTS['status'], bg=COLORS['bg_secondary'], 
                                    fg=COLORS['primary'])
        self.status_label.pack(side='left', padx=(SPACING['element_gap'], 0))
        
        # Result Section
        result_frame = self.create_styled_frame(main_container, "Detection Result")
        result_frame.pack(fill='x', pady=(0, SPACING['section_gap']))
        
        result_container = tk.Frame(result_frame, bg=COLORS['bg_secondary'])
        result_container.pack(fill='x', padx=SPACING['frame_padding'], 
                            pady=SPACING['element_gap'])
        
        self.result_label = tk.Label(result_container, textvariable=self.result_var, 
                                    font=FONTS['result'], bg=COLORS['bg_secondary'], 
                                    fg=COLORS['success'])
        self.result_label.pack()
        
        # Image Display Section
        image_display_frame = self.create_styled_frame(main_container, "Test Image Preview")
        image_display_frame.pack(fill='x', pady=(0, SPACING['section_gap']))
        
        image_container = tk.Frame(image_display_frame, bg=COLORS['bg_secondary'])
        image_container.pack(fill='x', padx=SPACING['frame_padding'], 
                        pady=SPACING['element_gap'])
        
        self.image_label = tk.Label(image_container, text="No image loaded",
                                bg=COLORS['entry_bg'], fg=COLORS['text_secondary'],
                                font=FONTS['normal'], relief='sunken', 
                                bd=SIZES['border_width'])
        self.image_label.pack(expand=True, fill='both', padx=SPACING['small_gap'], 
                            pady=SPACING['small_gap'])
    def connect_to_server(self):
        try:
            self.channel = grpc.insecure_channel('192.168.1.97:50051')
            self.stub = app_pb2_grpc.AppServiceStub(self.channel)
            self.status_var.set("Connected to server")
        except Exception as e:
            self.status_var.set(f"Connection failed: {e}")
            messagebox.showerror("Connection Error", f"Failed to connect to server: {e}")
    
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if filename:
            self.image_path_var.set(filename)
    
    def test_component(self):
        component = self.component_var.get().strip()
        image_path = self.image_path_var.get().strip()
        
        if not component:
            messagebox.showerror("Error", "Please enter a component name")
            return
        
        if not image_path:
            messagebox.showerror("Error", "Please select an image path")
            return
        
        if not os.path.exists(image_path):
            messagebox.showerror("Error", "Image file does not exist")
            return
        
        # Disable button during processing
        self.test_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        self.status_var.set("Testing in progress...")
        self.result_var.set("")
        
        # Run test in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self._run_test, args=(component, image_path))
        thread.daemon = True
        thread.start()
    
    def _run_test(self, component, image_path):
        try:
            requests = [
                app_pb2.ClientRequest(text_input=app_pb2.TextInput(text=component)),
                app_pb2.ClientRequest(image_path=app_pb2.ImagePath(path=image_path))
            ]
            
            received_image = False
            needs_training = False
            result = None
            
            for response in self.stub.Communicate(iter(requests)):
                if response.HasField('status'):
                    result = response.status.status
                elif response.HasField('message'):
                    message = response.message.message
                    self.root.after(0, lambda: self.status_var.set(message))
                    if "not found" in message.lower():
                        needs_training = True
                elif response.HasField('image_data'):
                    # Display received image
                    self.root.after(0, lambda: self._display_image(response.image_data))
                    received_image = True
            
            # Update UI in main thread
            if result:
                if result == "OK":
                    self.root.after(0, lambda: self._show_result(result, COLORS['success']))
                else:
                    self.root.after(0, lambda: self._show_result(result, COLORS['danger']))
            
            if needs_training:
                self.root.after(0, lambda: self.train_button.configure(state="normal"))
                self.root.after(0, lambda: self.status_var.set("Model not found - Training required"))
            else:
                self.root.after(0, lambda: self.status_var.set("Test completed"))
                
        except grpc.RpcError as e:
            self.root.after(0, lambda: messagebox.showerror("gRPC Error", f"Communication error: {e}"))
            self.root.after(0, lambda: self.status_var.set("Test failed"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Test failed: {e}"))
            self.root.after(0, lambda: self.status_var.set("Test failed"))
        finally:
            self.root.after(0, lambda: self.test_button.configure(state="normal"))
    
    def _display_image(self, image_data):
        try:
            # Convert bytes to PIL Image
            image_bytes = BytesIO(image_data.image_bytes)
            pil_image = Image.open(image_bytes)
            
            # Resize image to fit in display area (max 400x300)
            pil_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for Tkinter
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update image label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.status_var.set(f"Failed to display image: {e}")
    
    def _show_result(self, result, color):
        self.result_var.set(f"Result: {result}")
        self.result_label.configure(foreground=color)
    
    def train_model(self):
        component = self.component_var.get().strip()
        
        if not component:
            messagebox.showerror("Error", "Please enter a component name")
            return
        
        # Confirm training
        if not messagebox.askyesno("Confirm Training", 
                                   f"Train model for component '{component}'?\nThis may take several minutes."):
            return
        
        self.test_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        self.status_var.set("Training in progress...")
        
        # Run training in separate thread
        thread = threading.Thread(target=self._run_training, args=(component,))
        thread.daemon = True
        thread.start()
    
    def _run_training(self, component):
        try:
            requests = [
                app_pb2.ClientRequest(text_input=app_pb2.TextInput(text=component)),
                app_pb2.ClientRequest(train_command=app_pb2.TrainCommand(start=True))
            ]
            
            for response in self.stub.Communicate(iter(requests)):
                if response.HasField('message'):
                    message = response.message.message
                    self.root.after(0, lambda m=message: self.status_var.set(f"Training: {m}"))
                    
                    if "successfully" in message.lower():
                        self.root.after(0, lambda: messagebox.showinfo("Training Complete", 
                                                                       "Model trained successfully!"))
                        self.root.after(0, lambda: self.status_var.set("Training completed successfully"))
                        self.root.after(0, lambda: self.train_button.configure(state="disabled"))
                        break
                    elif "failed" in message.lower():
                        self.root.after(0, lambda: messagebox.showerror("Training Failed", 
                                                                        f"Training failed: {message}"))
                        break
                        
        except grpc.RpcError as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Training error: {e}"))
            self.root.after(0, lambda: self.status_var.set("Training failed"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training error: {e}"))
            self.root.after(0, lambda: self.status_var.set("Training failed"))
        finally:
            self.root.after(0, lambda: self.test_button.configure(state="normal"))
    
    def __del__(self):
        if self.channel:
            self.channel.close()

def main():
    root = tk.Tk()
    app = DefectDetectionGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")
    finally:
        if app.channel:
            app.channel.close()

if __name__ == "__main__":
    main()