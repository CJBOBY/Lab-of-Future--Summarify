import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import docx2txt
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import torch

# Ensure 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- File Reader ---
def read_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.docx':
            return docx2txt.process(filepath)
        elif ext == '.pdf':
            doc = fitz.open(filepath)
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
            return text
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        messagebox.showerror("File Read Error", f"Failed to read file:\n{e}")
        return ""

# --- Simple App ---
class SimpleSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Summarify - AI Document Summarizer")
        self.root.geometry("900x700")
        
        self.summarizer = None
        self.models_loaded = False
        
        self.LENGTH_CONFIG = {
            "Short": 0.15,
            "Medium": 0.30,
            "Long": 0.60
        }
        
        self.build_ui()
        
        # Load model in background
        threading.Thread(target=self.load_model, daemon=True).start()
    
    def load_model(self):
        """Load the BART summarization model"""
        self.update_status("Loading AI model...")
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
            self.models_loaded = True
            self.root.after(0, lambda: self.update_status("Model loaded! Ready to summarize."))
        except Exception as e:
            self.root.after(0, lambda: self.update_status("Model loading failed."))
            messagebox.showerror("Model Load Error", f"Failed to load model: {str(e)}")
    
    def build_ui(self):
        """Build the user interface"""
        # Title
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=10)
        
        title = tk.Label(title_frame, text="📄 Summarify", font=("Arial", 20, "bold"))
        title.pack()
        
        subtitle = tk.Label(title_frame, text="AI-Powered Document Summarization", 
                          font=("Arial", 12), fg="gray")
        subtitle.pack()
        
        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="📁 Upload File", 
                  command=self.load_file).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="🧹 Clear Input", 
                  command=self.clear_input).grid(row=0, column=1, padx=5)
        self.summarize_btn = ttk.Button(control_frame, text="🧠 Summarize", 
                                       command=self.start_summarization)
        self.summarize_btn.grid(row=0, column=2, padx=5)
        
        # Status
        self.status_label = tk.Label(self.root, text="Status: Initializing...", fg="blue")
        self.status_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self.root, length=400, mode="indeterminate")
        self.progress_bar.pack(pady=5)
        
        # Main content frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Input section
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        input_header = tk.Frame(input_frame)
        input_header.pack(fill=tk.X)
        
        tk.Label(input_header, text="Input Text:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.word_count_label = tk.Label(input_header, text="Words: 0")
        self.word_count_label.pack(side=tk.RIGHT)
        
        self.text_area = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=12)
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        self.text_area.bind('<KeyRelease>', self.update_word_count)
        
        # Summary length selection
        length_frame = tk.Frame(main_frame)
        length_frame.pack(pady=5)
        
        tk.Label(length_frame, text="Summary Length:").pack(side=tk.LEFT)
        self.length_var = tk.StringVar(value="Medium")
        self.length_combo = ttk.Combobox(length_frame, textvariable=self.length_var,
                                        values=list(self.LENGTH_CONFIG.keys()), 
                                        state="readonly", width=10)
        self.length_combo.pack(side=tk.LEFT, padx=10)
        
        # Output section
        output_frame = tk.Frame(main_frame)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(output_frame, text="Summary:", font=("Arial", 12, "bold")).pack(anchor='w')
        
        self.summary_area = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
        self.summary_area.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Output controls
        output_controls = tk.Frame(main_frame)
        output_controls.pack(pady=5)
        
        ttk.Button(output_controls, text="📋 Copy", command=self.copy_summary).grid(row=0, column=0, padx=5)
        ttk.Button(output_controls, text="💾 Save", command=self.save_summary).grid(row=0, column=1, padx=5)
        ttk.Button(output_controls, text="🧹 Clear", command=self.clear_summary).grid(row=0, column=2, padx=5)
    
    def load_file(self):
        """Load a file into the text area"""
        filepath = filedialog.askopenfilename(
            title="Select a document to summarize",
            filetypes=[
                ("Text Files", "*.txt"),
                ("Word Documents", "*.docx"),
                ("PDF Files", "*.pdf"),
                ("All Files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        self.update_status("Loading file...")
        text = read_file(filepath)
        
        if text:
            self.text_area.delete('1.0', tk.END)
            self.text_area.insert('1.0', text)
            self.update_word_count()
            self.update_status(f"Loaded: {os.path.basename(filepath)}")
        else:
            self.update_status("Failed to load file")
    
    def clear_input(self):
        """Clear the input text area"""
        self.text_area.delete('1.0', tk.END)
        self.update_word_count()
        self.update_status("Input cleared")
    
    def clear_summary(self):
        """Clear the summary area"""
        self.summary_area.delete('1.0', tk.END)
    
    def update_word_count(self, event=None):
        """Update word count display"""
        text = self.text_area.get('1.0', tk.END).strip()
        word_count = len(text.split()) if text else 0
        self.word_count_label.config(text=f"Words: {word_count}")
    
    def start_summarization(self):
        """Start the summarization process"""
        if not self.models_loaded:
            messagebox.showwarning("Model Not Ready", "Please wait for the AI model to load.")
            return
        
        text = self.text_area.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter or upload text to summarize.")
            return
        
        if len(text.split()) < 50:
            messagebox.showwarning("Text Too Short", 
                                 "Please provide at least 50 words for meaningful summarization.")
            return
        
        # Disable button and start progress
        self.summarize_btn.config(state='disabled')
        self.progress_bar.start()
        
        # Run in background thread
        threading.Thread(target=self.perform_summarization, args=(text,), daemon=True).start()
    
    def perform_summarization(self, text):
        """Perform the actual summarization"""
        try:
            self.root.after(0, lambda: self.update_status("Generating summary..."))
            
            # Calculate summary length
            ratio = self.LENGTH_CONFIG[self.length_var.get()]
            word_count = len(text.split())
            target_length = max(30, min(int(word_count * ratio), 500))
            min_length = max(10, int(target_length * 0.3))
            
            # Handle long texts by chunking
            if len(text) > 1000:
                # Split into chunks
                chunks = [text[i:i+900] for i in range(0, len(text), 900)]
                summaries = []
                
                for i, chunk in enumerate(chunks):
                    self.root.after(0, lambda i=i: self.update_status(f"Processing part {i+1}/{len(chunks)}..."))
                    chunk_summary = self.summarizer(
                        chunk,
                        max_length=min(150, target_length//len(chunks) + 30),
                        min_length=min(20, min_length//len(chunks) + 5),
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(chunk_summary)
                
                # Combine summaries
                combined = " ".join(summaries)
                if len(combined.split()) > target_length:
                    final_summary = self.summarizer(
                        combined,
                        max_length=target_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                else:
                    final_summary = combined
            else:
                # Direct summarization for shorter texts
                final_summary = self.summarizer(
                    text,
                    max_length=target_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
            
            # Update UI in main thread
            self.root.after(0, lambda: self.display_summary(final_summary))
            
        except Exception as e:
            self.root.after(0, lambda: self.handle_error(str(e)))
    
    def display_summary(self, summary):
        """Display the generated summary"""
        self.summary_area.delete('1.0', tk.END)
        self.summary_area.insert('1.0', summary)
        self.summarize_btn.config(state='normal')
        self.progress_bar.stop()
        
        word_count = len(summary.split())
        self.update_status(f"Summary complete! ({word_count} words)")
    
    def handle_error(self, error_msg):
        """Handle summarization errors"""
        self.summarize_btn.config(state='normal')
        self.progress_bar.stop()
        self.update_status("Summarization failed")
        messagebox.showerror("Error", f"Failed to generate summary:\n{error_msg}")
    
    def copy_summary(self):
        """Copy summary to clipboard"""
        summary = self.summary_area.get('1.0', tk.END).strip()
        if summary:
            self.root.clipboard_clear()
            self.root.clipboard_append(summary)
            self.update_status("Summary copied to clipboard")
        else:
            messagebox.showinfo("No Summary", "No summary to copy")
    
    def save_summary(self):
        """Save summary to file"""
        summary = self.summary_area.get('1.0', tk.END).strip()
        if not summary:
            messagebox.showinfo("No Summary", "No summary to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Summary",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(summary)
                self.update_status(f"Summary saved: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file:\n{e}")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=f"Status: {message}")

# --- Main ---
if __name__ == "__main__":
    # Check dependencies
    try:
        import transformers
        import torch
    except ImportError:
        messagebox.showerror("Missing Dependencies", 
                           "Please install required packages:\n\npip install transformers torch")
        exit(1)
    
    try:
        import fitz
    except ImportError:
        messagebox.showerror("Missing Dependencies", 
                           "Please install PyMuPDF:\n\npip install PyMuPDF")
        exit(1)
    
    root = tk.Tk()
    app = SimpleSummarizerApp(root)
    root.mainloop()