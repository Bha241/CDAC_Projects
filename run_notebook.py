import os
import json
import traceback
import sys

# Set working directory to the project directory
project_dir = r"D:\CDAC PGCP AI\Project Idea\Pose Suggestor\Code2"
os.chdir(project_dir)
sys.path.append(project_dir)

print(f"Working directory set to: {os.getcwd()}")

# Load the notebook
notebook_path = "AestheticScorer_Training.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Global dictionary to maintain state across executed cells
globals_dict = {
    "__name__": "__main__",
    "__file__": os.path.abspath(notebook_path)
}
locals_dict = globals_dict

# Prevent matplotlib from blocking execution by mocking plt.show()
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

print("Starting notebook cell execution...")

cell_idx = 0
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell_idx += 1
        source_code = "".join(cell["source"])
        
        # Clean magic commands (like !nvidia-smi or !pip)
        clean_lines = []
        for line in source_code.split("\n"):
            if line.strip().startswith("!"):
                print(f"Skipping shell magic command: {line}")
                continue
            clean_lines.append(line)
        
        clean_code = "\n".join(clean_lines)
        if not clean_code.strip():
            continue
            
        print(f"\n--- Executing Cell {cell_idx} ---")
        try:
            exec(clean_code, globals_dict, locals_dict)
            print(f"Cell {cell_idx} executed successfully.")
        except Exception as e:
            print(f"Error executing Cell {cell_idx}:")
            traceback.print_exc()
            sys.exit(1)

print("\nAll notebook cells executed successfully!")
