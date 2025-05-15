# Importing useful libraries
import re
import matplotlib.pyplot as plt

# Defining input filenames
clip_file = "CLIP.out"
prompted_clip_file = "PROMPTED CLIP.out"

# Defining titles, x-axis labels, and inversion flags for each subplot
subplot_info = [
    ("a) Gaussian Noise", "Gaussian Noise Standard Deviation", False),
    ("b) Gaussian Blurring", "Gaussian Blur Level (number of iterative 3x3 mask applications)", False),
    ("c) Contrast Increase", "Contrast Factor", False),
    ("d) Contrast Decrease", "Contrast Factor", True),
    ("e) Brightness Increase", "Brightness Offset", False),
    ("f) Brightness Decrease", "Brightness Offset", True),
    ("g) Square Occlusion", "Occlusion Size (edge length in pixels)", False),
    ("h) Salt & Pepper Noise", "Salt & Pepper Noise Level", False),
    ("i) Imprecise prompt", "Heatmap Shift (pixels)", False)
]

def extract_perturbation_data(lines):
    pattern = re.compile(r"^\w\) ")
    results = {}
    current_name = None
    block = []

    for line in lines:
        if pattern.match(line):
            if current_name and block:
                results[current_name] = block
            current_name = line.strip().split("...")[0]
            block = []
        elif current_name:
            block.append(line.strip())
    if current_name and block:
        results[current_name] = block
    return results

def parse_table_safe(block):
    def safe_parse(line):
        return [float(x.strip()) for x in line.split("=")[1].split("|") if x.strip()]
    x_line = next((l for l in block if "=" in l and "score" not in l), "")
    y_line = next((l for l in block if "mean Dice score" in l), "")
    x_vals = safe_parse(x_line) if x_line else []
    y_vals = safe_parse(y_line) if y_line else []
    return x_vals, y_vals

# Loading data
with open(clip_file, "r") as f:
    clip_lines = f.readlines()
with open(prompted_clip_file, "r") as f:
    prompted_clip_lines = f.readlines()

clip_data = extract_perturbation_data(clip_lines)
prompted_clip_data = extract_perturbation_data(prompted_clip_lines)

# Setting up plot
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
colors = {"CLIP": "lightblue", "Prompted-CLIP": "darkblue"}

for i, (key, xlabel, invert) in enumerate(subplot_info):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    if key in clip_data:
        x, y = parse_table_safe(clip_data[key])
        if x and y:
            ax.plot(x, y, marker='o', label="CLIP", color=colors["CLIP"])
    if key in prompted_clip_data:
        x, y = parse_table_safe(prompted_clip_data[key])
        if x and y:
            ax.plot(x, y, marker='o', label="Prompted-enriched CLIP", color=colors["Prompted-CLIP"])

    ax.set_title(key)
    ax.set_xlabel(xlabel)
    if col == 0:
        ax.set_ylabel("Mean Dice Score")
    ax.grid()
    if invert:
        ax.invert_xaxis()

# Collecting and setting shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize=10, ncol=2)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("dice_robustness_comparison.pdf")
plt.close()

print("Updated plot saved to dice_robustness_comparison.pdf")