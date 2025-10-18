import matplotlib.pyplot as plt
import os

def show_case_study(folder="outputs/shap", n_cases=3):
    files = [f for f in os.listdir(folder) if f.startswith("local_case_") and f.endswith(".png")]
    files = sorted(files)[:n_cases]
    fig, axs = plt.subplots(1, len(files), figsize=(5 * len(files), 4))
    if len(files) == 1:
        axs = [axs]
    for ax, f in zip(axs, files):
        img = plt.imread(os.path.join(folder, f))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f)
    plt.tight_layout()
    plt.show()
