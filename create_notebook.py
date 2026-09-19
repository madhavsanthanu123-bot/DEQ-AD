import json
from pathlib import Path

def create_cell(cell_type, source):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source
        }
    elif cell_type == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        }

def main():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    cells = []
    
    # SETUP
    cells.append(create_cell("code", [
        "import sys\n",
        "import os\n",
        "from pathlib import Path\n",
        "# Ensure we are in the project root\n",
        "project_root = Path(r'D:\\DEQ-AD')\n",
        "if os.getcwd() != str(project_root):\n",
        "    os.chdir(project_root)\n",
        "sys.path.append(str(project_root))\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import roc_curve, auc, confusion_matrix\n",
        "import json\n",
        "plt.style.use('ggplot')\n"
    ]))
    
    # SECTION 1
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 1 — TITLE\n",
        "==================================================\n",
        "\n",
        "# DEQ-AD: Final Experimental Results\n",
        "\n",
        "**Deep Equilibrium Network for Alzheimer’s Disease Classification from Structural MRI**\n",
        "\n",
        "- **Dataset:** OASIS-1\n",
        "- **Subjects:** 235\n",
        "- **Primary task:** CDR=0 vs CDR>0\n",
        "- **Secondary task:** CDR severity prediction\n",
        "- **Architecture:** Lightweight 3D Encoder + Deep Equilibrium Layer\n",
        "- **Seeds:** 42, 123, 2026\n"
    ]))
    
    # SECTION 2
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 2 — RESEARCH QUESTION\n",
        "==================================================\n",
        "\n",
        "*\"Can an equilibrium-based neural representation learn clinically meaningful patterns from structural MRI for Alzheimer’s classification and cognitive-severity prediction?\"*\n",
        "\n",
        "MRI -> 3D encoder -> equilibrium representation -> classification/CDR heads\n"
    ]))
    
    # SECTION 3
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 3 — DATASET SUMMARY\n",
        "==================================================\n",
        "\n",
        "Total OASIS-1 labelled subjects: 235\n",
        "\n",
        "| Split | Count |\n",
        "|---|---|\n",
        "| Train | 164 |\n",
        "| Validation | 35 |\n",
        "| Test | 36 |\n",
        "\n",
        "| Class | Count |\n",
        "|---|---|\n",
        "| Control | 135 |\n",
        "| Impaired | 100 |\n",
        "\n",
        "| CDR | Count |\n",
        "|---|---|\n",
        "| 0.0 | 135 |\n",
        "| 0.5 | 70 |\n",
        "| 1.0 | 28 |\n",
        "| 2.0 | 2 |\n"
    ]))
    
    cells.append(create_cell("code", [
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n",
        "\n",
        "classes = ['Control', 'Impaired']\n",
        "counts = [135, 100]\n",
        "ax1.bar(classes, counts, color=['#4C72B0', '#C44E52'])\n",
        "ax1.set_title('Control vs Impaired')\n",
        "ax1.set_ylabel('Subjects')\n",
        "\n",
        "cdrs = ['0.0', '0.5', '1.0', '2.0']\n",
        "cdr_counts = [135, 70, 28, 2]\n",
        "ax2.bar(cdrs, cdr_counts, color=['#4C72B0', '#DD8452', '#C44E52', '#8172B3'])\n",
        "ax2.set_title('CDR Severity Distribution')\n",
        "ax2.set_ylabel('Subjects')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))
    
    # SECTION 4
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 4 — MODEL ARCHITECTURE\n",
        "==================================================\n"
    ]))
    
    cells.append(create_cell("code", [
        "fig, ax = plt.subplots(figsize=(8, 10))\n",
        "\n",
        "boxes = [\n",
        "    \"Input MRI\\n(1 × 96 × 112 × 96)\",\n",
        "    \"Conv3D 1→8\",\n",
        "    \"Conv3D 8→16\",\n",
        "    \"Conv3D 16→32\",\n",
        "    \"Adaptive Average Pooling\",\n",
        "    \"Linear 32→64\",\n",
        "    \"DEQ Layer\\nz* = β tanh(W_h z* + W_x x + b)\\nβ = 0.7\",\n",
        "    \"Equilibrium representation (64-D)\"\n",
        "]\n",
        "\n",
        "y_pos = np.linspace(0.9, 0.3, len(boxes))\n",
        "\n",
        "for i, (text, y) in enumerate(zip(boxes, y_pos)):\n",
        "    ax.text(0.5, y, text, ha='center', va='center', \n",
        "            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', alpha=0.9),\n",
        "            fontsize=11)\n",
        "    if i < len(boxes) - 1:\n",
        "        ax.annotate('', xy=(0.5, y_pos[i+1]+0.04), xytext=(0.5, y-0.04), \n",
        "                    arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6))\n",
        "\n",
        "# Split heads\n",
        "ax.text(0.3, 0.15, \"Classification Head\\n64 → 1\", ha='center', va='center',\n",
        "        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='black', alpha=0.9),\n",
        "        fontsize=11)\n",
        "ax.text(0.7, 0.15, \"CDR Head\\n64 → 4\", ha='center', va='center',\n",
        "        bbox=dict(boxstyle='round,pad=0.5', facecolor='salmon', edgecolor='black', alpha=0.9),\n",
        "        fontsize=11)\n",
        "\n",
        "ax.annotate('', xy=(0.3, 0.2), xytext=(0.5, 0.28),\n",
        "            arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6))\n",
        "ax.annotate('', xy=(0.7, 0.2), xytext=(0.5, 0.28),\n",
        "            arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6))\n",
        "\n",
        "ax.axis('off')\n",
        "ax.set_title(\"DEQ-AD Architecture Diagram\", fontsize=14, fontweight='bold')\n",
        "plt.show()\n"
    ]))
    
    # SECTION 5
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 5 — WHY DEQ?\n",
        "==================================================\n",
        "\n",
        "**Traditional network:**\n",
        "Layer 1 → Layer 2 → Layer 3 → ... → Layer N\n",
        "\n",
        "**DEQ:**\n",
        "Input → equilibrium transformation → stable representation\n",
        "\n",
        "$$ z^* = f_\\theta(z^*, x) $$\n",
        "\n",
        "Actual implementation equation:\n",
        "$$ z^* = \\beta \\tanh(W_h z^* + W_x x + b) $$\n",
        "\n",
        "The model does not explicitly unroll a fixed number of hidden layers. Instead, it solves for a stable equilibrium representation and trains it using implicit differentiation.\n"
    ]))
    
    # SECTION 6
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 6 — FINAL EXPERIMENTAL PROTOCOL\n",
        "==================================================\n",
        "\n",
        "| Parameter | Value |\n",
        "|---|---|\n",
        "| Beta | 0.7 |\n",
        "| Seeds | 42, 123, 2026 |\n",
        "| Batch size | 2 |\n",
        "| Optimizer | AdamW |\n",
        "| Learning rate | 1e-4 |\n",
        "| Weight decay | 1e-4 |\n",
        "| Epochs | 20 |\n",
        "| Early stopping | 5 epochs |\n",
        "| Checkpoint criterion | Validation ROC-AUC |\n",
        "| Threshold selection | Validation F1 |\n",
        "| Final threshold | 0.42 |\n",
        "\n",
        "**The test set was used only once for final evaluation and was not used for checkpoint selection or threshold tuning.**\n"
    ]))
    
    # SECTION 7
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 7 — MULTI-SEED RESULTS\n",
        "==================================================\n"
    ]))
    
    cells.append(create_cell("code", [
        "results_path = project_root / 'results' / 'final' / 'final_seed_comparison.csv'\n",
        "df = pd.read_csv(results_path)\n",
        "\n",
        "display_df = df[['seed', 'test_roc_auc', 'test_f1', 'test_accuracy', 'test_sensitivity', 'test_specificity']].copy()\n",
        "display_df.columns = ['Seed', 'Test AUC', 'F1', 'Accuracy', 'Sensitivity', 'Specificity']\n",
        "\n",
        "mean_row = display_df.mean()\n",
        "std_row = display_df.std()\n",
        "mean_str = [f\"{m:.4f} ± {s:.4f}\" if isinstance(m, float) else \"Mean ± SD\" for m, s in zip(mean_row, std_row)]\n",
        "display_df.loc['Mean ± SD'] = mean_str\n",
        "display(display_df)\n"
    ]))
    
    cells.append(create_cell("code", [
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n",
        "\n",
        "metrics = ['test_roc_auc', 'test_f1', 'test_accuracy', 'test_sensitivity']\n",
        "titles = ['ROC-AUC', 'F1 Score', 'Accuracy', 'Sensitivity']\n",
        "colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']\n",
        "\n",
        "for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):\n",
        "    axes[i].bar([str(s) for s in df['seed']], df[metric], color=color)\n",
        "    axes[i].set_title(f\"{title} per Seed\")\n",
        "    axes[i].set_ylim(0, 1.05)\n",
        "    axes[i].set_ylabel(title)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))
    
    # SECTION 8
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 8 — MAIN RESULT\n",
        "==================================================\n",
        "\n",
        "<div style=\"border: 2px solid #4C72B0; padding: 20px; text-align: center; background-color: #f8f9fa; border-radius: 10px;\">\n",
        "  <h2 style=\"color: #4C72B0; margin-top: 0;\">FINAL RESULT</h2>\n",
        "  <h1 style=\"font-size: 3em; margin: 10px 0;\">ROC-AUC</h1>\n",
        "  <h1 style=\"font-size: 4em; color: #C44E52; margin: 0;\">0.8825 ± 0.0084</h1>\n",
        "  <p style=\"font-size: 1.2em; font-style: italic;\">Across 3 independent random seeds</p>\n",
        "</div>\n",
        "\n",
        "**Additional Metrics:**\n",
        "- **F1:** 0.6854 ± 0.0857\n",
        "- **Accuracy:** 0.6296 ± 0.1891\n",
        "- **DEQ convergence:** 100%\n",
        "- **Average equilibrium iterations:** 10.35 ± 0.36\n"
    ]))
    
    # SECTION 9
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 9 — ROC CURVES\n",
        "==================================================\n"
    ]))
    
    cells.append(create_cell("code", [
        "plt.figure(figsize=(8, 6))\n",
        "\n",
        "seeds = [42, 123, 2026]\n",
        "colors = ['#4C72B0', '#55A868', '#C44E52']\n",
        "\n",
        "for seed, color in zip(seeds, colors):\n",
        "    pred_file = project_root / 'results' / 'final' / f'seed_{seed}' / 'test_predictions.csv'\n",
        "    if pred_file.exists():\n",
        "        preds = pd.read_csv(pred_file)\n",
        "        fpr, tpr, _ = roc_curve(preds['true_binary_label'], preds['binary_probability'])\n",
        "        auc_val = auc(fpr, tpr)\n",
        "        plt.plot(fpr, tpr, color=color, lw=2, label=f'Seed {seed} (AUC = {auc_val:.4f})')\n",
        "\n",
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Baseline')\n",
        "plt.xlim([0.0, 1.0])\n",
        "plt.ylim([0.0, 1.05])\n",
        "plt.xlabel('False Positive Rate')\n",
        "plt.ylabel('True Positive Rate')\n",
        "plt.title('Receiver Operating Characteristic (ROC)')\n",
        "plt.legend(loc=\"lower right\")\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.show()\n"
    ]))
    
    # SECTION 10
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 10 — CONFUSION MATRICES\n",
        "==================================================\n",
        "\n",
        "ROC-AUC evaluates ranking quality across thresholds, whereas accuracy/F1 depend on the selected classification threshold. This is why a seed can have a high ROC-AUC while showing poor specificity at the fixed threshold of 0.42.\n"
    ]))
    
    cells.append(create_cell("code", [
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
        "\n",
        "for i, seed in enumerate(seeds):\n",
        "    pred_file = project_root / 'results' / 'final' / f'seed_{seed}' / 'test_predictions.csv'\n",
        "    if pred_file.exists():\n",
        "        preds = pd.read_csv(pred_file)\n",
        "        cm = confusion_matrix(preds['true_binary_label'], preds['predicted_binary_label'])\n",
        "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], \n",
        "                    xticklabels=['Control', 'Impaired'], yticklabels=['Control', 'Impaired'])\n",
        "        axes[i].set_title(f'Seed {seed} Confusion Matrix')\n",
        "        axes[i].set_xlabel('Predicted')\n",
        "        axes[i].set_ylabel('True')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))
    
    # SECTION 11
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 11 — DEQ CONVERGENCE ANALYSIS\n",
        "==================================================\n",
        "\n",
        "- **Average iterations:** 10.35 ± 0.36\n",
        "- **Maximum:** 11\n",
        "- **Convergence:** 100%\n",
        "\n",
        "The equilibrium solver consistently reached the residual tolerance of 1e-5 within the maximum of 50 iterations.\n"
    ]))
    
    cells.append(create_cell("code", [
        "plt.figure(figsize=(6, 4))\n",
        "plt.bar([str(s) for s in df['seed']], df['avg_deq_iterations'], color='#DD8452')\n",
        "plt.title('Average DEQ Iterations per Seed')\n",
        "plt.ylabel('Iterations')\n",
        "plt.xlabel('Seed')\n",
        "plt.ylim(0, 15)\n",
        "for i, v in enumerate(df['avg_deq_iterations']):\n",
        "    plt.text(i, v + 0.2, f'{v:.2f}', ha='center')\n",
        "plt.show()\n"
    ]))
    
    # SECTION 12
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 12 — EQUILIBRIUM REPRESENTATION ANALYSIS\n",
        "==================================================\n",
        "\n",
        "This analysis examines whether equilibrium representations remain distinguishable across subjects.\n",
        "\n",
        "- **Seed 42:** Mean feature STD = 0.0162 | Avg pairwise distance = 0.166\n",
        "- **Seed 123:** Mean feature STD = 0.0049 | Avg pairwise distance = 0.052\n",
        "- **Seed 2026:** Mean feature STD = 0.0154 | Avg pairwise distance = 0.161\n",
        "\n",
        "Seed 123 produced a tighter representation and also showed lower specificity, which is an observed association, not necessarily proof of causation.\n"
    ]))
    
    cells.append(create_cell("code", [
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
        "\n",
        "ax1.bar([str(s) for s in df['seed']], df['feature_mean_std'], color='#8172B3')\n",
        "ax1.set_title('Mean Feature Standard Deviation')\n",
        "ax1.set_ylabel('STD')\n",
        "\n",
        "ax2.bar([str(s) for s in df['seed']], df['avg_pairwise_distance'], color='#937860')\n",
        "ax2.set_title('Average Pairwise Distance')\n",
        "ax2.set_ylabel('Distance')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))
    
    # SECTION 13
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 13 — CDR SEVERITY RESULTS\n",
        "==================================================\n",
        "\n",
        "CDR Confusion Matrix (identical across all seeds):\n",
        "```python\n",
        "[[21, 0, 0, 0],\n",
        " [10, 0, 0, 0],\n",
        " [ 5, 0, 0, 0],\n",
        " [ 0, 0, 0, 0]]\n",
        "```\n",
        "\n",
        "**True Distribution:** CDR 0.0 (21), CDR 0.5 (10), CDR 1.0 (5), CDR 2.0 (0)\n",
        "**Predicted Distribution:** CDR 0.0 (36), CDR 0.5 (0), CDR 1.0 (0), CDR 2.0 (0)\n",
        "\n",
        "**CDR Macro-F1 = 0.2456**\n",
        "\n",
        "The secondary CDR severity head did not successfully distinguish the four severity classes and predicted CDR=0 for all test subjects. \n",
        "\n",
        "**Possible Data Limitations:**\n",
        "- Only 235 labelled subjects total.\n",
        "- CDR classes are highly imbalanced.\n",
        "- Only 28 subjects have CDR=1.\n",
        "- Only 2 subjects have CDR=2 in the full labelled cohort.\n",
        "- Test set contains no CDR=2 subjects.\n"
    ]))
    
    # SECTION 14
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 14 — LIMITATIONS\n",
        "==================================================\n",
        "\n",
        "1. Small labelled OASIS-1 cohort.\n",
        "2. Single-dataset evaluation.\n",
        "3. No external validation.\n",
        "4. CDR severity prediction remains weak.\n",
        "5. Accuracy/F1 vary considerably with threshold and random seed.\n",
        "6. DEQ is not inherently proven superior to conventional CNNs by this experiment alone.\n",
        "7. Benchmark comparisons across papers are not directly equivalent because datasets, splits and evaluation protocols differ.\n"
    ]))
    
    # SECTION 15
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 15 — SELECTED BENCHMARK CONTEXT\n",
        "==================================================\n",
        "\n",
        "Our result should be interpreted as a comparison with selected OASIS-1 studies, not as a claim of state-of-the-art performance.\n",
        "\n",
        "**Our DEQ-AD:**\n",
        "ROC-AUC = 0.8825 ± 0.0084\n",
        "\n",
        "**Selected reported OASIS-1 examples:**\n",
        "3D DenseNet121 ≈ 0.8698 AUC\n",
        "Other OASIS studies may report ≈0.90 depending on architecture, split and evaluation protocol.\n",
        "\n",
        "These values are not directly apples-to-apples because the experimental protocols differ.\n"
    ]))
    
    # SECTION 16
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 16 — FINAL TAKEAWAY\n",
        "==================================================\n",
        "\n",
        "### KEY FINDINGS\n",
        "\n",
        "✓ DEQ successfully processed 3D structural MRI.\n",
        "✓ Equilibrium solver converged in all final runs.\n",
        "✓ Mean test ROC-AUC = 0.8825 ± 0.0084.\n",
        "✓ Results were reproduced across three independent seeds.\n",
        "✓ β=0.7 was the strongest setting among the tested β values.\n",
        "✓ Representation analysis provided insight into model behavior.\n",
        "⚠ Fine-grained CDR severity prediction remains a limitation.\n",
        "\n",
        "**The main contribution is not a claim of state-of-the-art accuracy, but an equilibrium-based framework with systematic analysis of stability, representation behavior, and reproducibility for structural MRI Alzheimer’s classification.**\n"
    ]))
    
    # SECTION 17
    cells.append(create_cell("markdown", [
        "==================================================\n",
        "## SECTION 17 — VIVA / CONFERENCE QUESTIONS\n",
        "==================================================\n",
        "\n",
        "### Questions the Panel May Ask\n",
        "\n",
        "**1. Why did you choose DEQ?**\n",
        "To evaluate if infinite-depth weight-tied representations could stabilize robust features from high-dimensional 3D MRI without extreme memory costs.\n",
        "\n",
        "**2. What exactly is the equilibrium state?**\n",
        "It is the stable feature vector $z^*$ where passing it through the network block again produces the exact same vector $z^*$.\n",
        "\n",
        "**3. Why beta = 0.7?**\n",
        "Ablation studies across 0.5, 0.7, and 0.9 showed 0.7 provided the best balance of feature variance and ranking performance without compromising convergence.\n",
        "\n",
        "**4. Why not ResNet?**\n",
        "A 3D ResNet for 96x112x96 volumes quickly exceeds the 4GB VRAM constraint of the development environment. DEQ trades compute time for O(1) memory.\n",
        "\n",
        "**5. How does DEQ differ from a normal deep network?**\n",
        "It doesn't have N distinct layers. It uses one layer repeatedly via a root-finding solver until the output stops changing.\n",
        "\n",
        "**6. What does implicit differentiation mean?**\n",
        "Instead of storing the computational graph of all 50 solver steps (which causes Out-Of-Memory errors), we compute gradients analytically directly at the equilibrium point using the Implicit Function Theorem.\n",
        "\n",
        "**7. How did you prevent data leakage?**\n",
        "Strict subject-level splits mapping directly to disk, validation-only model selection, and zero test-set access for thresholding or PCA fitting.\n",
        "\n",
        "**8. Why use ROC-AUC as the primary metric?**\n",
        "It evaluates the model's fundamental ranking capability across all possible thresholds, which is crucial given the class imbalance.\n",
        "\n",
        "**9. Why is accuracy lower than ROC-AUC might suggest?**\n",
        "Because ROC-AUC is threshold-independent, but accuracy is bound to our fixed 0.42 threshold. Seed variability caused calibration shifts, penalizing fixed-threshold accuracy.\n",
        "\n",
        "**10. Why does the CDR head fail?**\n",
        "Severe class imbalance (0.0=135, 2.0=2) and small dataset size meant the 0.5 loss weight couldn't overcome the base-rate prior. The network just predicted the majority class.\n",
        "\n",
        "**11. Why are three seeds necessary?**\n",
        "To mathematically prove the result is a property of the architecture/data interaction, not just a lucky lottery ticket initialization.\n",
        "\n",
        "**12. Is this state-of-the-art?**\n",
        "No. It is a competitive baseline (~0.88 AUC) serving as a proof-of-concept for DEQs on 3D MRI, rather than an attempt to beat all OASIS-1 leaderboards.\n",
        "\n",
        "**13. What is actually novel in your work?**\n",
        "Applying Deep Equilibrium networks natively to 3D volumetric MRI for Alzheimer's, coupled with rigorous representation collapse diagnostics.\n",
        "\n",
        "**14. What would you do next?**\n",
        "Pretrain the 3D encoder (e.g. MedicalNet), implement focal loss for the CDR head, and augment the training data to mitigate representation collapse.\n",
        "\n",
        "**15. Can this be clinically deployed?**\n",
        "No. The dataset is too small, uncalibrated (high false positives in seed 123), and lacks external cross-site validation required for medical devices.\n"
    ]))
    
    cells.append(create_cell("code", [
        "print(\"DEQ-AD Final Results Notebook completed successfully.\")"
    ]))
    
    notebook["cells"] = cells
    
    out_dir = Path(r"D:\DEQ-AD\notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "DEQ_AD_Final_Results.ipynb"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Created notebook at {out_file}")

if __name__ == "__main__":
    main()
