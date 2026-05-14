import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt


def compute_auc_safe(true_labels, probs, num_classes=None):
    unique = np.unique(true_labels)
    if len(unique) < 2:
        return None
    if num_classes is None:
        num_classes = probs.shape[1]
    if not np.all(np.isfinite(probs)):
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        probs = probs / row_sums
    try:
        if num_classes == 2:
            return float(roc_auc_score(true_labels, probs[:, 1]))
        else:
            return float(roc_auc_score(
                true_labels, probs, multi_class='ovr', average='weighted',
                labels=list(range(num_classes))
            ))
    except ValueError:
        return None


class Evaluator:
    
    CLASS_NAMES = [
        "Adipose",
        "Background", 
        "Debris",
        "Lymphocytes",
        "Mucus",
        "Smooth Muscle",
        "Normal Colon Mucosa",
        "Cancer-associated Stroma",
        "Colorectal Adenocarcinoma"
    ]
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        results_dir: str = "results",
        log_dir: str = "logs"
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(device)
        self.device = device
        self.results_dir = Path(results_dir)
        self.log_dir = Path(log_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions = None
        self.true_labels = None
        self.probabilities = None
        self.sample_indices = None
        
    @torch.no_grad()
    def run_inference(
        self,
        dataloader: torch.utils.data.DataLoader,
        desc: str = "Evaluating"
    ) -> Dict[str, np.ndarray]:
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(self.device)
            
            outputs = self.model(images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        self.predictions = np.array(all_preds)
        self.true_labels = np.array(all_labels)
        self.probabilities = np.array(all_probs)
        self.sample_indices = np.arange(len(self.predictions))
        
        return {
            'predictions': self.predictions,
            'true_labels': self.true_labels,
            'probabilities': self.probabilities
        }
    
    def compute_metrics(self) -> Dict[str, Any]:
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        accuracy = accuracy_score(self.true_labels, self.predictions)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average=None,
            zero_division=0
        )
        
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average='macro',
            zero_division=0
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average='weighted',
            zero_division=0
        )
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        errors = self.predictions != self.true_labels
        error_rate = errors.mean()
        error_indices = np.where(errors)[0]

        auc = compute_auc_safe(self.true_labels, self.probabilities)

        return {
            'accuracy': accuracy,
            'auc': auc,
            'error_rate': error_rate,
            'num_errors': len(error_indices),
            'total_samples': len(self.predictions),
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            },
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'weighted': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            },
            'confusion_matrix': cm,
            'error_indices': error_indices
        }
    
    def compute_edit_success_rate(
        self,
        edit_indices: List[int],
        original_predictions: np.ndarray = None
    ) -> Dict[str, Any]:
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        edit_indices = np.array(edit_indices)
        
        edit_preds = self.predictions[edit_indices]
        edit_labels = self.true_labels[edit_indices]
        
        successful = edit_preds == edit_labels
        success_rate = successful.mean()
        
        result = {
            'total_edits': len(edit_indices),
            'successful_edits': successful.sum(),
            'failed_edits': (~successful).sum(),
            'success_rate': success_rate,
            'successful_indices': edit_indices[successful].tolist(),
            'failed_indices': edit_indices[~successful].tolist()
        }
        
        if original_predictions is not None:
            original_preds = original_predictions[edit_indices]
            
            fixed = (original_preds != edit_labels) & (edit_preds == edit_labels)
            
            broken = (original_preds == edit_labels) & (edit_preds != edit_labels)
            
            changed_still_wrong = (original_preds != edit_preds) & (edit_preds != edit_labels)
            
            result['flip_analysis'] = {
                'fixed': fixed.sum(),
                'broken': broken.sum(),
                'changed_still_wrong': changed_still_wrong.sum(),
                'unchanged': (original_preds == edit_preds).sum()
            }
        
        return result
    
    def export_confusion_matrix(
        self,
        filename: str = "confusion_matrix.csv",
        normalize: bool = False
    ) -> str:
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        
        df = pd.DataFrame(
            cm,
            index=[f"True_{i}_{name}" for i, name in enumerate(self.CLASS_NAMES)],
            columns=[f"Pred_{i}_{name}" for i, name in enumerate(self.CLASS_NAMES)]
        )
        
        csv_path = self.results_dir / filename
        df.to_csv(csv_path)
        
        print(f"Confusion matrix exported to: {csv_path}")
        return str(csv_path)
    
    def export_evaluation_report(
        self,
        filename: str = "evaluation_report.csv",
        additional_info: Dict[str, Any] = None
    ) -> str:
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        metrics = self.compute_metrics()
        
        rows = []
        
        rows.append({
            'category': 'overall',
            'metric': 'accuracy',
            'value': metrics['accuracy'],
            'notes': f"{metrics['accuracy']*100:.2f}%"
        })
        rows.append({
            'category': 'overall',
            'metric': 'error_rate',
            'value': metrics['error_rate'],
            'notes': f"{metrics['num_errors']}/{metrics['total_samples']}"
        })
        
        for metric_name in ['precision', 'recall', 'f1']:
            rows.append({
                'category': 'macro_avg',
                'metric': metric_name,
                'value': metrics['macro'][metric_name],
                'notes': ''
            })
        
        for metric_name in ['precision', 'recall', 'f1']:
            rows.append({
                'category': 'weighted_avg',
                'metric': metric_name,
                'value': metrics['weighted'][metric_name],
                'notes': ''
            })
        
        for class_id in range(len(self.CLASS_NAMES)):
            class_name = self.CLASS_NAMES[class_id]
            for metric_name in ['precision', 'recall', 'f1', 'support']:
                rows.append({
                    'category': f'class_{class_id}',
                    'metric': metric_name,
                    'value': metrics['per_class'][metric_name][class_id],
                    'notes': class_name
                })
        
        if additional_info:
            for key, value in additional_info.items():
                rows.append({
                    'category': 'additional',
                    'metric': key,
                    'value': value if isinstance(value, (int, float)) else str(value),
                    'notes': ''
                })
        
        df = pd.DataFrame(rows)
        csv_path = self.results_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"Evaluation report exported to: {csv_path}")
        return str(csv_path)
    
    def export_predictions(
        self,
        filename: str = "predictions.csv"
    ) -> str:
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        rows = []
        for i in range(len(self.predictions)):
            row = {
                'sample_idx': i,
                'true_label': self.true_labels[i],
                'true_class': self.CLASS_NAMES[self.true_labels[i]],
                'predicted_label': self.predictions[i],
                'predicted_class': self.CLASS_NAMES[self.predictions[i]],
                'correct': self.predictions[i] == self.true_labels[i],
                'confidence': self.probabilities[i].max()
            }
            
            for j in range(len(self.CLASS_NAMES)):
                row[f'prob_class_{j}'] = self.probabilities[i, j]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.results_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"Predictions exported to: {csv_path}")
        return str(csv_path)
    
    def print_summary(self):
        if self.predictions is None:
            print("No predictions available. Run inference first.")
            return
        
        metrics = self.compute_metrics()
        
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Error Rate: {metrics['error_rate']*100:.2f}%")
        print(f"  Total Samples: {metrics['total_samples']}")
        print(f"  Errors: {metrics['num_errors']}")
        
        print(f"\nMacro Averages:")
        print(f"  Precision: {metrics['macro']['precision']:.4f}")
        print(f"  Recall: {metrics['macro']['recall']:.4f}")
        print(f"  F1-Score: {metrics['macro']['f1']:.4f}")
        
        print(f"\nPer-Class Performance:")
        print(f"{'Class':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 75)
        
        for i, class_name in enumerate(self.CLASS_NAMES):
            print(f"{i}: {class_name:<32} "
                  f"{metrics['per_class']['precision'][i]:>10.4f} "
                  f"{metrics['per_class']['recall'][i]:>10.4f} "
                  f"{metrics['per_class']['f1'][i]:>10.4f} "
                  f"{metrics['per_class']['support'][i]:>10.0f}")
        
        print("\nConfusion Matrix (top-5 confusions):")
        cm = metrics['confusion_matrix']
        
        confusions = []
        for i in range(len(self.CLASS_NAMES)):
            for j in range(len(self.CLASS_NAMES)):
                if i != j and cm[i, j] > 0:
                    confusions.append((i, j, cm[i, j]))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        for true_idx, pred_idx, count in confusions[:5]:
            print(f"  {self.CLASS_NAMES[true_idx]} → {self.CLASS_NAMES[pred_idx]}: {count}")
        
        print("=" * 70)
    
    def plot_confusion_matrix(
        self,
        filename: str = "confusion_matrix.png",
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ) -> str:
        if self.predictions is None:
            raise ValueError("No predictions available. Run inference first.")
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        
        short_names = [name[:15] + "..." if len(name) > 15 else name 
                       for name in self.CLASS_NAMES]
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=short_names,
            yticklabels=short_names
        )
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        png_path = self.results_dir / filename
        plt.savefig(png_path, dpi=150)
        plt.close()
        
        print(f"Confusion matrix plot saved to: {png_path}")
        return str(png_path)


def evaluate_before_after(
    model_before: nn.Module,
    model_after: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = None,
    results_dir: str = "results"
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Evaluating BEFORE editing ===")
    eval_before = Evaluator(model_before, device, results_dir / "before")
    eval_before.run_inference(dataloader, desc="Before")
    metrics_before = eval_before.compute_metrics()
    eval_before.export_confusion_matrix("confusion_matrix_before.csv")
    eval_before.export_evaluation_report("evaluation_before.csv")
    eval_before.print_summary()

    print("\n=== Evaluating AFTER editing ===")
    eval_after = Evaluator(model_after, device, results_dir / "after")
    eval_after.run_inference(dataloader, desc="After")
    metrics_after = eval_after.compute_metrics()
    eval_after.export_confusion_matrix("confusion_matrix_after.csv")
    eval_after.export_evaluation_report("evaluation_after.csv")
    eval_after.print_summary()

    auc_before = metrics_before.get('auc')
    auc_after = metrics_after.get('auc')
    auc_change = (auc_after - auc_before) if (auc_before is not None and auc_after is not None) else None

    comparison = {
        'accuracy_before': metrics_before['accuracy'],
        'accuracy_after': metrics_after['accuracy'],
        'accuracy_change': metrics_after['accuracy'] - metrics_before['accuracy'],
        'auc_before': auc_before,
        'auc_after': auc_after,
        'auc_change': auc_change,
        'macro_f1_before': metrics_before['macro']['f1'],
        'macro_f1_after': metrics_after['macro']['f1'],
        'macro_f1_change': metrics_after['macro']['f1'] - metrics_before['macro']['f1'],
        'errors_before': metrics_before['num_errors'],
        'errors_after': metrics_after['num_errors'],
        'errors_fixed': metrics_before['num_errors'] - metrics_after['num_errors']
    }

    df = pd.DataFrame([comparison])
    df.to_csv(results_dir / "comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("EDIT SUCCESS CHECKLIST")
    print("=" * 70)
    print(f"Accuracy: {comparison['accuracy_before']*100:.2f}% → {comparison['accuracy_after']*100:.2f}% "
          f"({comparison['accuracy_change']*100:+.2f}%)")
    print(f"Macro-F1: {comparison['macro_f1_before']:.4f} → {comparison['macro_f1_after']:.4f} "
          f"({comparison['macro_f1_change']:+.4f})")
    print(f"Errors: {comparison['errors_before']} → {comparison['errors_after']} "
          f"({comparison['errors_fixed']:+d} fixed)")

    if auc_before is not None and auc_after is not None:
        print(f"AUC:      {auc_before:.4f} → {auc_after:.4f} ({auc_change:+.4f})")

    print("\nJudgment Indicators:")
    print("  • Accuracy improvement    : " + ("PASS" if comparison['accuracy_change'] > 0 else "CHECK"))
    print("  • Macro-F1 improvement    : " + ("PASS" if comparison['macro_f1_change'] > 0 else "CHECK"))
    print("  • Errors reduced          : " + ("PASS" if comparison['errors_fixed'] > 0 else "CHECK"))
    print("  • No regression (errors)  : " + ("PASS" if comparison['errors_after'] <= comparison['errors_before'] else "CHECK"))
    print("=" * 70)

    return comparison


def evaluate_comparative(
    model_orig: nn.Module,
    model_edit: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = None,
    results_dir: str = "results",
    set_name: str = "Test Set"
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    set_name_safe = set_name.lower().replace(" ", "_").replace("-", "_")

    print("\n" + "=" * 70)
    print(f"COMPARATIVE EVALUATION ON {set_name.upper()}")
    print("(4-Set Protocol: Pre-Edit vs Post-Edit Comparison)")
    print("=" * 70)

    model_orig.eval()
    model_edit.eval()

    all_labels = []
    preds_orig = []
    preds_edit = []
    probs_orig = []
    probs_edit = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating on {set_name}"):
            images = images.to(device)

            out_orig = model_orig(images)
            logits_orig = out_orig.logits
            prob_orig = torch.softmax(logits_orig, dim=-1)

            out_edit = model_edit(images)
            logits_edit = out_edit.logits
            prob_edit = torch.softmax(logits_edit, dim=-1)

            all_labels.extend(labels.numpy())
            preds_orig.extend(logits_orig.argmax(dim=1).cpu().numpy())
            preds_edit.extend(logits_edit.argmax(dim=1).cpu().numpy())
            probs_orig.extend(prob_orig.cpu().numpy())
            probs_edit.extend(prob_edit.cpu().numpy())

    all_labels = np.array(all_labels)
    preds_orig = np.array(preds_orig)
    preds_edit = np.array(preds_edit)
    probs_orig = np.array(probs_orig)
    probs_edit = np.array(probs_edit)

    acc_orig = accuracy_score(all_labels, preds_orig)
    acc_edit = accuracy_score(all_labels, preds_edit)
    accuracy_delta = acc_edit - acc_orig

    auc_orig = compute_auc_safe(all_labels, probs_orig)
    auc_edit = compute_auc_safe(all_labels, probs_edit)
    auc_delta = (auc_edit - auc_orig) if (auc_orig is not None and auc_edit is not None) else None

    correct_orig = (preds_orig == all_labels)
    correct_edit = (preds_edit == all_labels)

    stable_mask = correct_orig & correct_edit
    stability = stable_mask.sum() / correct_orig.sum() if correct_orig.sum() > 0 else 0.0

    error_orig = ~correct_orig
    fixed_mask = error_orig & correct_edit
    fix_rate = fixed_mask.sum() / error_orig.sum() if error_orig.sum() > 0 else 0.0

    regression_mask = correct_orig & ~correct_edit
    regression_rate = regression_mask.sum() / correct_orig.sum() if correct_orig.sum() > 0 else 0.0

    cm_orig = confusion_matrix(all_labels, preds_orig)
    cm_edit = confusion_matrix(all_labels, preds_edit)

    n_total = len(all_labels)
    n_correct_orig = correct_orig.sum()
    n_error_orig = error_orig.sum()
    n_correct_edit = correct_edit.sum()
    n_fixed = fixed_mask.sum()
    n_regressed = regression_mask.sum()
    n_stable = stable_mask.sum()

    result = {
        'accuracy_orig': acc_orig,
        'accuracy_edit': acc_edit,
        'accuracy_delta': accuracy_delta,
        'auc_orig': auc_orig,
        'auc_edit': auc_edit,
        'auc_delta': auc_delta,
        'stability': stability,
        'fix_rate': fix_rate,
        'regression_rate': regression_rate,
        'n_total': n_total,
        'n_correct_orig': int(n_correct_orig),
        'n_error_orig': int(n_error_orig),
        'n_correct_edit': int(n_correct_edit),
        'n_fixed': int(n_fixed),
        'n_regressed': int(n_regressed),
        'n_stable': int(n_stable),
        'confusion_matrix_orig': cm_orig,
        'confusion_matrix_edit': cm_edit
    }

    print(f"\n=== Comparative Evaluation Results ({set_name}) ===")
    print(f"{set_name} Size: {n_total} samples")
    print(f"\nAccuracy:")
    print(f"  Pre-Edit:  {acc_orig*100:.2f}% ({n_correct_orig}/{n_total})")
    print(f"  Post-Edit: {acc_edit*100:.2f}% ({n_correct_edit}/{n_total})")
    print(f"  Delta:     {accuracy_delta*100:+.2f}%")

    if auc_orig is not None and auc_edit is not None:
        print(f"\nAUC:")
        print(f"  Pre-Edit:  {auc_orig:.4f}")
        print(f"  Post-Edit: {auc_edit:.4f}")
        print(f"  Delta:     {auc_delta:+.4f}")

    print(f"\nTransition Analysis:")
    print(f"  Stability (correct->correct): {stability*100:.1f}% ({n_stable}/{n_correct_orig})")
    print(f"  Fix Rate (error->correct):    {fix_rate*100:.1f}% ({n_fixed}/{n_error_orig})")
    print(f"  Regression (correct->error):  {regression_rate*100:.1f}% ({n_regressed}/{n_correct_orig})")

    print(f"\nJudgment Indicators:")
    print(f"  [{'PASS' if accuracy_delta > 0 else 'FAIL'}] Accuracy improved")
    print(f"  [{'PASS' if stability > 0.95 else 'WARN' if stability > 0.90 else 'FAIL'}] Stability > 95%")
    print(f"  [{'PASS' if fix_rate > 0 else 'INFO'}] Some errors fixed")
    print(f"  [{'PASS' if regression_rate < 0.05 else 'WARN' if regression_rate < 0.10 else 'FAIL'}] Regression < 5%")

    summary_rows = [
        {'metric': 'accuracy_orig', 'value': acc_orig, 'notes': f'{acc_orig*100:.2f}%'},
        {'metric': 'accuracy_edit', 'value': acc_edit, 'notes': f'{acc_edit*100:.2f}%'},
        {'metric': 'accuracy_delta', 'value': accuracy_delta, 'notes': f'{accuracy_delta*100:+.2f}%'},
        {'metric': 'auc_orig', 'value': auc_orig if auc_orig is not None else '', 'notes': f'{auc_orig:.4f}' if auc_orig is not None else 'N/A'},
        {'metric': 'auc_edit', 'value': auc_edit if auc_edit is not None else '', 'notes': f'{auc_edit:.4f}' if auc_edit is not None else 'N/A'},
        {'metric': 'auc_delta', 'value': auc_delta if auc_delta is not None else '', 'notes': f'{auc_delta:+.4f}' if auc_delta is not None else 'N/A'},
        {'metric': 'stability', 'value': stability, 'notes': f'{n_stable}/{n_correct_orig}'},
        {'metric': 'fix_rate', 'value': fix_rate, 'notes': f'{n_fixed}/{n_error_orig}'},
        {'metric': 'regression_rate', 'value': regression_rate, 'notes': f'{n_regressed}/{n_correct_orig}'},
        {'metric': 'n_total', 'value': n_total, 'notes': f'{set_name} size'},
        {'metric': 'n_fixed', 'value': n_fixed, 'notes': 'errors corrected'},
        {'metric': 'n_regressed', 'value': n_regressed, 'notes': 'new errors introduced'},
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / f'comparative_evaluation_{set_name_safe}.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\nResults exported to: {summary_path}")

    n_classes = cm_orig.shape[0]
    df_cm_orig = pd.DataFrame(
        cm_orig,
        index=[f"True_{i}" for i in range(n_classes)],
        columns=[f"Pred_{i}" for i in range(n_classes)]
    )
    df_cm_edit = pd.DataFrame(
        cm_edit,
        index=[f"True_{i}" for i in range(n_classes)],
        columns=[f"Pred_{i}" for i in range(n_classes)]
    )

    df_cm_orig.to_csv(results_dir / f'confusion_matrix_orig_{set_name_safe}.csv')
    df_cm_edit.to_csv(results_dir / f'confusion_matrix_edit_{set_name_safe}.csv')

    print("=" * 70)

    return result


def evaluate_edit_samples(
    model: nn.Module,
    images: torch.Tensor,
    true_labels: torch.Tensor,
    sample_indices: List[int],
    device: torch.device = None,
    desc: str = "Edit Samples",
    batch_size: int = 128
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    if batch_size is None or batch_size <= 0:
        batch_size = len(images)

    predictions_list = []
    probs_list = []

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = start + batch_size
            batch_images = images[start:end].to(device)

            outputs = model(batch_images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=1)

            predictions_list.append(predictions.cpu().numpy())
            probs_list.append(probs.cpu().numpy())

    predictions = np.concatenate(predictions_list, axis=0)
    probs_np = np.concatenate(probs_list, axis=0)
    true_labels_np = true_labels.cpu().numpy()

    correct = predictions == true_labels_np
    accuracy = correct.mean()
    auc = compute_auc_safe(true_labels_np, probs_np)

    per_sample_info = []
    for i in range(len(predictions)):
        per_sample_info.append({
            'dataset_idx': sample_indices[i],
            'true_label': int(true_labels_np[i]),
            'predicted_label': int(predictions[i]),
            'correct': bool(correct[i]),
            'confidence': float(probs_np[i].max()),
            'true_class_prob': float(probs_np[i, true_labels_np[i]])
        })

    return {
        'predictions': predictions,
        'true_labels': true_labels_np,
        'probabilities': probs_np,
        'correct': correct,
        'accuracy': accuracy,
        'auc': auc,
        'num_correct': int(correct.sum()),
        'num_total': len(predictions),
        'per_sample_info': per_sample_info
    }


def compare_edit_samples_before_after(
    results_before: Dict[str, Any],
    results_after: Dict[str, Any],
    sample_indices: List[int]
) -> Dict[str, Any]:
    preds_before = results_before['predictions']
    preds_after = results_after['predictions']
    true_labels = results_before['true_labels']

    correct_before = results_before['correct']
    correct_after = results_after['correct']

    fixed = (~correct_before) & correct_after
    broken = correct_before & (~correct_after)
    stayed_correct = correct_before & correct_after
    stayed_wrong = (~correct_before) & (~correct_after)

    per_sample_transitions = []
    for i in range(len(preds_before)):
        if fixed[i]:
            status = "FIXED"
        elif broken[i]:
            status = "BROKEN"
        elif stayed_correct[i]:
            status = "STAYED_CORRECT"
        else:
            status = "STAYED_WRONG"

        per_sample_transitions.append({
            'dataset_idx': sample_indices[i],
            'true_label': int(true_labels[i]),
            'pred_before': int(preds_before[i]),
            'pred_after': int(preds_after[i]),
            'correct_before': bool(correct_before[i]),
            'correct_after': bool(correct_after[i]),
            'status': status
        })

    auc_before = results_before.get('auc')
    auc_after = results_after.get('auc')
    auc_delta = (auc_after - auc_before) if (auc_before is not None and auc_after is not None) else None

    comparison = {
        'accuracy_before': results_before['accuracy'],
        'accuracy_after': results_after['accuracy'],
        'accuracy_delta': results_after['accuracy'] - results_before['accuracy'],
        'auc_before': auc_before,
        'auc_after': auc_after,
        'auc_delta': auc_delta,
        'num_fixed': int(fixed.sum()),
        'num_broken': int(broken.sum()),
        'num_stayed_correct': int(stayed_correct.sum()),
        'num_stayed_wrong': int(stayed_wrong.sum()),
        'num_total': len(preds_before),
        'fix_rate': float(fixed.sum()) / max(1, (~correct_before).sum()),
        'break_rate': float(broken.sum()) / max(1, correct_before.sum()),
        'per_sample_transitions': per_sample_transitions
    }

    return comparison


def print_edit_samples_comparison(
    comparison: Dict[str, Any],
    results_before: Dict[str, Any],
    results_after: Dict[str, Any]
):
    print("\n" + "=" * 70)
    print("EDIT SAMPLES PERFORMANCE COMPARISON")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Total Edit Samples: {comparison['num_total']}")
    print(f"  Accuracy Before: {comparison['accuracy_before']*100:.1f}% "
          f"({results_before['num_correct']}/{results_before['num_total']})")
    print(f"  Accuracy After:  {comparison['accuracy_after']*100:.1f}% "
          f"({results_after['num_correct']}/{results_after['num_total']})")
    print(f"  Accuracy Change: {comparison['accuracy_delta']*100:+.1f}%")

    auc_before = comparison.get('auc_before')
    auc_after = comparison.get('auc_after')
    auc_delta = comparison.get('auc_delta')
    if auc_before is not None and auc_after is not None:
        print(f"  AUC Before:      {auc_before:.4f}")
        print(f"  AUC After:       {auc_after:.4f}")
        print(f"  AUC Change:      {auc_delta:+.4f}")

    print(f"\nTransition Analysis:")
    print(f"  FIXED (wrong->correct):    {comparison['num_fixed']} samples")
    print(f"  BROKEN (correct->wrong):   {comparison['num_broken']} samples")
    print(f"  STAYED_CORRECT:            {comparison['num_stayed_correct']} samples")
    print(f"  STAYED_WRONG:              {comparison['num_stayed_wrong']} samples")

    print(f"\nPer-Sample Details (showing up to 20):")
    print(f"{'Idx':>6} {'True':>6} {'Before':>8} {'After':>8} {'Status':>15}")
    print("-" * 50)

    for info in comparison['per_sample_transitions'][:20]:
        status_symbol = {
            'FIXED': '+',
            'BROKEN': 'X',
            'STAYED_CORRECT': '=',
            'STAYED_WRONG': '-'
        }.get(info['status'], '?')

        print(f"{info['dataset_idx']:>6} {info['true_label']:>6} "
              f"{info['pred_before']:>8} {info['pred_after']:>8} "
              f"{status_symbol} {info['status']:>13}")

    if len(comparison['per_sample_transitions']) > 20:
        print(f"  ... and {len(comparison['per_sample_transitions']) - 20} more samples")

    print(f"\nEdit Success Judgment:")
    if comparison['num_fixed'] > 0 and comparison['num_broken'] == 0:
        print(f"  [EXCELLENT] Fixed {comparison['num_fixed']} errors with no regressions!")
    elif comparison['num_fixed'] > comparison['num_broken']:
        print(f"  [GOOD] Net improvement: +{comparison['num_fixed'] - comparison['num_broken']} correct predictions")
    elif comparison['num_fixed'] == comparison['num_broken']:
        print(f"  [NEUTRAL] Equal fixes and regressions")
    else:
        print(f"  [POOR] More regressions than fixes: {comparison['num_broken']} broken vs {comparison['num_fixed']} fixed")

    print("=" * 70)


def evaluate_projection_samples(
    model: nn.Module,
    projection_samples: Dict[str, Any],
    device: torch.device = None,
    desc: str = "Projection Samples"
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if projection_samples is None:
        print(f"Warning: No projection samples available")
        return None

    model.eval()
    model.to(device)

    images = projection_samples['images'].to(device)
    labels = projection_samples['labels']
    if isinstance(labels, torch.Tensor):
        labels = labels.to(device)
    else:
        labels = torch.tensor(labels).to(device)

    num_samples = projection_samples['num_samples']

    batch_size = 32
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i+batch_size]
            outputs = model(batch_images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=1)

            all_predictions.append(predictions.cpu())
            all_probs.append(probs.cpu())

    predictions = torch.cat(all_predictions, dim=0).numpy()
    probs_np = torch.cat(all_probs, dim=0).numpy()
    true_labels_np = labels.cpu().numpy()

    correct = predictions == true_labels_np
    accuracy = correct.mean()
    auc = compute_auc_safe(true_labels_np, probs_np)

    unique_classes, class_counts = np.unique(true_labels_np, return_counts=True)
    class_distribution = dict(zip(unique_classes.tolist(), class_counts.tolist()))

    per_class_accuracy = {}
    for cls in unique_classes:
        mask = true_labels_np == cls
        if mask.sum() > 0:
            per_class_accuracy[int(cls)] = float(correct[mask].mean())

    per_sample_info = []
    for i in range(min(100, num_samples)):
        per_sample_info.append({
            'sample_idx': i,
            'true_label': int(true_labels_np[i]),
            'predicted_label': int(predictions[i]),
            'correct': bool(correct[i]),
            'confidence': float(probs_np[i].max()),
            'true_class_prob': float(probs_np[i, true_labels_np[i]])
        })

    return {
        'predictions': predictions,
        'true_labels': true_labels_np,
        'probabilities': probs_np,
        'correct': correct,
        'accuracy': accuracy,
        'auc': auc,
        'num_correct': int(correct.sum()),
        'num_total': num_samples,
        'class_distribution': class_distribution,
        'per_class_accuracy': per_class_accuracy,
        'per_sample_info': per_sample_info
    }


def compare_projection_samples_before_after(
    results_before: Dict[str, Any],
    results_after: Dict[str, Any]
) -> Dict[str, Any]:
    if results_before is None or results_after is None:
        return None

    preds_before = results_before['predictions']
    preds_after = results_after['predictions']
    true_labels = results_before['true_labels']

    correct_before = results_before['correct']
    correct_after = results_after['correct']

    fixed = (~correct_before) & correct_after
    broken = correct_before & (~correct_after)
    stayed_correct = correct_before & correct_after
    stayed_wrong = (~correct_before) & (~correct_after)

    per_class_stability = {}
    for cls in np.unique(true_labels):
        mask = true_labels == cls
        cls_correct_before = correct_before[mask]
        cls_correct_after = correct_after[mask]
        cls_stayed_correct = (cls_correct_before & cls_correct_after).sum()
        cls_total_correct_before = cls_correct_before.sum()

        per_class_stability[int(cls)] = {
            'stability': float(cls_stayed_correct / max(1, cls_total_correct_before)),
            'correct_before': int(cls_total_correct_before),
            'correct_after': int(cls_correct_after.sum()),
            'broken': int((cls_correct_before & ~cls_correct_after).sum())
        }

    auc_before = results_before.get('auc')
    auc_after = results_after.get('auc')
    auc_delta = (auc_after - auc_before) if (auc_before is not None and auc_after is not None) else None

    comparison = {
        'accuracy_before': results_before['accuracy'],
        'accuracy_after': results_after['accuracy'],
        'accuracy_delta': results_after['accuracy'] - results_before['accuracy'],
        'auc_before': auc_before,
        'auc_after': auc_after,
        'auc_delta': auc_delta,
        'num_fixed': int(fixed.sum()),
        'num_broken': int(broken.sum()),
        'num_stayed_correct': int(stayed_correct.sum()),
        'num_stayed_wrong': int(stayed_wrong.sum()),
        'num_total': len(preds_before),
        'stability': float(stayed_correct.sum()) / max(1, correct_before.sum()),
        'regression_rate': float(broken.sum()) / max(1, correct_before.sum()),
        'per_class_stability': per_class_stability
    }

    return comparison


def print_projection_samples_comparison(
    comparison: Dict[str, Any],
    results_before: Dict[str, Any],
    results_after: Dict[str, Any]
):
    if comparison is None:
        print("\n[WARNING] No projection samples comparison available")
        return

    print("\n" + "=" * 70)
    print("PROJECTION SAMPLES (FT-Train) PERFORMANCE COMPARISON")
    print("These samples were used to construct X-Edit's projection matrix P")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Total Projection Samples: {comparison['num_total']}")
    print(f"  Accuracy Before: {comparison['accuracy_before']*100:.2f}% "
          f"({results_before['num_correct']}/{results_before['num_total']})")
    print(f"  Accuracy After:  {comparison['accuracy_after']*100:.2f}% "
          f"({results_after['num_correct']}/{results_after['num_total']})")
    print(f"  Accuracy Change: {comparison['accuracy_delta']*100:+.2f}%")

    auc_before = comparison.get('auc_before')
    auc_after = comparison.get('auc_after')
    auc_delta = comparison.get('auc_delta')
    if auc_before is not None and auc_after is not None:
        print(f"  AUC Before:      {auc_before:.4f}")
        print(f"  AUC After:       {auc_after:.4f}")
        print(f"  AUC Change:      {auc_delta:+.4f}")

    print(f"\nKnowledge Preservation Analysis:")
    print(f"  Stability (correct->correct): {comparison['stability']*100:.1f}% "
          f"({comparison['num_stayed_correct']}/{results_before['num_correct']})")
    print(f"  Regression (correct->wrong):  {comparison['regression_rate']*100:.1f}% "
          f"({comparison['num_broken']}/{results_before['num_correct']})")
    print(f"  Fixed (wrong->correct):       {comparison['num_fixed']} samples")
    print(f"  Stayed Wrong:                 {comparison['num_stayed_wrong']} samples")

    print(f"\nPer-Class Stability (sorted by regression count):")
    print(f"{'Class':>6} {'Before':>8} {'After':>8} {'Broken':>8} {'Stability':>10}")
    print("-" * 45)

    sorted_classes = sorted(
        comparison['per_class_stability'].items(),
        key=lambda x: x[1]['broken'],
        reverse=True
    )

    for cls, stats in sorted_classes[:10]:
        print(f"{cls:>6} {stats['correct_before']:>8} {stats['correct_after']:>8} "
              f"{stats['broken']:>8} {stats['stability']*100:>9.1f}%")

    print(f"\nKnowledge Preservation Judgment:")
    if comparison['stability'] >= 0.99 and comparison['num_broken'] == 0:
        print(f"  [EXCELLENT] Perfect preservation - no regressions on projection samples!")
    elif comparison['stability'] >= 0.95:
        print(f"  [GOOD] High stability (>95%) - null-space projection working well")
    elif comparison['stability'] >= 0.90:
        print(f"  [WARNING] Moderate stability (90-95%) - some knowledge loss")
    else:
        print(f"  [POOR] Low stability (<90%) - significant knowledge loss, check projection matrix")

    if comparison['num_broken'] > 0:
        print(f"  [INFO] {comparison['num_broken']} previously correct samples now wrong (regression)")

    if comparison['accuracy_delta'] < -0.01:
        print(f"  [WARNING] Accuracy dropped by {abs(comparison['accuracy_delta'])*100:.2f}%")
    elif comparison['accuracy_delta'] > 0.01:
        print(f"  [INFO] Accuracy improved by {comparison['accuracy_delta']*100:.2f}% (unexpected but good)")

    print("=" * 70)


def export_edit_samples_comparison(
    comparison: Dict[str, Any],
    results_before: Dict[str, Any],
    results_after: Dict[str, Any],
    results_dir: str = "results"
) -> str:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {'metric': 'accuracy_before', 'value': comparison['accuracy_before'],
         'notes': f"{comparison['accuracy_before']*100:.2f}%"},
        {'metric': 'accuracy_after', 'value': comparison['accuracy_after'],
         'notes': f"{comparison['accuracy_after']*100:.2f}%"},
        {'metric': 'accuracy_delta', 'value': comparison['accuracy_delta'],
         'notes': f"{comparison['accuracy_delta']*100:+.2f}%"},
        {'metric': 'num_total', 'value': comparison['num_total'],
         'notes': 'total edit samples'},
        {'metric': 'num_fixed', 'value': comparison['num_fixed'],
         'notes': 'wrong->correct'},
        {'metric': 'num_broken', 'value': comparison['num_broken'],
         'notes': 'correct->wrong (regression)'},
        {'metric': 'num_stayed_correct', 'value': comparison['num_stayed_correct'],
         'notes': 'correct->correct'},
        {'metric': 'num_stayed_wrong', 'value': comparison['num_stayed_wrong'],
         'notes': 'wrong->wrong'},
        {'metric': 'fix_rate', 'value': comparison['fix_rate'],
         'notes': f"{comparison['fix_rate']*100:.1f}%"},
        {'metric': 'break_rate', 'value': comparison['break_rate'],
         'notes': f"{comparison['break_rate']*100:.1f}%"},
    ]

    auc_before = comparison.get('auc_before')
    auc_after = comparison.get('auc_after')
    auc_delta = comparison.get('auc_delta')
    if auc_before is not None:
        summary_rows.append({'metric': 'auc_before', 'value': auc_before, 'notes': f'{auc_before:.4f}'})
    if auc_after is not None:
        summary_rows.append({'metric': 'auc_after', 'value': auc_after, 'notes': f'{auc_after:.4f}'})
    if auc_delta is not None:
        summary_rows.append({'metric': 'auc_delta', 'value': auc_delta, 'notes': f'{auc_delta:+.4f}'})

    df_summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / 'comparative_evaluation_edit_samples.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\nEdit samples comparison exported to: {summary_path}")

    if 'per_sample_transitions' in comparison:
        df_transitions = pd.DataFrame(comparison['per_sample_transitions'])
        transitions_path = results_dir / 'edit_samples_transitions.csv'
        df_transitions.to_csv(transitions_path, index=False)
        print(f"Edit samples transitions exported to: {transitions_path}")

    return str(summary_path)


def export_projection_samples_comparison(
    comparison: Dict[str, Any],
    results_before: Dict[str, Any],
    results_after: Dict[str, Any],
    results_dir: str = "results"
) -> str:
    if comparison is None:
        print("[WARNING] No projection samples comparison to export")
        return None

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {'metric': 'accuracy_before', 'value': comparison['accuracy_before'],
         'notes': f"{comparison['accuracy_before']*100:.2f}%"},
        {'metric': 'accuracy_after', 'value': comparison['accuracy_after'],
         'notes': f"{comparison['accuracy_after']*100:.2f}%"},
        {'metric': 'accuracy_delta', 'value': comparison['accuracy_delta'],
         'notes': f"{comparison['accuracy_delta']*100:+.2f}%"},
        {'metric': 'num_total', 'value': comparison['num_total'],
         'notes': 'total projection samples (FT-Train)'},
        {'metric': 'num_stayed_correct', 'value': comparison['num_stayed_correct'],
         'notes': 'correct->correct (preserved)'},
        {'metric': 'num_broken', 'value': comparison['num_broken'],
         'notes': 'correct->wrong (regression)'},
        {'metric': 'num_fixed', 'value': comparison['num_fixed'],
         'notes': 'wrong->correct'},
        {'metric': 'num_stayed_wrong', 'value': comparison['num_stayed_wrong'],
         'notes': 'wrong->wrong'},
        {'metric': 'stability', 'value': comparison['stability'],
         'notes': f"{comparison['stability']*100:.1f}% (correct samples preserved)"},
        {'metric': 'regression_rate', 'value': comparison['regression_rate'],
         'notes': f"{comparison['regression_rate']*100:.1f}% (correct samples broken)"},
    ]

    auc_before = comparison.get('auc_before')
    auc_after = comparison.get('auc_after')
    auc_delta = comparison.get('auc_delta')
    if auc_before is not None:
        summary_rows.append({'metric': 'auc_before', 'value': auc_before, 'notes': f'{auc_before:.4f}'})
    if auc_after is not None:
        summary_rows.append({'metric': 'auc_after', 'value': auc_after, 'notes': f'{auc_after:.4f}'})
    if auc_delta is not None:
        summary_rows.append({'metric': 'auc_delta', 'value': auc_delta, 'notes': f'{auc_delta:+.4f}'})

    df_summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / 'comparative_evaluation_projection_samples.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\nProjection samples comparison exported to: {summary_path}")

    if 'per_class_stability' in comparison:
        class_rows = []
        for cls, stats in comparison['per_class_stability'].items():
            class_rows.append({
                'class': cls,
                'correct_before': stats['correct_before'],
                'correct_after': stats['correct_after'],
                'broken': stats['broken'],
                'stability': stats['stability']
            })
        df_class = pd.DataFrame(class_rows)
        class_path = results_dir / 'projection_samples_per_class_stability.csv'
        df_class.to_csv(class_path, index=False)
        print(f"Per-class stability exported to: {class_path}")

    return str(summary_path)



def export_baseline_summary(
    results: Dict[str, Any],
    results_dir,
    baseline_name: str
) -> str:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    if 'edit_samples' in results and results['edit_samples']:
        es = results['edit_samples']
        rows.append({
            'evaluation_level': 'edit_samples',
            'metric': 'accuracy_delta',
            'value': es.get('accuracy_delta', 0),
            'before': es.get('accuracy_before', 0),
            'after': es.get('accuracy_after', 0),
            'auc_before': es.get('auc_before', ''),
            'auc_after': es.get('auc_after', ''),
            'auc_delta': es.get('auc_delta', ''),
            'notes': f"Fixed: {es.get('num_fixed', 0)}, Broken: {es.get('num_broken', 0)}"
        })

    if 'ft_train_samples' in results and results['ft_train_samples']:
        ft = results['ft_train_samples']
        rows.append({
            'evaluation_level': 'ft_train_samples',
            'metric': 'accuracy_delta',
            'value': ft.get('accuracy_delta', 0),
            'before': ft.get('accuracy_orig', 0),
            'after': ft.get('accuracy_edit', 0),
            'auc_before': ft.get('auc_orig', ''),
            'auc_after': ft.get('auc_edit', ''),
            'auc_delta': ft.get('auc_delta', ''),
            'notes': f"Stability: {ft.get('stability', 0)*100:.1f}%"
        })

    if 'test_set' in results and results['test_set']:
        ts = results['test_set']
        rows.append({
            'evaluation_level': 'test_set',
            'metric': 'accuracy_delta',
            'value': ts.get('accuracy_delta', 0),
            'before': ts.get('accuracy_orig', 0),
            'after': ts.get('accuracy_edit', 0),
            'auc_before': ts.get('auc_orig', ''),
            'auc_after': ts.get('auc_edit', ''),
            'auc_delta': ts.get('auc_delta', ''),
            'notes': f"Regression: {ts.get('regression_rate', 0)*100:.1f}%"
        })

    if 'edit_discovery' in results and results['edit_discovery']:
        ed = results['edit_discovery']
        rows.append({
            'evaluation_level': 'edit_discovery_set',
            'metric': 'accuracy_delta',
            'value': ed.get('accuracy_delta', 0),
            'before': ed.get('accuracy_orig', 0),
            'after': ed.get('accuracy_edit', 0),
            'auc_before': ed.get('auc_orig', ''),
            'auc_after': ed.get('auc_edit', ''),
            'auc_delta': ed.get('auc_delta', ''),
            'notes': f"Fix rate: {ed.get('fix_rate', 0)*100:.1f}%"
        })

    df = pd.DataFrame(rows)
    csv_path = results_dir / f'baseline_{baseline_name}_summary.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nBaseline summary exported to: {csv_path}")
    return str(csv_path)


def evaluate_baseline_4level(
    model_original: nn.Module,
    model_baseline: nn.Module,
    edit_images: torch.Tensor,
    edit_labels: torch.Tensor,
    edit_indices: List[int],
    ft_train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    discovery_loader: torch.utils.data.DataLoader,
    device: torch.device = None,
    results_dir: str = "results",
    baseline_name: str = "baseline"
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"BASELINE 4-LEVEL EVALUATION: {baseline_name.upper()}")
    print("=" * 70)

    all_results = {}

    print("\n--- Level 1: Edit Samples ---")
    edit_results_before = evaluate_edit_samples(
        model_original, edit_images, edit_labels, edit_indices, device, "Before"
    )
    edit_results_after = evaluate_edit_samples(
        model_baseline, edit_images, edit_labels, edit_indices, device, "After"
    )
    edit_comparison = compare_edit_samples_before_after(
        edit_results_before, edit_results_after, edit_indices
    )
    print_edit_samples_comparison(edit_comparison, edit_results_before, edit_results_after)
    export_edit_samples_comparison(
        edit_comparison, edit_results_before, edit_results_after, results_dir
    )
    all_results['edit_samples'] = edit_comparison

    print("\n--- Level 2: FT-Train Samples ---")
    ft_train_results = evaluate_comparative(
        model_original, model_baseline, ft_train_loader, device,
        results_dir, set_name="FT-Train Samples"
    )
    all_results['ft_train_samples'] = ft_train_results

    print("\n--- Level 3: Test Set ---")
    test_results = evaluate_comparative(
        model_original, model_baseline, test_loader, device,
        results_dir, set_name="Test Set"
    )
    all_results['test_set'] = test_results

    print("\n--- Level 4: Edit-Discovery Set ---")
    discovery_results = evaluate_comparative(
        model_original, model_baseline, discovery_loader, device,
        results_dir, set_name="Edit-Discovery Set"
    )
    all_results['edit_discovery'] = discovery_results

    export_baseline_summary(all_results, results_dir, baseline_name)

    print("\n" + "=" * 70)
    print(f"BASELINE {baseline_name.upper()} - FINAL SUMMARY")
    print("=" * 70)

    def _fmt_auc(val):
        return f"{val:.4f}" if val is not None else "  N/A "

    print(f"\n{'Level':<25} {'Acc Before':>12} {'Acc After':>12} {'Delta':>10} {'AUC Before':>12} {'AUC After':>12} {'AUC Delta':>11}")
    print("-" * 95)

    if 'edit_samples' in all_results:
        es = all_results['edit_samples']
        print(f"{'Edit Samples':<25} {es['accuracy_before']*100:>11.2f}% "
              f"{es['accuracy_after']*100:>11.2f}% {es['accuracy_delta']*100:>+9.2f}%"
              f" {_fmt_auc(es.get('auc_before')):>12} {_fmt_auc(es.get('auc_after')):>12}"
              f" {_fmt_auc(es.get('auc_delta')):>11}")

    if 'ft_train_samples' in all_results:
        ft = all_results['ft_train_samples']
        print(f"{'FT-Train Samples':<25} {ft['accuracy_orig']*100:>11.2f}% "
              f"{ft['accuracy_edit']*100:>11.2f}% {ft['accuracy_delta']*100:>+9.2f}%"
              f" {_fmt_auc(ft.get('auc_orig')):>12} {_fmt_auc(ft.get('auc_edit')):>12}"
              f" {_fmt_auc(ft.get('auc_delta')):>11}")

    if 'test_set' in all_results:
        ts = all_results['test_set']
        print(f"{'Test Set':<25} {ts['accuracy_orig']*100:>11.2f}% "
              f"{ts['accuracy_edit']*100:>11.2f}% {ts['accuracy_delta']*100:>+9.2f}%"
              f" {_fmt_auc(ts.get('auc_orig')):>12} {_fmt_auc(ts.get('auc_edit')):>12}"
              f" {_fmt_auc(ts.get('auc_delta')):>11}")

    if 'edit_discovery' in all_results:
        ed = all_results['edit_discovery']
        print(f"{'Edit-Discovery Set':<25} {ed['accuracy_orig']*100:>11.2f}% "
              f"{ed['accuracy_edit']*100:>11.2f}% {ed['accuracy_delta']*100:>+9.2f}%"
              f" {_fmt_auc(ed.get('auc_orig')):>12} {_fmt_auc(ed.get('auc_edit')):>12}"
              f" {_fmt_auc(ed.get('auc_delta')):>11}")

    print("=" * 70)

    return all_results


def main():
    print("=" * 70)
    print("ViT Model Editing Pipeline - Evaluator")
    print("=" * 70)
    
    from transformers import ViTForImageClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=9,
        ignore_mismatched_sizes=True
    )
    
    evaluator = Evaluator(model, device)
    
    from torch.utils.data import DataLoader, TensorDataset
    
    dummy_images = torch.randn(100, 3, 224, 224)
    dummy_labels = torch.randint(0, 9, (100,))
    
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=16)
    
    print("\nRunning inference...")
    evaluator.run_inference(dummy_loader)
    
    evaluator.print_summary()
    
    evaluator.export_confusion_matrix()
    evaluator.export_evaluation_report()
    evaluator.export_predictions()
    
    print("\n[OK] Evaluator test complete!")


if __name__ == "__main__":
    main()
