import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from copy import deepcopy
import contextlib
from collections import OrderedDict


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    
    def __init__(
        self,
        module: nn.Module,
        layers: List[str],
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = True,
        detach: bool = True
    ):
        super().__init__()
        self.module = module
        self.layers = layers
        self.retain_output = retain_output
        self.retain_input = retain_input
        self.clone = clone
        self.detach = detach
        self._hooks = []
        
        for layer in layers:
            self[layer] = {'output': None, 'input': None}
    
    def __enter__(self):
        for layer in self.layers:
            target = self._get_module(self.module, layer)
            hook = target.register_forward_hook(self._make_hook_fn(layer))
            self._hooks.append(hook)
        return self
    
    def __exit__(self, *args):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def _make_hook_fn(self, layer_name):
        def hook_fn(module, input, output):
            if self.retain_input:
                inp = input[0] if isinstance(input, tuple) else input
                self[layer_name]['input'] = self._process(inp)
            if self.retain_output:
                out = output[0] if isinstance(output, tuple) else output
                self[layer_name]['output'] = self._process(out)
        return hook_fn
    
    def _process(self, x):
        if x is None:
            return None
        if self.clone:
            x = x.clone()
        if self.detach:
            x = x.detach()
        return x
    
    @staticmethod
    def _get_module(model, name):
        for part in name.split('.'):
            model = getattr(model, part)
        return model


class XEditHyperParams:
    
    def __init__(
        self,
        layers: List[int] = None,
        rewrite_module_tmp: str = "vit.encoder.layer.{}.output.dense",
        layer_module_tmp: str = "vit.encoder.layer.{}",
        v_num_grad_steps: int = 25,
        v_lr: float = 0.1,
        v_weight_decay: float = 0.01,
        nullspace_threshold: float = 1e-2,
        L2: float = 1e-4,
        clamp_norm_factor: float = 0.75,
        fact_token: str = "cls",
    ):
        self.layers = layers if layers is not None else [8, 9, 10, 11]
        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp
        self.v_num_grad_steps = v_num_grad_steps
        self.v_lr = v_lr
        self.v_weight_decay = v_weight_decay
        self.nullspace_threshold = nullspace_threshold
        self.L2 = L2
        self.clamp_norm_factor = clamp_norm_factor
        self.fact_token = fact_token


class HeadEditHyperParams:

    def __init__(
        self,
        num_steps: int = 50,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 500,
        closed_form: bool = False,
        reg_lambda: float = 1.0,
    ):
        self.num_steps = num_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.closed_form = closed_form
        self.reg_lambda = reg_lambda


class KCollector:
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        hparams: XEditHyperParams
    ):
        self.model = model
        self.device = device
        self.hparams = hparams
    
    def compute_ks(
        self,
        images: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        self.model.eval()

        module_name = self.hparams.rewrite_module_tmp.format(layer)

        with TraceDict(self.model, [module_name], retain_input=True) as traces:
            with torch.no_grad():
                images = images.to(self.device)
                _ = self.model(images)

                k_input = traces[module_name]['input']

        if self.hparams.fact_token == "cls":
            k = k_input[:, 0, :]
        else:
            k = k_input.mean(dim=1)

        return k.T
    
    def compute_current_output(
        self,
        images: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        self.model.eval()
        
        module_name = self.hparams.rewrite_module_tmp.format(layer)
        
        with TraceDict(self.model, [module_name], retain_output=True) as traces:
            with torch.no_grad():
                images = images.to(self.device)
                _ = self.model(images)
                
                output = traces[module_name]['output']
        
        if self.hparams.fact_token == "cls":
            return output[:, 0, :]
        else:
            return output.mean(dim=1)


class ZComputer:
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        hparams: XEditHyperParams
    ):
        self.model = model
        self.device = device
        self.hparams = hparams
    
    def compute_target_z(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        self.model.eval()
        images = images.to(self.device)
        true_labels = true_labels.to(self.device)
        
        module_name = self.hparams.rewrite_module_tmp.format(layer)
        
        with TraceDict(self.model, [module_name], retain_output=True) as traces:
            with torch.no_grad():
                _ = self.model(images)
                current_z = traces[module_name]['output'].clone()
        
        if self.hparams.fact_token == "cls":
            current_z_token = current_z[:, 0, :].clone()
        else:
            current_z_token = current_z.mean(dim=1).clone()
        
        delta = torch.zeros_like(current_z_token, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.hparams.v_lr)
        
        for step in range(self.hparams.v_num_grad_steps):
            optimizer.zero_grad()
            
            def edit_output(output, layer_name):
                if layer_name == module_name:
                    new_output = output.clone()
                    if self.hparams.fact_token == "cls":
                        new_output[:, 0, :] = current_z_token + delta
                    else:
                        new_output = new_output + delta.unsqueeze(1)
                    return new_output
                return output
            
            with TraceDict(
                self.model,
                [module_name],
                retain_output=True
            ) as traces:
                target_module = TraceDict._get_module(self.model, module_name)
                
                def hook_fn(module, input, output):
                    return edit_output(output, module_name)
                
                hook = target_module.register_forward_hook(hook_fn)
                
                try:
                    outputs = self.model(images)
                    logits = outputs.logits
                finally:
                    hook.remove()
            
            loss = nn.CrossEntropyLoss()(logits, true_labels)
            
            loss = loss + self.hparams.v_weight_decay * torch.norm(delta) ** 2
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                max_norm = self.hparams.clamp_norm_factor * current_z_token.norm(dim=-1, keepdim=True)
                delta_norm = delta.norm(dim=-1, keepdim=True)
                scale = torch.clamp(max_norm / (delta_norm + 1e-8), max=1.0)
                delta.data = delta.data * scale
        
        target_z = current_z_token + delta.detach()
        
        return target_z


class NullSpaceProjector:
    
    def __init__(
        self,
        threshold: float = 1e-2
    ):
        self.threshold = threshold
    
    def compute_projection_matrix(
        self,
        K0: torch.Tensor
    ) -> torch.Tensor:
        num_samples = K0.shape[1]
        cov = (K0 @ K0.T) / num_samples

        U, S, Vh = torch.linalg.svd(cov)
        
        null_mask = S < self.threshold
        
        if not null_mask.any():
            print("Warning: No null space found, using identity projection")
            return torch.eye(K0.shape[0], device=K0.device, dtype=K0.dtype)
        
        U_null = U[:, null_mask]
        
        P = U_null @ U_null.T
        
        return P
    
    def compute_from_random_samples(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer: int,
        hparams: XEditHyperParams,
        device: torch.device,
        num_samples: int = 1000,
        return_sample_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        collector = KCollector(model, device, hparams)

        K_list = []
        total = 0

        sample_images = [] if return_sample_info else None
        sample_labels = [] if return_sample_info else None
        batch_start_indices = [] if return_sample_info else None
        current_idx = 0

        for images, labels in tqdm(dataloader, desc="Collecting K0"):
            if total >= num_samples:
                break

            k = collector.compute_ks(images, layer)
            K_list.append(k.cpu())

            if return_sample_info:
                sample_images.append(images.cpu())
                sample_labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels))
                batch_start_indices.append(current_idx)
                current_idx += images.shape[0]

            total += images.shape[0]

        K0 = torch.cat(K_list, dim=1).to(device)
        P = self.compute_projection_matrix(K0)

        sample_info = None
        if return_sample_info:
            all_images = torch.cat(sample_images, dim=0)[:num_samples]
            all_labels = torch.cat(sample_labels, dim=0)[:num_samples]
            sample_info = {
                'images': all_images,
                'labels': all_labels,
                'num_samples': min(total, num_samples),
                'layer': layer
            }

        return P, sample_info


class Editor:

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        hparams: XEditHyperParams = None,
        log_dir: str = "logs",
        dataset_name: str = "pathmnist",
        model_short: str = "vit-base"
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)
        self.hparams = hparams if hparams else XEditHyperParams()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.model_short = model_short

        self.k_collector = KCollector(model, device, self.hparams)
        self.z_computer = ZComputer(model, device, self.hparams)
        self.projector = NullSpaceProjector(self.hparams.nullspace_threshold)

        self.cache_KKT = {}
        self.P = {}

        self.edit_history = []

        self.projection_samples = None
    
    def precompute_projection(
        self,
        stats_loader: torch.utils.data.DataLoader,
        num_samples: int = 1000,
        track_samples: bool = True
    ):
        print("\nPrecomputing projection matrices (using FT-Train for stats)...")

        first_layer = True

        for layer in tqdm(self.hparams.layers, desc="Layers"):
            P, sample_info = self.projector.compute_from_random_samples(
                model=self.model,
                dataloader=stats_loader,
                layer=layer,
                hparams=self.hparams,
                device=self.device,
                num_samples=num_samples,
                return_sample_info=(track_samples and first_layer)
            )
            self.P[layer] = P

            if track_samples and first_layer and sample_info is not None:
                self.projection_samples = sample_info
                print(f"\n  Tracked {sample_info['num_samples']} FT-Train samples for projection matrix")
                first_layer = False

            hidden_size = P.shape[0]
            self.cache_KKT[layer] = torch.zeros(
                (hidden_size, hidden_size),
                device=self.device,
                dtype=P.dtype
            )

        print(f"Projection matrices computed for layers: {self.hparams.layers}")

    def get_projection_samples(self) -> Optional[Dict[str, Any]]:
        return self.projection_samples

    def apply_edit(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        sample_indices: List[int] = None
    ) -> Dict[str, Any]:
        images = images.to(self.device)
        true_labels = true_labels.to(self.device)
        
        if sample_indices is None:
            sample_indices = list(range(len(images)))
        
        edit_info = {
            'sample_indices': sample_indices,
            'num_samples': len(images),
            'layer_updates': {}
        }
        
        num_layers = len(self.hparams.layers)
        
        target_zs = {}
        for layer in self.hparams.layers:
            target_z = self.z_computer.compute_target_z(images, true_labels, layer)
            target_zs[layer] = target_z
        
        layer_K_vectors = {}

        for i, layer in enumerate(self.hparams.layers):
            print(f"\nEditing layer {layer} ({i+1}/{num_layers})...")

            K = self.k_collector.compute_ks(images, layer)
            layer_K_vectors[layer] = K

            current_z = self.k_collector.compute_current_output(images, layer)

            target_z = target_zs[layer]

            R = (target_z - current_z).T

            R = R / (num_layers - i)

            P = self.P.get(layer)
            if P is None:
                P = torch.eye(K.shape[0], device=self.device, dtype=K.dtype)

            cache_KKT = self.cache_KKT.get(layer, torch.zeros_like(P))

            KKT = K @ K.T

            A = P @ (KKT + cache_KKT) + self.hparams.L2 * torch.eye(
                K.shape[0], device=self.device, dtype=K.dtype
            )

            B = P @ K @ R.T

            try:
                delta = torch.linalg.solve(A, B)
            except Exception as e:
                print(f"Warning: Solve failed, using pseudoinverse: {e}")
                A_inv = torch.linalg.pinv(A)
                delta = A_inv @ B

            module_name = self.hparams.rewrite_module_tmp.format(layer)
            target_module = self._get_module(self.model, module_name)

            with torch.no_grad():
                old_weight = target_module.weight.data.clone()

                if delta.shape == old_weight.shape:
                    upd_matrix = delta
                elif delta.T.shape == old_weight.shape:
                    upd_matrix = delta.T
                else:
                    raise ValueError(
                        f"Update matrix shape {delta.shape} does not match "
                        f"weight shape {old_weight.shape} or its transpose. "
                        f"K shape: {K.shape}, module: {module_name}"
                    )

                target_module.weight.data = old_weight + upd_matrix
                update_norm = torch.norm(upd_matrix).item()

            edit_info['layer_updates'][layer] = {
                'update_norm': update_norm,
                'K_norm': K.norm().item(),
                'R_norm': R.norm().item()
            }

            print(f"  Update norm: {update_norm:.6f}")

        for layer in self.hparams.layers:
            K = layer_K_vectors[layer]
            KKT = K @ K.T
            cache_KKT = self.cache_KKT.get(layer, torch.zeros_like(KKT))
            self.cache_KKT[layer] = cache_KKT + KKT
        
        self.edit_history.append(edit_info)
        
        return edit_info
    
    def apply_batch_edit(
        self,
        dataloader: torch.utils.data.DataLoader,
        misclassified_info: Dict[str, Any],
        max_edits: int = 50
    ) -> List[Dict[str, Any]]:
        indices = misclassified_info['indices'][:max_edits]
        
        results = []
        
        dataset = dataloader.dataset
        
        for idx in tqdm(indices, desc="Applying edits"):
            image, label = dataset[idx]
            image = image.unsqueeze(0)
            label = torch.tensor([label])
            
            result = self.apply_edit(
                images=image,
                true_labels=label,
                sample_indices=[idx]
            )
            results.append(result)
        
        return results
    
    def export_edit_log(self, filename: str = "edit_log.csv") -> str:
        if not self.edit_history:
            print("No edit history to export.")
            return None
        
        rows = []
        for edit in self.edit_history:
            for layer, info in edit['layer_updates'].items():
                rows.append({
                    'sample_indices': str(edit['sample_indices']),
                    'layer': layer,
                    'update_norm': info['update_norm'],
                    'K_norm': info['K_norm'],
                    'R_norm': info['R_norm']
                })
        
        df = pd.DataFrame(rows)
        csv_path = self.log_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"Edit log exported to: {csv_path}")
        return str(csv_path)
    
    def save_edited_model(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = Path("checkpoints") / f"{self.model_short}_{self.dataset_name}_edited.pt"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'edit_history': self.edit_history,
            'hparams': vars(self.hparams),
            'dataset_name': self.dataset_name
        }, filepath)

        print(f"Edited model saved to: {filepath}")
        return str(filepath)
    
    @staticmethod
    def _get_module(model, name):
        for part in name.split('.'):
            model = getattr(model, part)
        return model


class HeadEditor:

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        hparams: HeadEditHyperParams = None,
        log_dir: str = "logs",
        dataset_name: str = "pathmnist",
        model_short: str = "vit-base"
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)
        self.hparams = hparams if hparams else HeadEditHyperParams()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.model_short = model_short

        self.fisher_weight = None
        self.fisher_bias = None
        self.original_weight = None
        self.original_bias = None

        self.edit_history = []

    def compute_fisher_information(
        self,
        stats_loader: torch.utils.data.DataLoader,
        num_samples: int = None
    ):
        if num_samples is None:
            num_samples = self.hparams.fisher_samples

        print(f"\nComputing Fisher information from {num_samples} samples (FT-Train)...")
        self.model.eval()

        self.original_weight = self.model.classifier.weight.data.clone()
        self.original_bias = self.model.classifier.bias.data.clone()

        self.fisher_weight = torch.zeros_like(self.model.classifier.weight)
        self.fisher_bias = torch.zeros_like(self.model.classifier.bias)

        count = 0
        for images, labels in tqdm(stats_loader, desc="Computing Fisher"):
            if count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            outputs = self.model(images)

            log_probs = torch.log_softmax(outputs.logits, dim=1)
            loss = nn.functional.nll_loss(log_probs, labels)
            loss.backward()

            if self.model.classifier.weight.grad is not None:
                self.fisher_weight += self.model.classifier.weight.grad.data ** 2
            if self.model.classifier.bias.grad is not None:
                self.fisher_bias += self.model.classifier.bias.grad.data ** 2

            count += images.shape[0]

        self.fisher_weight /= count
        self.fisher_bias /= count

        print(f"Fisher information computed (weight norm: {self.fisher_weight.norm():.4f})")

    def _compute_ewc_loss(self) -> torch.Tensor:
        if self.fisher_weight is None or self.original_weight is None:
            return torch.tensor(0.0, device=self.device)

        weight_diff = self.model.classifier.weight - self.original_weight
        bias_diff = self.model.classifier.bias - self.original_bias

        ewc_loss = (
            (self.fisher_weight * weight_diff ** 2).sum() +
            (self.fisher_bias * bias_diff ** 2).sum()
        )

        return ewc_loss

    def apply_edit(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        sample_indices: List[int] = None
    ) -> Dict[str, Any]:
        if self.hparams.closed_form:
            return self.apply_closed_form_edit(images, true_labels, sample_indices)

        images = images.to(self.device)
        true_labels = true_labels.to(self.device)

        if sample_indices is None:
            sample_indices = list(range(len(images)))

        pre_weight = self.model.classifier.weight.data.clone()
        pre_bias = self.model.classifier.bias.data.clone()

        for name, param in self.model.named_parameters():
            param.requires_grad = 'classifier' in name

        self.model.eval()
        with torch.no_grad():
            vit_outputs = self.model.vit(images)
            cls_representations = vit_outputs.last_hidden_state[:, 0, :]

        with torch.no_grad():
            pre_logits = self.model.classifier(cls_representations)
            pre_preds = pre_logits.argmax(dim=1)

        optimizer = torch.optim.Adam(
            self.model.classifier.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        losses = []
        for step in range(self.hparams.num_steps):
            optimizer.zero_grad()

            logits = self.model.classifier(cls_representations)

            ce_loss = nn.CrossEntropyLoss()(logits, true_labels)

            ewc_loss = self._compute_ewc_loss()

            loss = ce_loss + self.hparams.ewc_lambda * ewc_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        for param in self.model.parameters():
            param.requires_grad = True

        with torch.no_grad():
            post_logits = self.model.classifier(cls_representations)
            post_preds = post_logits.argmax(dim=1)

        weight_change = (self.model.classifier.weight.data - pre_weight).norm().item()
        bias_change = (self.model.classifier.bias.data - pre_bias).norm().item()

        corrected = (post_preds == true_labels).sum().item()
        total = len(true_labels)

        edit_info = {
            'sample_indices': sample_indices,
            'num_samples': total,
            'corrected': corrected,
            'success_rate': corrected / total,
            'weight_change_norm': weight_change,
            'bias_change_norm': bias_change,
            'final_loss': losses[-1] if losses else 0,
            'pre_predictions': pre_preds.cpu().tolist(),
            'post_predictions': post_preds.cpu().tolist(),
            'true_labels': true_labels.cpu().tolist(),
            'method': 'gradient'
        }

        self.edit_history.append(edit_info)

        print(f"  Corrected: {corrected}/{total} ({100*corrected/total:.1f}%)")
        print(f"  Weight change: {weight_change:.6f}, Bias change: {bias_change:.6f}")

        return edit_info

    def apply_closed_form_edit(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        sample_indices: List[int] = None
    ) -> Dict[str, Any]:
        images = images.to(self.device)
        true_labels = true_labels.to(self.device)

        if sample_indices is None:
            sample_indices = list(range(len(images)))

        pre_weight = self.model.classifier.weight.data.clone()
        pre_bias = self.model.classifier.bias.data.clone()

        self.model.eval()
        with torch.no_grad():
            vit_outputs = self.model.vit(images)
            cls_representations = vit_outputs.last_hidden_state[:, 0, :]

            pre_logits = self.model.classifier(cls_representations)
            pre_preds = pre_logits.argmax(dim=1)

        X = cls_representations.T
        num_classes = self.model.classifier.weight.shape[0]
        Y = nn.functional.one_hot(true_labels, num_classes=num_classes).float().T

        W_old = self.model.classifier.weight.data

        lambda_reg = self.hparams.reg_lambda

        XXT = X @ X.T
        YXT = Y @ X.T

        A = XXT + lambda_reg * torch.eye(X.shape[0], device=self.device, dtype=X.dtype)
        B = YXT + lambda_reg * W_old

        try:
            W_new = torch.linalg.solve(A.T, B.T).T
        except Exception as e:
            print(f"Warning: Closed-form solve failed, using pseudoinverse: {e}")
            A_inv = torch.linalg.pinv(A)
            W_new = B @ A_inv

        with torch.no_grad():
            self.model.classifier.weight.data = W_new

        with torch.no_grad():
            post_logits = self.model.classifier(cls_representations)
            post_preds = post_logits.argmax(dim=1)

        weight_change = (self.model.classifier.weight.data - pre_weight).norm().item()

        corrected = (post_preds == true_labels).sum().item()
        total = len(true_labels)

        edit_info = {
            'sample_indices': sample_indices,
            'num_samples': total,
            'corrected': corrected,
            'success_rate': corrected / total,
            'weight_change_norm': weight_change,
            'bias_change_norm': 0.0,
            'final_loss': 0,
            'pre_predictions': pre_preds.cpu().tolist(),
            'post_predictions': post_preds.cpu().tolist(),
            'true_labels': true_labels.cpu().tolist(),
            'method': 'closed_form'
        }

        self.edit_history.append(edit_info)

        print(f"  Corrected: {corrected}/{total} ({100*corrected/total:.1f}%)")
        print(f"  Weight change: {weight_change:.6f}")

        return edit_info

    def apply_batch_edit(
        self,
        dataloader: torch.utils.data.DataLoader,
        misclassified_info: Dict[str, Any],
        max_edits: int = 50
    ) -> List[Dict[str, Any]]:
        indices = misclassified_info['indices'][:max_edits]
        dataset = dataloader.dataset

        images_list = []
        labels_list = []

        for idx in indices:
            image, label = dataset[idx]
            images_list.append(image)
            labels_list.append(label)

        images = torch.stack(images_list)
        labels = torch.tensor(labels_list)

        result = self.apply_edit(
            images=images,
            true_labels=labels,
            sample_indices=indices
        )

        return [result]

    def export_edit_log(self, filename: str = "head_edit_log.csv") -> str:
        if not self.edit_history:
            print("No edit history to export.")
            return None

        rows = []
        for edit in self.edit_history:
            rows.append({
                'sample_indices': str(edit['sample_indices']),
                'num_samples': edit['num_samples'],
                'corrected': edit['corrected'],
                'success_rate': edit['success_rate'],
                'weight_change_norm': edit['weight_change_norm'],
                'bias_change_norm': edit['bias_change_norm'],
                'method': edit['method']
            })

        df = pd.DataFrame(rows)
        csv_path = self.log_dir / filename
        df.to_csv(csv_path, index=False)

        print(f"Head edit log exported to: {csv_path}")
        return str(csv_path)

    def save_edited_model(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = Path("checkpoints") / f"{self.model_short}_{self.dataset_name}_head_edited.pt"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'edit_history': self.edit_history,
            'hparams': vars(self.hparams),
            'dataset_name': self.dataset_name,
            'fisher_weight': self.fisher_weight,
            'fisher_bias': self.fisher_bias,
            'original_weight': self.original_weight,
            'original_bias': self.original_bias
        }, filepath)

        print(f"Head-edited model saved to: {filepath}")
        return str(filepath)


def main():
    print("=" * 70)
    print("ViT Model Editing Pipeline - Weight Editor (X-Edit)")
    print("=" * 70)
    
    from transformers import ViTForImageClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=9,
        ignore_mismatched_sizes=True
    )
    
    hparams = XEditHyperParams(
        layers=[9, 10, 11],
        v_num_grad_steps=10,
        L2=1e-3
    )
    
    editor = Editor(
        model=model,
        device=device,
        hparams=hparams
    )
    
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1])
    
    print("\nApplying test edit...")
    result = editor.apply_edit(
        images=dummy_images,
        true_labels=dummy_labels,
        sample_indices=[0, 1]
    )
    
    print(f"\nEdit result: {result}")
    
    editor.export_edit_log()
    editor.save_edited_model()
    
    print("\n[OK] Editor test complete!")


if __name__ == "__main__":
    main()
