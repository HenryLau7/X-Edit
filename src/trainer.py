import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import transforms

from data_handler import DataHandler, MedMNISTDataset


class Trainer:

    def __init__(
        self,
        model_name: str = "vit-base-patch16-224",
        model_short: str = "vit-base",
        num_classes: int = 9,
        dataset_name: str = "pathmnist",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: str = None,
        n_channels: int = 3
    ):
        self.model_name = model_name
        self.model_short = model_short
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.n_channels = n_channels

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = []

        self.regularizer = None

        self.checkpoint_name = f"{self.model_short}_{self.dataset_name}_finetuned.pt"
        self.best_checkpoint_name = f"{self.model_short}_{self.dataset_name}_best.pt"
        
    def setup_model(self) -> nn.Module:
        print(f"\nLoading model: {self.model_name}")
        
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def get_transforms(self) -> transforms.Compose:
        transform_list = [
            transforms.Resize((224, 224)),
        ]

        if self.n_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transforms.Compose(transform_list)
    
    def setup_training(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 2,
        total_epochs: int = 10
    ):
        if self.model is None:
            self.setup_model()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_epochs - warmup_epochs,
            T_mult=1
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining setup:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Epochs: {total_epochs}")
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits = outputs.logits
            
            loss = self.criterion(logits, labels)

            if self.regularizer is not None:
                loss = loss + self.regularizer()

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_acc': 100. * correct / total
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        desc: str = "Eval"
    ) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc=f"[{desc}]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            logits = outputs.logits
            
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(val_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_acc': 100. * correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        save_best: bool = True
    ) -> Dict[str, Any]:
        self.setup_training(learning_rate=learning_rate, total_epochs=epochs)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader, epoch)
            
            val_metrics = self.evaluate(val_loader, desc="Val")
            
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            metrics = {
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics,
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': epoch_time
            }
            self.training_history.append(metrics)
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.2f}%")
            print(f"  Time: {epoch_time:.1f}s")
            
            if save_best and val_metrics['val_acc'] > self.best_acc:
                self.best_acc = val_metrics['val_acc']
                self.save_checkpoint(
                    filepath=self.checkpoint_dir / self.checkpoint_name,
                    is_best=True
                )
                print(f"  [OK] New best model saved! (Acc: {self.best_acc:.2f}%)")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best validation accuracy: {self.best_acc:.2f}%")
        print(f"{'='*60}")
        
        self.export_training_log()
        
        return {
            'history': self.training_history,
            'best_acc': self.best_acc,
            'total_time': total_time
        }
    
    def save_checkpoint(
        self,
        filepath: Path = None,
        is_best: bool = False
    ):
        if filepath is None:
            filepath = self.checkpoint_dir / self.checkpoint_name

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_acc': self.best_acc,
            'training_history': self.training_history,
            'config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'dataset_name': self.dataset_name,
                'n_channels': self.n_channels,
                'device': str(self.device)
            }
        }

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = self.checkpoint_dir / self.best_checkpoint_name
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(
        self,
        filepath: Path = None,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        if filepath is None:
            filepath = self.checkpoint_dir / self.checkpoint_name

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        print(f"Loading checkpoint from: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            self.setup_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and checkpoint.get('optimizer_state_dict'):
            if self.optimizer is None:
                self.setup_training()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"  Restored from epoch {self.current_epoch+1}")
        print(f"  Best accuracy: {self.best_acc:.2f}%")
        
        return checkpoint
    
    def export_training_log(self) -> str:
        if not self.training_history:
            print("No training history to export.")
            return None

        df = pd.DataFrame(self.training_history)
        csv_path = self.log_dir / f"{self.dataset_name}_training_metrics.csv"
        df.to_csv(csv_path, index=False)

        print(f"Training log exported to: {csv_path}")
        return str(csv_path)
    
    @torch.no_grad()
    def get_predictions(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(self.device)
            
            outputs = self.model(images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return (
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs)
        )
    
    def find_misclassified(
        self,
        dataloader: DataLoader,
        max_samples: int = None
    ) -> Dict[str, Any]:
        dataset = dataloader.dataset
        unshuffled_loader = DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
        )

        preds, labels, probs = self.get_predictions(unshuffled_loader)

        misclassified_mask = preds != labels
        misclassified_indices = np.where(misclassified_mask)[0]

        if max_samples and len(misclassified_indices) > max_samples:
            misclassified_indices = misclassified_indices[:max_samples]

        print(f"\nFound {len(misclassified_indices)} misclassified samples")
        print(f"  Error rate: {100.*len(misclassified_indices)/len(labels):.2f}%")

        if len(misclassified_indices) > 0:
            sample_idx = misclassified_indices[0]
            _, dataset_label = dataset[sample_idx]
            if isinstance(dataset_label, torch.Tensor):
                dataset_label = dataset_label.item()
            if dataset_label != labels[sample_idx]:
                print(f"  WARNING: Index mismatch detected! dataset[{sample_idx}].label={dataset_label}, labels[{sample_idx}]={labels[sample_idx]}")

        return {
            'indices': misclassified_indices,
            'predictions': preds[misclassified_indices],
            'true_labels': labels[misclassified_indices],
            'probabilities': probs[misclassified_indices],
            'total_samples': len(labels),
            'error_count': len(misclassified_indices)
        }




    def train_from_scratch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        checkpoint_suffix: str = "retrained"
    ) -> Dict[str, Any]:
        print("\nReinitializing model from pretrained weights...")
        self.model = None
        self.setup_model()

        self.checkpoint_name = f"{self.model_short}_{self.dataset_name}_{checkpoint_suffix}.pt"
        self.best_checkpoint_name = f"{self.model_short}_{self.dataset_name}_{checkpoint_suffix}_best.pt"

        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = []

        return self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate
        )

    def finetune_on_samples(
        self,
        finetune_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 1e-5,
        checkpoint_suffix: str = "finetuned_on_errors"
    ) -> Dict[str, Any]:
        finetuned_path = self.checkpoint_dir / f"{self.model_short}_{self.dataset_name}_finetuned.pt"
        if not finetuned_path.exists():
            raise FileNotFoundError(
                f"Finetuned model not found: {finetuned_path}. "
                f"Run --stage train first."
            )

        print(f"\nLoading finetuned model from: {finetuned_path}")
        self.load_checkpoint(filepath=finetuned_path, load_optimizer=False)

        self.checkpoint_name = f"{self.model_short}_{self.dataset_name}_{checkpoint_suffix}.pt"
        self.best_checkpoint_name = f"{self.model_short}_{self.dataset_name}_{checkpoint_suffix}_best.pt"

        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = []

        return self.train(
            train_loader=finetune_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate
        )

    def finetune_with_l2_reg(
        self,
        finetune_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 1e-5,
        l2_lambda: float = 0.01,
        checkpoint_suffix: str = "l2reg"
    ) -> Dict[str, Any]:
        finetuned_path = self.checkpoint_dir / f"{self.model_short}_{self.dataset_name}_finetuned.pt"
        if not finetuned_path.exists():
            raise FileNotFoundError(
                f"Finetuned model not found: {finetuned_path}. "
                f"Run --stage train first."
            )

        print(f"\nLoading finetuned model from: {finetuned_path}")
        self.load_checkpoint(filepath=finetuned_path, load_optimizer=False)

        original_state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        device = self.device

        def l2_regularizer():
            l2_loss = torch.tensor(0.0, device=device)
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in original_state:
                    l2_loss = l2_loss + ((param - original_state[name]) ** 2).sum()
            return l2_lambda * l2_loss

        self.regularizer = l2_regularizer

        self.checkpoint_name = f"{self.model_short}_{self.dataset_name}_{checkpoint_suffix}.pt"
        self.best_checkpoint_name = (
            f"{self.model_short}_{self.dataset_name}_{checkpoint_suffix}_best.pt"
        )

        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = []

        try:
            results = self.train(
                train_loader=finetune_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate
            )
        finally:
            self.regularizer = None

        return results

    def compute_fisher_information_full(
        self,
        stats_loader: DataLoader,
        num_samples: int = 500
    ) -> Dict[str, torch.Tensor]:
        print(f"\nComputing full-model Fisher information from {num_samples} samples...")
        self.model.eval()

        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

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

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            count += images.shape[0]

        for name in fisher:
            fisher[name] /= count

        total_fisher_norm = sum(f.norm().item() for f in fisher.values())
        print(f"  Fisher computed for {len(fisher)} parameters (total norm: {total_fisher_norm:.4f})")

        return fisher

    def finetune_with_ewc(
        self,
        finetune_loader: DataLoader,
        val_loader: DataLoader,
        ft_train_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 1e-5,
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 500,
        checkpoint_suffix: str = "ewc"
    ) -> Dict[str, Any]:
        finetuned_path = self.checkpoint_dir / f"{self.model_short}_{self.dataset_name}_finetuned.pt"
        if not finetuned_path.exists():
            raise FileNotFoundError(
                f"Finetuned model not found: {finetuned_path}. "
                f"Run --stage train first."
            )

        print(f"\nLoading finetuned model from: {finetuned_path}")
        self.load_checkpoint(filepath=finetuned_path, load_optimizer=False)

        fisher = self.compute_fisher_information_full(ft_train_loader, fisher_samples)

        original_state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        device = self.device

        def ewc_regularizer():
            ewc_loss = torch.tensor(0.0, device=device)
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in fisher:
                    ewc_loss = ewc_loss + (
                        fisher[name] * (param - original_state[name]) ** 2
                    ).sum()
            return ewc_lambda * ewc_loss

        self.regularizer = ewc_regularizer

        self.checkpoint_name = f"{self.model_short}_{self.dataset_name}_{checkpoint_suffix}.pt"
        self.best_checkpoint_name = (
            f"{self.model_short}_{self.dataset_name}_{checkpoint_suffix}_best.pt"
        )

        self.current_epoch = 0
        self.best_acc = 0.0
        self.training_history = []

        try:
            results = self.train(
                train_loader=finetune_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate
            )
        finally:
            self.regularizer = None

        return results


def main():
    print("=" * 70)
    print("ViT Model Editing Pipeline - Trainer (Multi-Dataset Support)")
    print("=" * 70)

    dataset_name = "pathmnist"
    data_handler = DataHandler(
        dataset_name=dataset_name,
        ft_train_ratio=0.9,
        random_seed=42
    )
    data_handler.load_data()
    data_handler.create_resplit()

    trainer = Trainer(
        model_name="vit-base-patch16-224",
        num_classes=data_handler.n_classes,
        dataset_name=dataset_name,
        n_channels=data_handler.n_channels
    )

    trainer.setup_model()

    transform = trainer.get_transforms()
    dataloaders = data_handler.get_dataloaders(
        batch_size=32,
        transform=transform
    )

    print("\nStarting training (test run with 2 epochs)...")
    results = trainer.train(
        train_loader=dataloaders['ft_train'],
        val_loader=dataloaders['val'],
        epochs=2,
        learning_rate=1e-4
    )

    print("\n[OK] Trainer test complete!")
    print(f"  Dataset: {dataset_name}")
    print(f"  Classes: {data_handler.n_classes}")
    print(f"  Best accuracy: {results['best_acc']:.2f}%")
    print(f"  Checkpoint: {trainer.checkpoint_dir}/{trainer.checkpoint_name}")


if __name__ == "__main__":
    main()
