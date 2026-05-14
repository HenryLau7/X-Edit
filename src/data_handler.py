import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from PIL import Image
from sklearn.model_selection import train_test_split


MEDMNIST_INFO = {
    "pathmnist": {
        "n_channels": 3,
        "n_classes": 9,
        "data_file": "pathmnist_224.npz",
        "task": "multi-class",
        "description": "Colon Pathology - 9 tissue types",
        "class_names": [
            "Adipose", "Background", "Debris", "Lymphocytes", "Mucus",
            "Smooth Muscle", "Normal Colon Mucosa", "Cancer-associated Stroma",
            "Colorectal Adenocarcinoma Epithelium"
        ]
    },
    "dermamnist": {
        "n_channels": 3,
        "n_classes": 7,
        "data_file": "dermamnist_224.npz",
        "task": "multi-class",
        "description": "Dermatoscopy - 7 skin lesion types (Imbalanced)",
        "class_names": [
            "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
            "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesions"
        ]
    },
    "retinamnist": {
        "n_channels": 3,
        "n_classes": 5,
        "data_file": "retinamnist_224.npz",
        "task": "ordinal-regression",
        "description": "Retinal Fundus - 5 diabetic retinopathy grades (Fine-grained)",
        "class_names": [
            "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"
        ]
    },
    "organamnist": {
        "n_channels": 1,
        "n_classes": 11,
        "data_file": "organamnist_224.npz",
        "task": "multi-class",
        "description": "Abdominal CT - 11 organ types (Grayscale/Shape)",
        "class_names": [
            "Bladder", "Femur-Left", "Femur-Right", "Heart", "Kidney-Left",
            "Kidney-Right", "Liver", "Lung-Left", "Lung-Right", "Spleen", "Pancreas"
        ]
    },
    "bloodmnist": {
        "n_channels": 3,
        "n_classes": 8,
        "data_file": "bloodmnist_224.npz",
        "task": "multi-class",
        "description": "Blood Cell Microscopy - 8 cell types",
        "class_names": [
            "Basophil", "Eosinophil", "Erythroblast", "Immature Granulocytes",
            "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
        ]
    },
    "tissuemnist": {
        "n_channels": 1,
        "n_classes": 8,
        "data_file": "tissuemnist_224.npz",
        "task": "multi-class",
        "description": "Kidney Cortex Microscopy - 8 tissue types (Grayscale)",
        "class_names": [
            "Collecting Duct", "Distal Convoluted Tubule", "Glomerular Endothelial",
            "Interstitial", "Leukocytes", "Podocytes", "Proximal Tubule", "Thick Ascending Limb"
        ]
    },
    "liver4": {
        "n_channels": 1,
        "n_classes": 4,
        "data_file": "dataset/",
        "task": "multi-class",
        "description": "Liver Staging - 4 classes (F0, F1, F2, F3-F4)",
        "class_names": ["F0", "F1", "F2", "F3-F4"],
        "label_mapping": None,
        "is_liver_dataset": True
    },
    "liver2s": {
        "n_channels": 1,
        "n_classes": 2,
        "data_file": "dataset/",
        "task": "binary",
        "description": "Liver Fibrosis - Significant (F0-F2 vs F3-F4)",
        "class_names": ["No Significant Fibrosis (F0-F2)", "Significant Fibrosis (F3-F4)"],
        "label_mapping": {0: 0, 1: 0, 2: 0, 3: 1},
        "is_liver_dataset": True
    },
    "liver2a": {
        "n_channels": 1,
        "n_classes": 2,
        "data_file": "dataset/",
        "task": "binary",
        "description": "Liver Fibrosis - Any (F0 vs F1-F4)",
        "class_names": ["No Fibrosis (F0)", "Any Fibrosis (F1-F4)"],
        "label_mapping": {0: 0, 1: 1, 2: 1, 3: 1},
        "is_liver_dataset": True
    }
}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    dataset_name = dataset_name.lower()
    if dataset_name not in MEDMNIST_INFO:
        available = ", ".join(MEDMNIST_INFO.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return MEDMNIST_INFO[dataset_name]


class MedMNISTDataset(Dataset):

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
        indices: np.ndarray = None,
        n_channels: int = 3,
        class_names: list = None
    ):
        self.images = images
        self.labels = labels.flatten()
        self.transform = transform
        self.indices = indices if indices is not None else np.arange(len(images))
        self.n_channels = n_channels
        self.class_names = class_names or []

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image = self.images[real_idx]
        label = int(self.labels[real_idx])

        if self.n_channels == 1 and image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if self.n_channels == 1:
            image = Image.fromarray(image.squeeze().astype(np.uint8), mode='L')
        else:
            image = Image.fromarray(image.astype(np.uint8), mode='RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


PathMNISTDataset = MedMNISTDataset


class LiverFibrosisDataset(Dataset):

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
        indices: np.ndarray = None,
        label_mapping: dict = None,
        class_names: list = None
    ):
        self.images = images
        self.original_labels = labels.flatten()
        self.transform = transform
        self.indices = indices if indices is not None else np.arange(len(images))
        self.label_mapping = label_mapping
        self.class_names = class_names or []
        self.n_channels = 1

        if self.label_mapping is not None:
            self.labels = np.array([self.label_mapping[lbl] for lbl in self.original_labels])
        else:
            self.labels = self.original_labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image = self.images[real_idx]
        label = int(self.labels[real_idx])

        image_2d = image.squeeze()
        img_min, img_max = image_2d.min(), image_2d.max()
        if img_max > img_min:
            image_normalized = (image_2d - img_min) / (img_max - img_min)
        else:
            image_normalized = np.zeros_like(image_2d)
        image_uint8 = (image_normalized * 255).astype(np.uint8)

        image_pil = Image.fromarray(image_uint8, mode='L')

        if self.transform:
            image = self.transform(image_pil)
        else:
            image = np.array(image_pil)
            image = np.stack([image] * 3, axis=-1)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


class DataHandler:

    DEFAULT_FT_TRAIN_RATIO = 0.9

    def __init__(
        self,
        dataset_name: str = "pathmnist",
        data_path: str = None,
        ft_train_ratio: float = 0.9,
        random_seed: int = 42,
        log_dir: str = "logs"
    ):
        self.dataset_name = dataset_name.lower()
        self.dataset_info = get_dataset_info(self.dataset_name)
        self.n_classes = self.dataset_info["n_classes"]
        self.n_channels = self.dataset_info["n_channels"]
        self.class_names = self.dataset_info["class_names"]

        if data_path is None:
            data_file = self.dataset_info["data_file"]
            data_path = os.path.expanduser(f"~/.medmnist/{data_file}")

        self.data_path = Path(data_path)
        self.ft_train_ratio = ft_train_ratio
        self.random_seed = random_seed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

        self.ft_train_indices = None
        self.discovery_indices = None

        self.train_indices = None
        self.held_out_indices = None

        self.split_indices_path = self.log_dir / f"{self.dataset_name}_split_indices.pt"

        self.split_info = {}

        print(f"DataHandler initialized for: {self.dataset_name}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Channels: {self.n_channels} ({'Grayscale' if self.n_channels == 1 else 'RGB'})")
        print(f"  Description: {self.dataset_info['description']}")
        
    def load_data(self) -> Dict[str, np.ndarray]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"{self.dataset_name} data not found at {self.data_path}. "
                f"Please download it first: "
                f"https://medmnist.com/"
            )

        print(f"Loading {self.dataset_name} from {self.data_path}...")
        data = np.load(self.data_path)

        self.train_images = data['train_images']
        self.train_labels = data['train_labels']
        self.val_images = data['val_images']
        self.val_labels = data['val_labels']
        self.test_images = data['test_images']
        self.test_labels = data['test_labels']

        print(f"  Official Train: {self.train_images.shape}")
        print(f"  Official Val (FT-Val): {self.val_images.shape}")
        print(f"  Official Test (Test Set): {self.test_images.shape}")

        return {
            'train_images': self.train_images,
            'train_labels': self.train_labels,
            'val_images': self.val_images,
            'val_labels': self.val_labels,
            'test_images': self.test_images,
            'test_labels': self.test_labels
        }

    def _save_split_indices(self) -> str:
        torch.save({
            'dataset_name': self.dataset_name,
            'ft_train_indices': self.ft_train_indices,
            'discovery_indices': self.discovery_indices,
            'ft_train_ratio': self.ft_train_ratio,
            'random_seed': self.random_seed
        }, self.split_indices_path)
        print(f"  Split indices saved to: {self.split_indices_path}")
        return str(self.split_indices_path)

    def _load_split_indices(self) -> bool:
        if not self.split_indices_path.exists():
            return False

        saved = torch.load(self.split_indices_path, weights_only=False)

        if (saved.get('dataset_name') != self.dataset_name or
            saved.get('ft_train_ratio') != self.ft_train_ratio or
            saved.get('random_seed') != self.random_seed):
            print(f"  Warning: Saved split config differs from current config.")
            print(f"    Saved: dataset={saved.get('dataset_name')}, ratio={saved.get('ft_train_ratio')}, seed={saved.get('random_seed')}")
            print(f"    Current: dataset={self.dataset_name}, ratio={self.ft_train_ratio}, seed={self.random_seed}")
            print(f"  Regenerating split...")
            return False

        self.ft_train_indices = saved['ft_train_indices']
        self.discovery_indices = saved['discovery_indices']

        self.train_indices = self.ft_train_indices
        self.held_out_indices = self.discovery_indices

        print(f"  Loaded existing split from: {self.split_indices_path}")
        return True

    def create_resplit(self, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.train_images is None:
            self.load_data()

        if not force and self._load_split_indices():
            n_ft_train = len(self.ft_train_indices)
            n_discovery = len(self.discovery_indices)
            print(f"\n=== Re-split Protocol (Loaded) ===")
            print(f"  FT-Train: {n_ft_train} samples")
            print(f"  Edit-Discovery: {n_discovery} samples")
            return self.ft_train_indices, self.discovery_indices

        n_total = len(self.train_images)
        n_ft_train = int(n_total * self.ft_train_ratio)
        n_discovery = n_total - n_ft_train

        np.random.seed(self.random_seed)

        all_indices = np.arange(n_total)
        np.random.shuffle(all_indices)

        self.ft_train_indices = all_indices[:n_ft_train]
        self.discovery_indices = all_indices[n_ft_train:]

        self.train_indices = self.ft_train_indices
        self.held_out_indices = self.discovery_indices

        assert len(set(self.ft_train_indices) & set(self.discovery_indices)) == 0, \
            "CRITICAL: Overlap detected between FT-Train and Edit-Discovery!"

        print(f"\n=== Re-split Protocol (4-Set Strategy) ===")
        print(f"  Source: Official Training Set ({n_total} samples)")
        print(f"  ├─ FT-Train: {n_ft_train} ({100*self.ft_train_ratio:.0f}%)")
        print(f"  │   Purpose: Fine-tuning + X-Edit covariance (K^T K)")
        print(f"  └─ Edit-Discovery: {n_discovery} ({100*(1-self.ft_train_ratio):.0f}%)")
        print(f"      Purpose: Find unseen errors for editing targets")
        print(f"  Random seed: {self.random_seed}")

        self.split_info = {
            'total_official_train': n_total,
            'ft_train_samples': n_ft_train,
            'discovery_samples': n_discovery,
            'ft_train_ratio': self.ft_train_ratio,
            'random_seed': self.random_seed,
            'val_samples': len(self.val_images),
            'test_samples': len(self.test_images)
        }

        self._save_split_indices()

        return self.ft_train_indices, self.discovery_indices

    def create_held_out_split(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.create_resplit()
    
    def get_class_distribution(self, indices: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
        subset_labels = labels.flatten()[indices]
        unique, counts = np.unique(subset_labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def export_split_info(self) -> str:
        if self.ft_train_indices is None:
            self.create_resplit()

        ft_train_dist = self.get_class_distribution(self.ft_train_indices, self.train_labels)
        discovery_dist = self.get_class_distribution(self.discovery_indices, self.train_labels)
        val_dist = self.get_class_distribution(np.arange(len(self.val_labels)), self.val_labels)
        test_dist = self.get_class_distribution(np.arange(len(self.test_labels)), self.test_labels)

        rows = []

        rows.append({
            'category': 'overall',
            'metric': 'total_samples',
            'ft_train': len(self.ft_train_indices),
            'edit_discovery': len(self.discovery_indices),
            'ft_val': len(self.val_labels),
            'test_set': len(self.test_labels),
            'notes': f'dataset={self.dataset_name}, seed={self.random_seed}, ratio={self.ft_train_ratio}'
        })

        for class_id in range(self.n_classes):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            rows.append({
                'category': 'class_distribution',
                'metric': f'class_{class_id}_{class_name}',
                'ft_train': ft_train_dist.get(class_id, 0),
                'edit_discovery': discovery_dist.get(class_id, 0),
                'ft_val': val_dist.get(class_id, 0),
                'test_set': test_dist.get(class_id, 0),
                'notes': ''
            })

        df = pd.DataFrame(rows)
        csv_path = self.log_dir / f'{self.dataset_name}_data_split_info.csv'
        df.to_csv(csv_path, index=False)

        print(f"\nSplit info exported to: {csv_path}")

        print(f"\n=== 4-Set Protocol Summary ({self.dataset_name}) ===")
        print(f"{'Set':<20} {'Samples':>10} {'Source':<25} {'Purpose':<35}")
        print("-" * 95)
        print(f"{'FT-Train':<20} {len(self.ft_train_indices):>10} {'Official Train (90%)':<25} {'Fine-tuning + X-Edit stats':<35}")
        print(f"{'Edit-Discovery':<20} {len(self.discovery_indices):>10} {'Official Train (10%)':<25} {'Find unseen errors for editing':<35}")
        print(f"{'FT-Val':<20} {len(self.val_labels):>10} {'Official Val (100%)':<25} {'Early stopping only':<35}")
        print(f"{'Test Set':<20} {len(self.test_labels):>10} {'Official Test (100%)':<25} {'Final comparative evaluation':<35}")

        print(f"\n=== Class Distribution ({self.dataset_name}, {self.n_classes} classes) ===")
        print(f"{'Class':<35} {'FT-Train':>10} {'Discovery':>10} {'FT-Val':>10} {'Test':>10}")
        print("-" * 80)
        for class_id in range(self.n_classes):
            class_name = self.class_names[class_id][:30] if class_id < len(self.class_names) else f"Class_{class_id}"
            print(f"{class_id}: {class_name:<32} "
                  f"{ft_train_dist.get(class_id, 0):>10} "
                  f"{discovery_dist.get(class_id, 0):>10} "
                  f"{val_dist.get(class_id, 0):>10} "
                  f"{test_dist.get(class_id, 0):>10}")

        return str(csv_path)
    
    def get_ft_train_dataset(self, transform=None) -> MedMNISTDataset:
        if self.ft_train_indices is None:
            self.create_resplit()

        return MedMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.ft_train_indices,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_discovery_dataset(self, transform=None) -> MedMNISTDataset:
        if self.discovery_indices is None:
            self.create_resplit()

        return MedMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.discovery_indices,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_val_dataset(self, transform=None) -> MedMNISTDataset:
        return MedMNISTDataset(
            images=self.val_images,
            labels=self.val_labels,
            transform=transform,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_test_dataset(self, transform=None) -> MedMNISTDataset:
        return MedMNISTDataset(
            images=self.test_images,
            labels=self.test_labels,
            transform=transform,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_combined_ft_train_with_errors(
        self,
        error_indices: np.ndarray,
        transform=None
    ) -> MedMNISTDataset:
        if self.ft_train_indices is None:
            self.create_resplit()

        discovery_error_original_indices = self.discovery_indices[error_indices]

        combined_indices = np.concatenate([
            self.ft_train_indices,
            discovery_error_original_indices
        ])

        print(f"  Combined dataset: FT-Train ({len(self.ft_train_indices)}) + "
              f"Errors ({len(error_indices)}) = {len(combined_indices)} samples")

        return MedMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=combined_indices,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_error_samples_dataset(
        self,
        error_indices: np.ndarray,
        transform=None
    ) -> MedMNISTDataset:
        if self.discovery_indices is None:
            self.create_resplit()

        error_original_indices = self.discovery_indices[error_indices]

        print(f"  Error samples dataset: {len(error_original_indices)} samples")

        return MedMNISTDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=error_original_indices,
            n_channels=self.n_channels,
            class_names=self.class_names
        )

    def get_train_dataset(self, transform=None) -> MedMNISTDataset:
        return self.get_ft_train_dataset(transform)

    def get_held_out_dataset(self, transform=None) -> MedMNISTDataset:
        return self.get_discovery_dataset(transform)

    def get_original_val_dataset(self, transform=None) -> MedMNISTDataset:
        return self.get_val_dataset(transform)

    def get_original_test_dataset(self, transform=None) -> MedMNISTDataset:
        return self.get_test_dataset(transform)
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        transform=None,
        pin_memory: bool = None
    ) -> Dict[str, DataLoader]:
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        ft_train_dataset = self.get_ft_train_dataset(transform)
        discovery_dataset = self.get_discovery_dataset(transform)
        val_dataset = self.get_val_dataset(transform)
        test_dataset = self.get_test_dataset(transform)

        ft_train_loader = DataLoader(
            ft_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        discovery_loader = DataLoader(
            discovery_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return {
            'ft_train': ft_train_loader,
            'discovery': discovery_loader,
            'val': val_loader,
            'test': test_loader,
            'train': ft_train_loader,
            'held_out': discovery_loader
        }



class LiverFibrosisDataHandler(DataHandler):

    TRAIN_RATIO = 0.60
    VAL_RATIO = 0.20
    TEST_RATIO = 0.20

    def __init__(
        self,
        dataset_name: str = "liver4",
        data_path: str = None,
        ft_train_ratio: float = 0.9,
        random_seed: int = 42,
        log_dir: str = "logs"
    ):
        self.dataset_name = dataset_name.lower()
        self.dataset_info = get_dataset_info(self.dataset_name)
        self.n_classes = self.dataset_info["n_classes"]
        self.n_channels = self.dataset_info["n_channels"]
        self.class_names = self.dataset_info["class_names"]
        self.label_mapping = self.dataset_info.get("label_mapping", None)

        if data_path is None:
            data_path = self.dataset_info["data_file"]

        self.data_path = Path(data_path)
        self.ft_train_ratio = ft_train_ratio
        self.random_seed = random_seed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.all_images = None
        self.all_labels = None
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

        self.train_split_indices = None
        self.val_split_indices = None
        self.test_split_indices = None
        self.ft_train_indices = None
        self.discovery_indices = None

        self.train_indices = None
        self.held_out_indices = None

        self.split_indices_path = self.log_dir / f"{self.dataset_name}_split_indices.pt"

        self.split_info = {}

        print(f"LiverFibrosisDataHandler initialized for: {self.dataset_name}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Channels: {self.n_channels} (Grayscale)")
        print(f"  Description: {self.dataset_info['description']}")
        if self.label_mapping:
            print(f"  Label mapping: {self.label_mapping}")

    def load_data(self) -> Dict[str, np.ndarray]:
        imgs_path = self.data_path / "imgs.npy"
        labs_path = self.data_path / "labs.npy"

        if not imgs_path.exists() or not labs_path.exists():
            raise FileNotFoundError(
                f"Liver fibrosis data not found at {self.data_path}. "
                f"Expected files: imgs.npy, labs.npy"
            )

        print(f"Loading {self.dataset_name} from {self.data_path}...")

        imgs_raw = np.load(imgs_path, allow_pickle=True)
        labs_raw = np.load(labs_path, allow_pickle=True)

        images_list = []
        for img_dict in imgs_raw:
            img_array = list(img_dict.values())[0]
            images_list.append(img_array)

        self.all_images = np.array(images_list, dtype=np.float32)
        self.all_labels = labs_raw.astype(np.int64)

        print(f"  Total samples: {len(self.all_images)}")
        print(f"  Image shape: {self.all_images[0].shape}")
        print(f"  Label distribution (original): {dict(zip(*np.unique(self.all_labels, return_counts=True)))}")

        self._create_initial_splits()

        return {
            'train_images': self.train_images,
            'train_labels': self.train_labels,
            'val_images': self.val_images,
            'val_labels': self.val_labels,
            'test_images': self.test_images,
            'test_labels': self.test_labels
        }

    def _create_initial_splits(self):
        n_total = len(self.all_images)
        all_indices = np.arange(n_total)

        train_idx, val_test_idx = train_test_split(
            all_indices,
            test_size=(self.VAL_RATIO + self.TEST_RATIO),
            random_state=self.random_seed,
            stratify=self.all_labels
        )

        relative_test_size = self.TEST_RATIO / (self.VAL_RATIO + self.TEST_RATIO)
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=relative_test_size,
            random_state=self.random_seed,
            stratify=self.all_labels[val_test_idx]
        )

        self.train_split_indices = train_idx
        self.val_split_indices = val_idx
        self.test_split_indices = test_idx

        self.train_images = self.all_images[train_idx]
        self.train_labels = self.all_labels[train_idx]
        self.val_images = self.all_images[val_idx]
        self.val_labels = self.all_labels[val_idx]
        self.test_images = self.all_images[test_idx]
        self.test_labels = self.all_labels[test_idx]

        print(f"  Created stratified splits:")
        print(f"    Train: {len(train_idx)} ({100*len(train_idx)/n_total:.1f}%)")
        print(f"    Val: {len(val_idx)} ({100*len(val_idx)/n_total:.1f}%)")
        print(f"    Test: {len(test_idx)} ({100*len(test_idx)/n_total:.1f}%)")

    def _save_split_indices(self) -> str:
        torch.save({
            'dataset_name': self.dataset_name,
            'train_split_indices': self.train_split_indices,
            'val_split_indices': self.val_split_indices,
            'test_split_indices': self.test_split_indices,
            'ft_train_indices': self.ft_train_indices,
            'discovery_indices': self.discovery_indices,
            'ft_train_ratio': self.ft_train_ratio,
            'random_seed': self.random_seed
        }, self.split_indices_path)
        print(f"  Split indices saved to: {self.split_indices_path}")
        return str(self.split_indices_path)

    def _load_split_indices(self) -> bool:
        if not self.split_indices_path.exists():
            return False

        saved = torch.load(self.split_indices_path, weights_only=False)

        if (saved.get('dataset_name') != self.dataset_name or
            saved.get('ft_train_ratio') != self.ft_train_ratio or
            saved.get('random_seed') != self.random_seed):
            print(f"  Warning: Saved split config differs. Regenerating...")
            return False

        self.train_split_indices = saved['train_split_indices']
        self.val_split_indices = saved['val_split_indices']
        self.test_split_indices = saved['test_split_indices']
        self.ft_train_indices = saved['ft_train_indices']
        self.discovery_indices = saved['discovery_indices']

        self.train_indices = self.ft_train_indices
        self.held_out_indices = self.discovery_indices

        print(f"  Loaded existing split from: {self.split_indices_path}")
        return True

    def create_resplit(self, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.train_images is None:
            self.load_data()

        if not force and self._load_split_indices():
            self.train_images = self.all_images[self.train_split_indices]
            self.train_labels = self.all_labels[self.train_split_indices]
            self.val_images = self.all_images[self.val_split_indices]
            self.val_labels = self.all_labels[self.val_split_indices]
            self.test_images = self.all_images[self.test_split_indices]
            self.test_labels = self.all_labels[self.test_split_indices]

            n_ft_train = len(self.ft_train_indices)
            n_discovery = len(self.discovery_indices)
            print(f"\n=== Re-split Protocol (Loaded) ===")
            print(f"  FT-Train: {n_ft_train} samples")
            print(f"  Edit-Discovery: {n_discovery} samples")
            return self.ft_train_indices, self.discovery_indices

        n_train = len(self.train_images)
        n_ft_train = int(n_train * self.ft_train_ratio)

        np.random.seed(self.random_seed)
        all_train_indices = np.arange(n_train)
        np.random.shuffle(all_train_indices)

        self.ft_train_indices = all_train_indices[:n_ft_train]
        self.discovery_indices = all_train_indices[n_ft_train:]

        self.train_indices = self.ft_train_indices
        self.held_out_indices = self.discovery_indices

        print(f"\n=== Re-split Protocol (4-Set Strategy) ===")
        print(f"  Source: Train Set ({n_train} samples)")
        print(f"  ├─ FT-Train: {n_ft_train} ({100*self.ft_train_ratio:.0f}%)")
        print(f"  └─ Edit-Discovery: {len(self.discovery_indices)} ({100*(1-self.ft_train_ratio):.0f}%)")

        self.split_info = {
            'total_samples': len(self.all_images),
            'train_samples': n_train,
            'ft_train_samples': n_ft_train,
            'discovery_samples': len(self.discovery_indices),
            'val_samples': len(self.val_images),
            'test_samples': len(self.test_images)
        }

        self._save_split_indices()

        return self.ft_train_indices, self.discovery_indices

    def get_ft_train_dataset(self, transform=None) -> LiverFibrosisDataset:
        if self.ft_train_indices is None:
            self.create_resplit()

        return LiverFibrosisDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.ft_train_indices,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_discovery_dataset(self, transform=None) -> LiverFibrosisDataset:
        if self.discovery_indices is None:
            self.create_resplit()

        return LiverFibrosisDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=self.discovery_indices,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_val_dataset(self, transform=None) -> LiverFibrosisDataset:
        return LiverFibrosisDataset(
            images=self.val_images,
            labels=self.val_labels,
            transform=transform,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_test_dataset(self, transform=None) -> LiverFibrosisDataset:
        return LiverFibrosisDataset(
            images=self.test_images,
            labels=self.test_labels,
            transform=transform,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_combined_ft_train_with_errors(
        self,
        error_indices: np.ndarray,
        transform=None
    ) -> LiverFibrosisDataset:
        if self.ft_train_indices is None:
            self.create_resplit()

        combined_indices = np.concatenate([
            self.ft_train_indices,
            self.discovery_indices[error_indices]
        ])

        print(f"  Combined dataset: FT-Train ({len(self.ft_train_indices)}) + "
              f"Errors ({len(error_indices)}) = {len(combined_indices)} samples")

        return LiverFibrosisDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=combined_indices,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )

    def get_error_samples_dataset(
        self,
        error_indices: np.ndarray,
        transform=None
    ) -> LiverFibrosisDataset:
        if self.discovery_indices is None:
            self.create_resplit()

        error_original_indices = self.discovery_indices[error_indices]
        print(f"  Error samples dataset: {len(error_original_indices)} samples")

        return LiverFibrosisDataset(
            images=self.train_images,
            labels=self.train_labels,
            transform=transform,
            indices=error_original_indices,
            label_mapping=self.label_mapping,
            class_names=self.class_names
        )


def get_data_handler(dataset_name: str, **kwargs) -> Union[DataHandler, LiverFibrosisDataHandler]:
    dataset_name = dataset_name.lower()
    dataset_info = get_dataset_info(dataset_name)

    if dataset_info.get("is_liver_dataset", False):
        return LiverFibrosisDataHandler(dataset_name=dataset_name, **kwargs)
    else:
        return DataHandler(dataset_name=dataset_name, **kwargs)


def main():
    print("=" * 70)
    print("ViT Model Editing Pipeline - Data Handler (Multi-Dataset Support)")
    print("=" * 70)

    dataset_name = "pathmnist"
    print(f"\n>>> Testing with {dataset_name} <<<")

    handler = DataHandler(
        dataset_name=dataset_name,
        ft_train_ratio=0.9,
        random_seed=42,
        log_dir="logs"
    )

    handler.load_data()
    handler.create_resplit()

    handler.export_split_info()

    ft_train_ds = handler.get_ft_train_dataset()
    discovery_ds = handler.get_discovery_dataset()
    val_ds = handler.get_val_dataset()
    test_ds = handler.get_test_dataset()

    print(f"\n=== Dataset Verification ===")
    print(f"FT-Train dataset size: {len(ft_train_ds)}")
    print(f"Edit-Discovery dataset size: {len(discovery_ds)}")
    print(f"FT-Val dataset size: {len(val_ds)}")
    print(f"Test Set dataset size: {len(test_ds)}")

    img, label = ft_train_ds[0]
    print(f"\nSample image shape: {img.shape}")
    class_name = handler.class_names[label] if label < len(handler.class_names) else f"Class_{label}"
    print(f"Sample label: {label} ({class_name})")

    loaders = handler.get_dataloaders(batch_size=32)
    print(f"\n=== DataLoader Keys ===")
    for key in loaders.keys():
        print(f"  '{key}': {len(loaders[key])} batches")

    print(f"\n=== Dataset Metadata ===")
    print(f"  Dataset: {handler.dataset_name}")
    print(f"  Classes: {handler.n_classes}")
    print(f"  Channels: {handler.n_channels}")

    print("\n[OK] Data handler with multi-dataset support initialized successfully!")


if __name__ == "__main__":
    main()
