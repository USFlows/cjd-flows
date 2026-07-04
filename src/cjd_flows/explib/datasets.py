import os
import typing as T
from abc import abstractmethod

import idx2numpy
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.datasets import make_blobs, make_checkerboard, make_circles, make_moons
from torch import Tensor
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from PIL import Image


# Base dataset classes
class DequantizedDataset(torch.utils.data.Dataset):
    """
    A dataset that dequantizes the data by adding uniform noise to each pixel.
    """

    def __init__(
        self,
        dataset: T.Union[os.PathLike, torch.utils.data.Dataset, np.ndarray],
        num_bits: int = 8,
        space_to_depth_factor: int = 1,
        device: torch.device = None, 
    ):
        super().__init__()
        if isinstance(dataset, torch.utils.data.Dataset) or isinstance(
            dataset, np.ndarray
        ) or isinstance(dataset, torch.Tensor):
            self.dataset = dataset
        else:
            self.dataset = pd.read_csv(dataset).values

        if not isinstance(self.dataset, torch.Tensor):
            self.dataset = Tensor(self.dataset)
        
        if len(self.dataset.shape) == 3:
            self.dataset = self.dataset.unsqueeze(1)
            
        if space_to_depth_factor > 1:
            n, c, h, w = self.dataset.shape
            f = space_to_depth_factor
            self.dataset = (
                self.dataset.reshape(n, c, h//f, f, w//f, f)  # Split spatial dims into (n, k) blocks
                .permute(0, 1, 3, 5, 2, 4)                # Reorder axes to (c, n, n, k, k)
                .reshape(n, c * f * f, h//f, w//f)               # Combine channels and blocks into (k²c, n, n)
            )

        self.dataset = self.dataset.to(device) 
        self.num_bits = num_bits
        self.num_levels = 2**num_bits
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / self.num_levels),
                transforms.Lambda(lambda x: x + torch.rand_like(x) / self.num_levels),
            ]
        )
        

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        x = Tensor(self.transform(x))
        return x, y

    def __len__(self):
        return len(self.dataset)


class DataSplit:
    def __init__(*agrs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_train(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_test(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_val(self) -> torch.utils.data.Dataset:
        raise NotImplementedError


class SimpleSplit(DataSplit):
    """
    Split of dataset
    """

    def __init__(
        self,
        train: torch.utils.data.Dataset,
        test: torch.utils.data.Dataset,
        val: torch.utils.data.Dataset,
    ):
        """Create split of dataset

        Args:
            train (torch.utils.data.Dataset): training set
            test (torch.utils.data.Dataset): test set
            val (torch.utils.data.Dataset): validation set
        """
        self.train = train
        self.test = test
        self.val = val

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val

def make_transformed_laplace(dim: int, n_samples: int, transform: Tensor = None) -> Tensor:
    """Create uniform dataset

    Args:
        dim (int): dimensionality of dataset
        num_samples (int): number of samples
        transform (Tensor, optional): transformation matrix. Defaults to None which applies no transformation.

    Returns:
        np.ndarray: dataset
    """
    sample = torch.distributions.Laplace(torch.zeros(dim), torch.ones(dim)).sample([n_samples])
    
    if transform is not None:
        inconsistent = len(transform.shape) != 2
        inconsistent = inconsistent or (transform.shape[0] != transform.shape[1])
        inconsistent = inconsistent or (transform.shape[0] != dim)
        if inconsistent:
            raise ValueError(f"transform must be {dim}X{dim} got {transform.shape}.")

        sample = (transform @ sample.T).T
    
    return sample  
    
    

class FlattenedDataset(torch.utils.data.Dataset):
    """
    A dataset that flattens the data.
    """

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        x = x.flatten()
        return x, y

    def __len__(self):
        return len(self.dataset)
    
class DataSplitFromCSV(DataSplit):
    def __init__(self, train: os.PathLike, test: os.PathLike, val: os.PathLike):
        self.train = train
        self.test = test
        self.val = val

    def get_train(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.train).values

    def get_test(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.test).values

    def get_val(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.val).values


# Synthetic datasets


GENERATORS = {
    "moons": make_moons,
    "blobs": make_blobs,
    "checkerboard": make_checkerboard,
    "circles": make_circles,
    "transformed": make_transformed_laplace,
}

class SyntheticDataset(torch.utils.data.Dataset):
    """
    Dataset from generator function
    """

    def __init__(
        self,
        generator: T.Union[T.Callable[..., np.ndarray], str],
        params: T.Dict[str, T.Any],
        *args,
        **kwargs
    ):
        """Create dataset from generator function

        Args:
            generator (function): generator function
            params: [dict]: parameters for generator function
        """
        super().__init__(*args, **kwargs)
        if isinstance(generator, str):
            generator = GENERATORS[generator]   

        self.dataset = generator(**params)[0]

    def __getitem__(self, index: int):
        if isinstance(self.dataset, np.ndarray):
            x = self.dataset[index].copy()
        else:
            x = self.dataset[index]
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x, torch.zeros_like(x)

    def __len__(self):
        return len(self.dataset)


class SyntheticSplit(SimpleSplit):
    """
    Split of synthetic dataset
    """

    def __init__(
        self,
        generator: T.Union[T.Callable[..., np.ndarray], str],
        params_train: T.Dict[str, T.Any],
        params_test: T.Dict[str, T.Any],
        params_val: T.Dict[str, T.Any],
        *args,
        **kwargs
    ):
        """Create dataset from generator function

        Args:
            generator (function): generator function
            params: ]dict]: parameters for generator function
        """
        if isinstance(generator, str):
            generator = GENERATORS[generator]

        train = SyntheticDataset(generator, params_train)
        test = SyntheticDataset(generator, params_test)
        val = SyntheticDataset(generator, params_val)
        super().__init__(train=train, test=test, val=val, *args, **kwargs)


# FashonMNIST
class FashionMnistDequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        label: T.Optional[int] = None,
        scale: bool = False,
        three_channel: bool = False,  # NEW
        *args,
        **kwargs,
    ):
        rel_path = (
            "FashionMNIST/raw/train-images-idx3-ubyte"
            if train
            else "FashionMNIST/raw/t10k-images-idx3-ubyte"
        )
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            FashionMNIST(dataloc, train=train, download=True)
        # TODO: remove hardcoding of 3x3 downsampling, vectorizing
        dataset = idx2numpy.convert_from_file(path)
        if scale:
            dataset = dataset[:, ::3, ::3]
        # If requested, convert grayscale -> 3-channel by repeating channel
        if three_channel:
            # dataset shape: (N, H, W) -> (N, 1, H, W) then repeat to 3 channels
            dataset = np.repeat(dataset[:, None, :, :], 3, axis=1)
        #dataset = dataset.reshape(dataset.shape[0], -1)
        if label is not None:
            rel_path = (
                "FashionMNIST/raw/train-labels-idx1-ubyte"
                if train
                else "FashionMNIST/raw/t10k-labels-idx1-ubyte"
            )
            path = os.path.join(dataloc, rel_path)
            labels = idx2numpy.convert_from_file(path)
            dataset = dataset[labels == label]
        super().__init__(dataset, num_bits=8, *args, **kwargs)

    def __getitem__(self, index: int):
        if not isinstance(self.dataset, torch.Tensor):
            x = Tensor(self.dataset[index].copy())
        else:
            x = self.dataset[index]
        x = self.transform(x)
        return x, 0


class FashionMnistSplit(DataSplit):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        val_split: float = 0.1,
        label: T.Optional[int] = None,
        space_to_depth_factor: int = 1,
        three_channel: bool = False,  # NEW
    ):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        self.train = FashionMnistDequantized(
            self.dataloc,
            train=True,
            label=label,
            space_to_depth_factor=space_to_depth_factor,
            three_channel=three_channel,  # PASS THROUGH
        )
        shuffle = torch.randperm(len(self.train))
        self.val = torch.utils.data.Subset(
            self.train, shuffle[: int(len(self.train) * val_split)]
        )
        self.train = torch.utils.data.Subset(
            self.train, shuffle[int(len(self.train) * val_split) :]
        )
        self.test = FashionMnistDequantized(self.dataloc, train=False, label=label, space_to_depth_factor=space_to_depth_factor, three_channel=three_channel)

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val


# MNIST
class MnistDequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        digit: T.Optional[int] = None,
        flatten=False,
        scale: bool = False,
        device: torch.device = None,
        space_to_depth_factor: int = 1,
        three_channel: bool = False,  # NEW
    ):
        if train:
            rel_path = "MNIST/raw/train-images-idx3-ubyte"
        else:
            rel_path = "MNIST/raw/t10k-images-idx3-ubyte"
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            MNIST(dataloc, train=train, download=True)

        dataset = idx2numpy.convert_from_file(path)
        if scale:
            dataset = dataset[:, ::3, ::3]
        if flatten:
            dataset = dataset.reshape(dataset.shape[0], -1)
        # Convert to 3-channel if requested
        if three_channel and dataset.ndim == 3:
            dataset = np.repeat(dataset[:, None, :, :], 3, axis=1)
        if digit is not None:
            if train:
                rel_path = "MNIST/raw/train-labels-idx1-ubyte"
            else:
                rel_path = "MNIST/raw/t10k-labels-idx1-ubyte"
            path = os.path.join(dataloc, rel_path)
            labels = idx2numpy.convert_from_file(path)
            dataset = dataset[labels == digit]
        super().__init__(
            torch.Tensor(dataset),
            num_bits=8,
            device=device,
            space_to_depth_factor=space_to_depth_factor
        )

    def __getitem__(self, index: int):
        if not isinstance(self.dataset, torch.Tensor):
            x = Tensor(self.dataset[index].copy())
        else:
            x = self.dataset[index]
        x = self.transform(x)

        return x, 0

class MnistSplit(DataSplit):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        val_split: float = 0.1,
        digit: T.Optional[int] = None,
        scale: bool = False,
        device: torch.device = None,
        space_to_depth_factor: int = 1,
        three_channel: bool = False,  # NEW
    ):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        self.train = MnistDequantized(
            self.dataloc,
            train=True,
            digit=digit,
            scale=scale,
            space_to_depth_factor=space_to_depth_factor,
            device=device,
            three_channel=three_channel,  # PASS THROUGH
        )
        shuffle = torch.randperm(len(self.train))
        self.val = torch.utils.data.Subset(
            self.train, shuffle[: int(len(self.train) * val_split)]
        )
        self.train = torch.utils.data.Subset(
            self.train, shuffle[int(len(self.train) * val_split) :]
        )
        self.test = MnistDequantized(
            self.dataloc,
            train=False,
            digit=digit,
            scale=scale,
            space_to_depth_factor=space_to_depth_factor,
            device=device,
            three_channel=three_channel,  # PASS THROUGH
        )

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val

# CFAIR-10

class Cifar10Dequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        label: T.Optional[int] = None,
        space_to_depth_factor: int = 1,
        device: torch.device = None,
        *args,
        **kwargs,
    ):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        
        # Transform to convert PIL image to tensor in [0,1]
        transform_to_tensor = transforms.ToTensor()
        full_dataset = CIFAR10(root=dataloc, train=train, download=True, transform=transform_to_tensor)
        
        # Collect all images and labels
        data = []
        labels = []
        for img, lbl in full_dataset:
            data.append(img)
            labels.append(lbl)
        data = torch.stack(data, dim=0)  # Shape: (N, 3, 32, 32)
        # Convert from [0,1] float to [0,255] uint8
        data = (data * 255).to(torch.uint8)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Filter by label if specified
        if label is not None:
            mask = (labels == label)
            data = data[mask]
        
        super().__init__(
            data,
            num_bits=8,
            space_to_depth_factor=space_to_depth_factor,
            device=device,
            *args,
            **kwargs
        )

    def __getitem__(self, index: int):
        if not isinstance(self.dataset, torch.Tensor):
            x = torch.tensor(self.dataset[index].copy())
        else:
            x = self.dataset[index]
        x = self.transform(x)
        return x, 0  # Return dummy label 0


class Cifar10Split(DataSplit):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        val_split: float = 0.1,
        label: T.Optional[int] = None,
        space_to_depth_factor: int = 1,
        device: torch.device = None,
    ):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        
        # Create training dataset
        self.train = Cifar10Dequantized(
            self.dataloc,
            train=True,
            label=label,
            space_to_depth_factor=space_to_depth_factor,
            device=device
        )
        
        # Split training data into train and validation
        shuffle = torch.randperm(len(self.train))
        val_size = int(len(self.train) * val_split)
        self.val = torch.utils.data.Subset(self.train, shuffle[:val_size])
        self.train = torch.utils.data.Subset(self.train, shuffle[val_size:])
        
        # Create test dataset
        self.test = Cifar10Dequantized(
            self.dataloc,
            train=False,
            label=label,
            space_to_depth_factor=space_to_depth_factor,
            device=device
        )

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val
            

class DistributionDataset(torch.utils.data.Dataset):
    """
    Dataset that generates samples from a given distribution.
    """
    
    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        num_samples: int,
        device: torch.device = None,
    ):
        """
        Initialize dataset with a distribution and number of samples.
        
        Args:
            distribution: Distribution to sample from
            num_samples: Number of samples to generate
            device: Device to store samples on
        """
        super().__init__()
        self.distribution = distribution
        self.num_samples = num_samples
        self.device = device
        self.data = self.distribution.sample((num_samples,)).to(device)
        
        # Dummy labels for compatibility
        self.labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    
    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return self.num_samples
    
    def to(self, device: torch.device):
        """
        Move dataset to a different device.
        
        Args:
            device: Device to move samples to
        """
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self



class DistributionSplit(SimpleSplit):
    """
    Data split that generates train/val/test from a distribution.
    """
    
    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        num_train: int,
        num_val: int,
        num_test: int,
        device: torch.device = None,
    ):
        """
        Create train/val/test splits from a distribution.
        
        Args:
            distribution: Distribution to sample from
            num_train: Number of training samples
            num_val: Number of validation samples
            num_test: Number of test samples
            device: Device to store samples on
        """
        train = DistributionDataset(distribution, num_train, device)
        val = DistributionDataset(distribution, num_val, device)
        test = DistributionDataset(distribution, num_test, device)
        super().__init__(train, test, val)

# MVTec AD Dataset
class MVTecADDequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        category: str = "bottle",
        is_anomaly: bool = False,
        device: torch.device = None,
        space_to_depth_factor: int = 1,
        download: bool = True,
        load_into_memory: bool = False,
        *args,
        **kwargs,
    ):
        """
        MVTec AD dataset for anomaly detection.
        
        Args:
            dataloc: Path to the dataset directory
            train: Whether to load training or test data
            category: Category/class of objects (e.g., 'bottle', 'cable', 'capsule')
            is_anomaly: Whether to load anomalous samples (only applicable for test set)
            device: Device to store data on
            space_to_depth_factor: Factor for space-to-depth transformation
            download: Whether to download the dataset if not found
        """
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data", "mvtec_ad")

        # Create directory if it doesn't exist
        os.makedirs(dataloc, exist_ok=True)

        # Check if dataset exists, download if needed
        category_path = os.path.join(dataloc, category)
        if not os.path.exists(category_path) and download:
            self._download_mvtec_ad(dataloc, category)

        # Define paths
        split = "train" if train else "test"
        base_path = os.path.join(dataloc, category, split)

        if not os.path.exists(base_path):
            raise RuntimeError(f"MVTec AD dataset not found at {base_path}. "
                              "Set download=True to download it automatically.")

        # Gather image paths depending on mode
        if train or not is_anomaly:
            # For training or normal test samples
            img_dir = os.path.join(base_path, "good")
            if not os.path.exists(img_dir):
                raise RuntimeError(f"Good images directory not found at {img_dir}. "
                                   "The dataset structure might be incorrect.")
            img_paths = [os.path.join(img_dir, f)
                         for f in sorted(os.listdir(img_dir))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            labels = [0] * len(img_paths)
        else:
            # For anomalous test samples, handle multiple anomaly types
            anomaly_dirs = [d for d in sorted(os.listdir(base_path))
                            if os.path.isdir(os.path.join(base_path, d)) and d != "good"]
            if not anomaly_dirs:
                raise RuntimeError(f"No anomaly directories found in {base_path}.")
            img_paths = []
            labels = []
            for anomaly_type in anomaly_dirs:
                anomaly_dir = os.path.join(base_path, anomaly_type)
                files = [os.path.join(anomaly_dir, f)
                         for f in sorted(os.listdir(anomaly_dir))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                img_paths.extend(files)
                labels.extend([1] * len(files))

        # Ensure we found images
        if not img_paths:
            raise RuntimeError(f"No images found in {base_path}.")

        # If requested, load everything into memory (backward compatible)
        transform_to_tensor = transforms.ToTensor()
        self.device = device
        self.space_to_depth_factor = space_to_depth_factor
        self.num_bits = 8
        self.num_levels = 2 ** self.num_bits
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / self.num_levels),
                transforms.Lambda(lambda x: x + torch.rand_like(x) / self.num_levels),
            ]
        )

        self._load_into_memory = bool(load_into_memory)

        if self._load_into_memory:
            images = []
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform_to_tensor(img) * 255.0
                images.append(img_tensor)
            dataset = torch.stack(images)
            # keep dtype float for safe operations; DequantizedDataset handled uint8 but transform works on float
            super().__init__(dataset, num_bits=self.num_bits, space_to_depth_factor=space_to_depth_factor, device=device, *args, **kwargs)
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            # Store paths and labels and avoid loading images until requested
            self.img_paths = img_paths
            self.labels = torch.tensor(labels, dtype=torch.long)
            # don't call parent constructor to avoid converting full dataset to tensor

    def _download_mvtec_ad(self, dataloc: os.PathLike, category: str):
        """
        Download the MVTec AD dataset if it doesn't exist, handling XZ compression.
        """
        import requests
        import tarfile
        import lzma
        from tqdm import tqdm
        import shutil
        
        # Official MVTec AD download URL (this might need to be updated)
        base_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094"
        url = f"{base_url}/{category}.xz"  # Note the .xz extension
        
        # Download to a temporary file
        temp_path = os.path.join(dataloc, f"{category}.tar.xz")
        
        print(f"Downloading MVTec AD {category} dataset (XZ compressed)...")
        
        try:
            # Stream the download with timeout and headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()  # Check for HTTP errors
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(temp_path, 'wb') as f, tqdm(
                desc=f"Downloading {category}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)
            
            # Extract the XZ-compressed tar archive
            print(f"Extracting {category} dataset...")
            
            # Method 1: Using tarfile with xz compression (Python 3.3+)
            try:
                with tarfile.open(temp_path, 'r:xz') as tar:
                    tar.extractall(dataloc)
            except Exception as e:
                print(f"Tar extraction failed: {e}, trying manual xz extraction...")
                
                # Method 2: Manual extraction as fallback
                # First decompress xz, then extract tar
                decompressed_path = temp_path.replace('.xz', '')
                
                # Decompress xz
                with lzma.open(temp_path) as compressed:
                    with open(decompressed_path, 'wb') as decompressed:
                        shutil.copyfileobj(compressed, decompressed)
                
                # Extract tar
                with tarfile.open(decompressed_path, 'r:') as tar:
                    tar.extractall(dataloc)
                
                # Clean up intermediate file
                os.remove(decompressed_path)
            
            print(f"Download and extraction of {category} complete.")
            
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            raise RuntimeError(f"Failed to download MVTec AD dataset. Please download manually from https://www.mvtec.com/company/research/datasets/mvtec-ad")
        except Exception as e:
            print(f"Failed to process MVTec AD dataset: {e}")
            raise
        finally:
            # Clean up the temporary file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def __getitem__(self, index: int):
        # Support both loaded-into-memory and on-demand modes
        if getattr(self, "_load_into_memory", False):
            x = self.dataset[index]
        else:
            # Load image on demand
            img_path = self.img_paths[index]
            img = Image.open(img_path).convert('RGB')
            img_tensor = transforms.ToTensor()(img) * 255.0
            x = img_tensor

        # Apply space-to-depth if needed
        if getattr(self, "space_to_depth_factor", 1) > 1:
            x = self._apply_space_to_depth(x)

        # Move to device if set
        if getattr(self, "device", None) is not None:
            x = x.to(self.device)

        # Apply dequantization transform (returns float)
        x = self.transform(x)
        y = self.labels[index]
        return x, y

    def _apply_space_to_depth(self, x: Tensor) -> Tensor:
        """Apply space-to-depth on a single image tensor of shape (C, H, W)."""
        f = int(self.space_to_depth_factor)
        if f <= 1:
            return x
        # Ensure dimensions are compatible
        c, h, w = x.shape
        if h % f != 0 or w % f != 0:
            raise RuntimeError(f"Image size ({h},{w}) not divisible by space_to_depth_factor {f}")
        # Reshape and permute: (C, H, W) -> (C, H//f, f, W//f, f) -> (C, f, f, H//f, W//f) -> (C*f*f, H//f, W//f)
        x = x.reshape(c, h // f, f, w // f, f).permute(0, 2, 4, 1, 3).reshape(c * f * f, h // f, w // f)
        return x

    def __len__(self):
        if getattr(self, "_load_into_memory", False):
            return len(self.dataset)
        return len(self.img_paths)

class MVTecADSplit(DataSplit):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        category: str = "bottle",
        val_split: float = 0.1,
        space_to_depth_factor: int = 1,
        device: torch.device = None,
        download: bool = True,
        load_into_memory: bool = False,
    ):
        """
        Data split for MVTec AD dataset.
        
        Args:
            dataloc: Path to the dataset directory
            category: Category/class of objects
            val_split: Fraction of training data to use for validation
            space_to_depth_factor: Factor for space-to-depth transformation
            device: Device to store data on
            download: Whether to download the dataset if not found
        """
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data", "mvtec_ad")
            
        # Training data (only normal samples)
        self.train = MVTecADDequantized(
            dataloc=dataloc,
            train=True,
            category=category,
            is_anomaly=False,
            space_to_depth_factor=space_to_depth_factor,
            device=device,
            download=download,
            load_into_memory=load_into_memory
        )
        
        # Split training data into train and validation
        shuffle = torch.randperm(len(self.train))
        val_size = int(len(self.train) * val_split)
        self.val = torch.utils.data.Subset(self.train, shuffle[:val_size])
        self.train = torch.utils.data.Subset(self.train, shuffle[val_size:])
        
        # Test data (both normal and anomalous samples)
        test_normal = MVTecADDequantized(
            dataloc=dataloc,
            train=False,
            category=category,
            is_anomaly=False,
            space_to_depth_factor=space_to_depth_factor,
            device=device,
            download=download
        )
        
        test_anomaly = MVTecADDequantized(
            dataloc=dataloc,
            train=False,
            category=category,
            is_anomaly=True,
            space_to_depth_factor=space_to_depth_factor,
            device=device,
            download=download
        )
        
        # Combine normal and anomalous test samples
        self.test = torch.utils.data.ConcatDataset([test_normal, test_anomaly])

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val