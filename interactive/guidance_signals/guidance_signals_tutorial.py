#!/usr/bin/env python3
"""
guidance_signal.py - Generate 3D guidance signals from scribble coordinates.

This module provides classes for generating different types of guidance signals:
- GaussianHeatmap: Standard Gaussian heatmaps
- EuclideanDistance: Euclidean distance transform
- GeodesicDistance: Geodesic distance transform (2D per slice)
- AdaptiveHeatmap: Adaptive heatmaps with per-point sigma based on geodesic statistics
- DiskHeatmap: Flat disk (circle) heatmaps with constant radius
- CombinedSignal: Combines multiple signals

The output is saved as a NIfTI file (.nii.gz).

Example usage (CLI):
    python guidance_signal.py --json clicks.json --ref reference.nii.gz --output guidance.nii.gz --signal combined --debug

Example usage (import):
    from guidance_signal import GuidanceSignalGenerator
    generator = GuidanceSignalGenerator(ref_path='reference.nii.gz')
    signal = generator.generate('gaussian', points=tumor_points, sigma=2.0)
"""

import argparse
import json
import os
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

# -----------------------------------------------------------------------------
# Library imports with fallbacks
# -----------------------------------------------------------------------------

# GPU-accelerated distance transform (optional)
try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as gpu_distance_transform
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Geodesic distance transform (optional)
try:
    import GeodisTK
    GEODIS_AVAILABLE = True
    GEODIS_2D_RASTER = hasattr(GeodisTK, 'geodesic2d_raster_scan')
    GEODIS_2D_FM = hasattr(GeodisTK, 'geodesic2d_fast_marching')
    if not GEODIS_2D_RASTER and not GEODIS_2D_FM:
        GEODIS_AVAILABLE = False
        warnings.warn("GeodisTK installed but no 2D geodesic function found.")
except ImportError:
    GEODIS_AVAILABLE = False
    GEODIS_2D_RASTER = False
    GEODIS_2D_FM = False


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def parse_json_coordinates(json_path: str) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Parse JSON file and extract tumor and background point coordinates.

    Args:
        json_path: Path to JSON file with scribble coordinates

    Returns:
        tuple: (tumor_points, background_points) as lists of [x, y, z] coordinates
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    tumor_points = []
    background_points = []

    for point_data in data.get('points', []):
        point = point_data.get('point', [])
        name = point_data.get('name', '')
        if name.lower() == 'tumor':
            tumor_points.append(point)
        elif name.lower() == 'background':
            background_points.append(point)

    return tumor_points, background_points


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to [0, 1] range."""
    if signal.max() > signal.min():
        signal = (signal - signal.min()) / (signal.max() - signal.min())
    return signal


def save_debug_slice(guidance_signal: np.ndarray, points: list, output_path: str,
                      signal_name: str, slice_dim: int = 2) -> None:
    """Save a PNG image of the axial slice with the most points."""
    try:
        slice_counts = {}
        for point in points:
            if len(point) > slice_dim:
                z = point[slice_dim]
                slice_counts[z] = slice_counts.get(z, 0) + 1

        if not slice_counts:
            print("No points found for debug visualization.")
            return

        max_slice = max(slice_counts, key=slice_counts.get)
        slice_data = guidance_signal[:, :, max_slice]

        plt.figure(figsize=(10, 8))
        plt.imshow(slice_data, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Guidance Signal Value')
        plt.title(f"{signal_name} - Slice Z={max_slice}\n"
                  f"Points: {slice_counts[max_slice]}\n"
                  f"min={slice_data.min():.3f}, max={slice_data.max():.3f}")
        plt.xlabel('Column')
        plt.ylabel('Row')

        debug_path = output_path.replace('.nii.gz', f'_{signal_name.lower()}_debug.png')
        if not debug_path.endswith('.png'):
            debug_path = output_path + f'_{signal_name.lower()}_debug.png'

        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Debug slice saved to: {debug_path}")

    except ImportError:
        print("matplotlib not available. Cannot save debug PNG.")
    except Exception as e:
        print(f"Failed to save debug PNG: {e}")


def save_reference_slice(reference_volume: np.ndarray, points: list, output_path: str,
                          slice_dim: int = 2) -> None:
    """
    Save a PNG image of the reference axial slice with points overlaid.

    Args:
        reference_volume: 3D reference volume
        points: List of [row, col, slice] coordinates
        output_path: Base output path for the PNG
        slice_dim: Dimension to slice (default: 2 for axial)
    """
    try:
        # Find the slice with the most points
        slice_counts = {}
        for point in points:
            if len(point) > slice_dim:
                z = point[slice_dim]
                slice_counts[z] = slice_counts.get(z, 0) + 1

        if not slice_counts:
            print("No points found for reference slice visualization.")
            return

        max_slice = max(slice_counts, key=slice_counts.get)

        # Extract the slice from reference volume
        slice_data = reference_volume[:, :, max_slice]

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Reference slice without points
        im0 = axes[0].imshow(slice_data, origin='lower', cmap='gray')
        axes[0].set_title(f'Reference Slice Z={max_slice} (no points)')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        plt.colorbar(im0, ax=axes[0])

        # Plot 2: Reference slice with points overlaid
        im1 = axes[1].imshow(slice_data, origin='lower', cmap='gray')

        # Get points in this slice
        points_in_slice = [p for p in points if int(p[2]) == max_slice]
        if points_in_slice:
            rows = [p[0] for p in points_in_slice]
            cols = [p[1] for p in points_in_slice]
            axes[1].scatter(cols, rows, c='red', s=5, marker='o',
                            alpha=0.8, label=f'Tumor points ({len(points_in_slice)})',
                            edgecolors='white', linewidth=1)
            axes[1].legend()

        axes[1].set_title(f'Reference Slice Z={max_slice} (with {len(points_in_slice)} points)')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')
        plt.colorbar(im1, ax=axes[1])

        plt.suptitle(f'Reference Slice with Scribble Points', fontsize=14)

        # Save the figure
        debug_path = output_path.replace('.nii.gz', '_reference_debug.png')
        if not debug_path.endswith('.png'):
            debug_path = output_path + '_reference_debug.png'

        plt.tight_layout()
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Reference slice debug saved to: {debug_path}")

    except ImportError:
        print("matplotlib not available. Cannot save reference debug PNG.")
    except Exception as e:
        print(f"Failed to save reference debug PNG: {e}")


def _save_slice_debug(slice_idx: int, reference_slice: np.ndarray, seed_mask: np.ndarray,
                       geodesic_result: np.ndarray, output_dir: str, method: str) -> None:
    """Save debug images for a single slice."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(reference_slice, cmap='gray', origin='lower')
    axes[0].set_title(f'Reference Slice {slice_idx}')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(seed_mask, cmap='hot', origin='lower', interpolation='nearest')
    axes[1].set_title(f'Seed Mask (seeds={np.sum(seed_mask)})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(geodesic_result, cmap='hot', origin='lower', interpolation='nearest')
    axes[2].set_title(f'Geodesic Distance\nmin={geodesic_result.min():.2f}, max={geodesic_result.max():.2f}')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f'Slice {slice_idx} - Geodesic Distance (inverted)', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'slice_{slice_idx:04d}_raw_debug.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Raw geodesic debug saved: {save_path}")


# -----------------------------------------------------------------------------
# Base Signal Class
# -----------------------------------------------------------------------------

class BaseSignal(ABC):
    """Abstract base class for all guidance signals."""
    
    def __init__(self, shape: Tuple[int, int, int], points: List[List[int]], 
                 use_gpu: bool = True, verbose: bool = True):
        """
        Initialize the signal generator.
        
        Args:
            shape: 3D volume shape (rows, cols, slices)
            points: List of [row, col, slice] coordinates
            use_gpu: Whether to attempt GPU acceleration
            verbose: Whether to print progress messages
        """
        self.shape = shape
        self.points = points
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.verbose = verbose
    
    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate the guidance signal."""
        pass
    
    def _log(self, message: str) -> None:
        """Print log message if verbose."""
        if self.verbose:
            print(message)


# -----------------------------------------------------------------------------
# Disk Utility Class
# -----------------------------------------------------------------------------

class Disk:
    """Utility class for creating and placing 2D disks in a 3D volume."""
    
    @staticmethod
    def create_disk(radius: float, normalize: bool = True) -> np.ndarray:
        """
        Create a 2D disk (circle) kernel.
        
        Args:
            radius: Radius of the disk
            normalize: Whether to normalize the kernel so sum equals 1
            
        Returns:
            2D numpy array with the disk kernel
        """
        if radius <= 0:
            return np.array([[1.0]])
        
        r = int(np.ceil(radius))
        kernel_size = 2 * r + 1
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = x*x + y*y <= radius*radius
        
        kernel = mask.astype(np.float32)
        if normalize and np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
        
        return kernel
    
    @staticmethod
    def place_disks(volume: np.ndarray, coordinates: List[Tuple[int, int, int]], 
                     radius: float, normalize: bool = True) -> None:
        """
        Place disks at multiple coordinates in a volume.
        
        Args:
            volume: 3D volume to place disks into (modified in-place)
            coordinates: List of (row, col, slice) coordinates
            radius: Radius of the disk
            normalize: Whether to normalize the disk kernel
        """
        if radius <= 0:
            for coord in coordinates:
                row, col, slice_idx = coord
                if (0 <= row < volume.shape[0] and 
                    0 <= col < volume.shape[1] and 
                    0 <= slice_idx < volume.shape[2]):
                    volume[row, col, slice_idx] += 1.0
            return
        
        # Create the disk kernel
        disk = Disk.create_disk(radius, normalize)
        r = int(np.ceil(radius))
        kernel_size = 2 * r + 1
        
        # Place disk at each coordinate
        for coord in coordinates:
            row, col, slice_idx = coord
            
            # Define disk boundaries
            row_start = max(0, row - r)
            row_end = min(volume.shape[0], row + r + 1)
            col_start = max(0, col - r)
            col_end = min(volume.shape[1], col + r + 1)
            
            # Define kernel boundaries (offset if at image edge)
            kr_start = max(0, r - row)
            kr_end = kernel_size - max(0, (row + r + 1) - volume.shape[0])
            kc_start = max(0, r - col)
            kc_end = kernel_size - max(0, (col + r + 1) - volume.shape[1])
            
            # Place disk in volume
            volume[row_start:row_end, col_start:col_end, slice_idx] += \
                disk[kr_start:kr_end, kc_start:kc_end]


# -----------------------------------------------------------------------------
# Signal Classes
# -----------------------------------------------------------------------------

class GaussianHeatmap(BaseSignal):
    """Gaussian heatmap signal generator."""
    
    def __init__(self, shape: Tuple[int, int, int], points: List[List[int]], 
                 sigma: float = 2.0, use_gpu: bool = True, verbose: bool = True):
        """
        Initialize Gaussian heatmap generator.
        
        Args:
            shape: 3D volume shape (rows, cols, slices)
            points: List of [row, col, slice] coordinates
            sigma: Standard deviation of Gaussian filter
            use_gpu: Whether to attempt GPU acceleration
            verbose: Whether to print progress messages
        """
        super().__init__(shape, points, use_gpu, verbose)
        self.sigma = sigma
    
    def generate(self) -> np.ndarray:
        """Generate the Gaussian heatmap signal."""
        self._log("  - Creating Gaussian heatmap...")
        
        heatmap = np.zeros(self.shape, dtype=np.float32)
        
        for point in self.points:
            if (0 <= point[0] < self.shape[0] and
                0 <= point[1] < self.shape[1] and
                0 <= point[2] < self.shape[2]):
                heatmap[point[0], point[1], point[2]] = 1.0
        
        if self.sigma > 0:
            gaussian_filter(heatmap, sigma=self.sigma, output=heatmap, mode='constant')
        
        return normalize_signal(heatmap)


class DiskHeatmap(BaseSignal):
    """Flat disk (circle) heatmap signal generator with constant radius."""
    
    def __init__(self, shape: Tuple[int, int, int], points: List[List[int]], 
                 radius: float = 3.0, use_gpu: bool = True, verbose: bool = True):
        """
        Initialize disk heatmap generator.
        
        Args:
            shape: 3D volume shape (rows, cols, slices)
            points: List of [row, col, slice] coordinates
            radius: Radius of the disk (default: 3.0)
            use_gpu: Whether to attempt GPU acceleration
            verbose: Whether to print progress messages
        """
        super().__init__(shape, points, use_gpu, verbose)
        self.radius = radius
    
    def generate(self) -> np.ndarray:
        """Generate the disk heatmap signal."""
        self._log(f"  - Creating disk heatmap with radius={self.radius}...")
        
        # Create output volume
        result = np.zeros(self.shape, dtype=np.float32)
        
        # Get coordinates of all points
        coords = []
        for point in self.points:
            if (0 <= point[0] < self.shape[0] and
                0 <= point[1] < self.shape[1] and
                0 <= point[2] < self.shape[2]):
                coords.append((point[0], point[1], point[2]))
        
        if not coords:
            return result
        
        # Place disks at all coordinates
        Disk.place_disks(result, coords, self.radius, normalize=True)
        
        return normalize_signal(result)


class EuclideanDistance(BaseSignal):
    """Euclidean distance transform signal generator."""
    
    def __init__(self, shape: Tuple[int, int, int], points: List[List[int]], 
                 invert: bool = True, use_gpu: bool = True, verbose: bool = True):
        """
        Initialize Euclidean distance transform generator.
        
        Args:
            shape: 3D volume shape (rows, cols, slices)
            points: List of [row, col, slice] coordinates
            invert: Whether to invert the signal (seeds become bright)
            use_gpu: Whether to attempt GPU acceleration
            verbose: Whether to print progress messages
        """
        super().__init__(shape, points, use_gpu, verbose)
        self.invert = invert
    
    def generate(self) -> np.ndarray:
        """Generate the Euclidean distance signal."""
        self._log("  - Computing Euclidean Distance Transform...")
        
        mask = np.zeros(self.shape, dtype=bool)
        
        for point in self.points:
            if (0 <= point[0] < self.shape[0] and
                0 <= point[1] < self.shape[1] and
                0 <= point[2] < self.shape[2]):
                mask[point[0], point[1], point[2]] = True
        
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                mask_gpu = cp.asarray(mask)
                dist_gpu = gpu_distance_transform(mask_gpu)
                dist = cp.asnumpy(dist_gpu)
            except Exception as e:
                warnings.warn(f"GPU distance transform failed: {e}. Falling back to CPU.")
                dist = distance_transform_edt(~mask)
        else:
            dist = distance_transform_edt(~mask)
        
        dist = dist.astype(np.float32)
        
        if self.invert:
            dist = 1.0 / (dist + 1e-6)
        
        return normalize_signal(dist)


class GeodesicDistance(BaseSignal):
    """Geodesic distance transform signal generator (2D per slice)."""
    
    def __init__(self, shape: Tuple[int, int, int], points: List[List[int]],
                 reference_volume: np.ndarray = None,
                 lambda_val: float = 0.5, iterations: int = 2,
                 method: str = 'raster_scan',
                 debug_slices: bool = False, output_dir: str = None,
                 use_gpu: bool = True, verbose: bool = True):
        """
        Initialize geodesic distance transform generator.
        
        Args:
            shape: 3D volume shape (rows, cols, slices)
            points: List of [row, col, slice] coordinates
            reference_volume: 3D reference volume for speed image
            lambda_val: weighting between 0.0 (Euclidean) and 1.0 (gradient-based)
            iterations: number of raster scan iterations (raster_scan only)
            method: 'raster_scan' or 'fast_marching'
            debug_slices: if True, save debug images for each slice
            output_dir: directory to save debug images
            use_gpu: Whether to attempt GPU acceleration
            verbose: Whether to print progress messages
        """
        super().__init__(shape, points, use_gpu, verbose)
        self.reference_volume = reference_volume
        self.lambda_val = lambda_val
        self.iterations = iterations
        self.method = method
        self.debug_slices = debug_slices
        self.output_dir = output_dir
    
    def generate(self) -> np.ndarray:
        """Generate the geodesic distance signal."""
        self._log("  - Computing 2D Geodesic Distance Transform per slice...")
        
        if not GEODIS_AVAILABLE:
            self._log("  WARNING: GeodisTK not available. Install with: pip install GeodisTK")
            self._log("  Falling back to Euclidean distance transform...")
            return EuclideanDistance(self.shape, self.points, invert=True, 
                                     use_gpu=self.use_gpu, verbose=False).generate()
        
        # Group points by slice
        points_by_slice = defaultdict(list)
        for point in self.points:
            if len(point) >= 3:
                z = int(point[2])
                points_by_slice[z].append((int(point[0]), int(point[1])))
        
        self._log(f"    Found {len(points_by_slice)} unique axial slices with scribbles")
        for slice_idx, pts in points_by_slice.items():
            self._log(f"      Slice {slice_idx}: {len(pts)} points")
        
        rows, cols, slices = self.shape
        geodesic_volume = np.zeros(self.shape, dtype=np.float32)
        
        # Determine which 2D function to use
        if self.method == 'raster_scan' and GEODIS_2D_RASTER:
            self._log(f"    Using GeodisTK 2D raster_scan (lambda={self.lambda_val}, iterations={self.iterations})")
            use_raster = True
        elif self.method == 'fast_marching' and GEODIS_2D_FM:
            self._log("    Using GeodisTK 2D fast_marching")
            use_raster = False
        elif GEODIS_2D_RASTER:
            self._log(f"    Using GeodisTK 2D raster_scan (lambda={self.lambda_val}, iterations={self.iterations})")
            use_raster = True
        elif GEODIS_2D_FM:
            self._log("    Using GeodisTK 2D fast_marching")
            use_raster = False
        else:
            self._log("    No suitable GeodisTK 2D function found. Falling back to Euclidean.")
            return EuclideanDistance(self.shape, self.points, invert=True,
                                     use_gpu=self.use_gpu, verbose=False).generate()
        
        # Process each slice
        slice_indices = sorted(points_by_slice.keys())
        self._log(f"    Processing {len(slice_indices)} slices with scribbles: {slice_indices}")
        
        for slice_idx in slice_indices:
            self._log(f"      Computing geodesic distance for slice {slice_idx}...")
            points_2d = points_by_slice[slice_idx]
            
            # Prepare 2D speed image
            if self.reference_volume is not None:
                I = self.reference_volume[:, :, slice_idx].astype(np.float32)
                if I.max() > I.min():
                    I = (I - I.min()) / (I.max() - I.min() + 1e-6)
                else:
                    I = np.ones((rows, cols), dtype=np.float32)
            else:
                I = np.ones((rows, cols), dtype=np.float32)
            
            # Prepare 2D seed mask
            S = np.zeros((rows, cols), dtype=np.uint8)
            for row, col in points_2d:
                if 0 <= row < rows and 0 <= col < cols:
                    S[row, col] = 1
            
            if np.sum(S) == 0:
                self._log(f"      No seeds found in slice {slice_idx}, skipping...")
                continue
            
            self._log(f"      Seeds in slice: {np.sum(S)}")
            
            try:
                if use_raster:
                    dist_2d = GeodisTK.geodesic2d_raster_scan(I, S, self.lambda_val, self.iterations)
                else:
                    dist_2d = GeodisTK.geodesic2d_fast_marching(I, S)
                
                dist_2d = np.nan_to_num(dist_2d, nan=0.0, posinf=0.0, neginf=0.0)
                self._log(f"        Geodesic stats: min={dist_2d.min():.4f}, max={dist_2d.max():.4f}, mean={dist_2d.mean():.4f}")
                
                dist_2d = 1.0 - normalize_signal(dist_2d)
                geodesic_volume[:, :, slice_idx] = dist_2d.astype(np.float32)
                
                if self.debug_slices and self.output_dir is not None:
                    _save_slice_debug(slice_idx, I, S, dist_2d, self.output_dir, self.method)
                
            except Exception as e:
                warnings.warn(f"GeodisTK 2D computation failed for slice {slice_idx}: {e}")
                self._log(f"      Falling back to Euclidean for slice {slice_idx}")
                mask_2d = ~S.astype(bool)
                dist_2d = distance_transform_edt(mask_2d)
                dist_2d = 1.0 - normalize_signal(dist_2d)
                geodesic_volume[:, :, slice_idx] = dist_2d.astype(np.float32)
                
                if self.debug_slices and self.output_dir is not None:
                    _save_slice_debug(slice_idx, I, S, dist_2d, self.output_dir, 'Euclidean (fallback)')
        
        return normalize_signal(geodesic_volume)


class AdaptiveHeatmap(BaseSignal):
    """
    Adaptive heatmap signal generator with per-point sigma based on geodesic statistics.
    
    This uses the original implementation with:
    - Per-slice percentile buckets
    - Window-based mean geodesic values
    - Disk-based placement for efficiency
    """
    
    def __init__(self, shape: Tuple[int, int, int], points: List[List[int]],
                 geodesic_signal: np.ndarray,
                 sigma_min: float = 1.0, sigma_max: float = 5.0,
                 num_buckets: int = 5, window_size: int = 9,
                 use_disk: bool = True,
                 use_gpu: bool = True, verbose: bool = True):
        """
        Initialize adaptive heatmap generator.
        
        Args:
            shape: 3D volume shape (rows, cols, slices)
            points: List of [row, col, slice] coordinates
            geodesic_signal: 3D geodesic distance map
            sigma_min: Minimum sigma (for high geodesic distance)
            sigma_max: Maximum sigma (for low geodesic distance)
            num_buckets: Number of percentile buckets
            window_size: Size of square window around each seed
            use_disk: If True, use disk-based placement; if False, use Gaussian
            use_gpu: Whether to attempt GPU acceleration
            verbose: Whether to print progress messages
        """
        super().__init__(shape, points, use_gpu, verbose)
        self.geodesic_signal = geodesic_signal
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_buckets = num_buckets
        self.window_size = window_size
        self.use_disk = use_disk
    
    def generate(self) -> np.ndarray:
        """
        Generate the adaptive heatmap signal using the original implementation.
        """
        self._log("  - Computing adaptive heatmap...")
        self._log(f"    Using {'disk' if self.use_disk else 'Gaussian'} placement")
        
        # Create base heatmap with point seeds
        heatmap = np.zeros(self.shape, dtype=np.float32)
        for point in self.points:
            if (0 <= point[0] < self.shape[0] and
                0 <= point[1] < self.shape[1] and
                0 <= point[2] < self.shape[2]):
                heatmap[point[0], point[1], point[2]] = 1.0
        
        # Get the coordinates of all seed points
        seed_coords = np.where(heatmap > 0)
        
        if len(seed_coords[0]) == 0:
            return heatmap
        
        # Group seeds by axial slice
        slices = np.unique(seed_coords[2])
        
        self._log(f"    Adaptive heatmap: {len(slices)} slices with seeds, {self.num_buckets} buckets, "
                  f"window={self.window_size}x{self.window_size}, "
                  f"sigma range: {self.sigma_min:.2f} - {self.sigma_max:.2f}")
        
        # Dictionary to store sigma for each coordinate
        coord_to_sigma = {}
        half_window = self.window_size // 2
        
        # For each slice, compute percentiles and assign sigmas
        for slice_idx in slices:
            # Get seeds in this slice
            slice_mask = seed_coords[2] == slice_idx
            slice_seed_coords = tuple(seed_coords[i][slice_mask] for i in range(3))
            
            if len(slice_seed_coords[0]) == 0:
                continue
            
            # Compute mean geodesic value in window around each seed
            window_means = []
            coord_list = []
            
            for i in range(len(slice_seed_coords[0])):
                row = slice_seed_coords[0][i]
                col = slice_seed_coords[1][i]
                
                # Define window boundaries
                row_start = max(0, row - half_window)
                row_end = min(self.shape[0], row + half_window + 1)
                col_start = max(0, col - half_window)
                col_end = min(self.shape[1], col + half_window + 1)
                
                # Extract window from geodesic signal
                window = self.geodesic_signal[row_start:row_end, col_start:col_end, slice_idx]
                
                # Compute mean of window
                if window.size > 0:
                    mean_val = np.mean(window)
                else:
                    mean_val = self.geodesic_signal[row, col, slice_idx]
                
                window_means.append(mean_val)
                coord_list.append((row, col, slice_idx))
            
            window_means = np.array(window_means)
            
            # Compute percentiles for this slice based on window means
            percentiles = np.percentile(window_means, np.linspace(0, 100, self.num_buckets + 1))
            
            # Create sigma mapping: each bucket gets a sigma
            # Low geodesic distance in window -> homogeneous region -> high sigma (large disk)
            # High geodesic distance in window -> edge/change region -> low sigma (small disk)
            bucket_sigmas = np.linspace(self.sigma_max, self.sigma_min, self.num_buckets)
            
            # Assign sigma to each coordinate
            for i, coord in enumerate(coord_list):
                mean_val = window_means[i]
                
                # Find which bucket this value belongs to
                bucket_idx = 0
                for b in range(self.num_buckets):
                    if b == self.num_buckets - 1:
                        # Last bucket includes the max value
                        if mean_val >= percentiles[b] and mean_val <= percentiles[b + 1]:
                            bucket_idx = b
                            break
                    else:
                        if mean_val >= percentiles[b] and mean_val < percentiles[b + 1]:
                            bucket_idx = b
                            break
                
                # Get sigma for this bucket
                sigma = bucket_sigmas[bucket_idx]
                coord_to_sigma[coord] = sigma
            
            # Print stats for this slice
            self._log(f"      Slice {slice_idx}: {len(coord_list)} seeds, "
                      f"window means: {window_means.min():.3f} - {window_means.max():.3f}")
            self._log(f"        Percentiles: {percentiles[0]:.3f} - {percentiles[-1]:.3f}")
        
        # Group coordinates by sigma value (rounded to 2 decimal places for grouping)
        sigma_groups = defaultdict(list)
        for coord, sigma in coord_to_sigma.items():
            sigma_key = round(sigma, 2)
            sigma_groups[sigma_key].append(coord)
        
        self._log(f"    Grouped {len(coord_to_sigma)} coordinates into {len(sigma_groups)} unique sigma values")
        
        # Create output volume
        result = np.zeros(self.shape, dtype=np.float32)
        
        if self.use_disk:
            # Use disk-based placement (fast)
            self._log("    Using disk-based placement...")
            for sigma, coords in sigma_groups.items():
                radius = self.sigma_max - sigma
                Disk.place_disks(result, coords, radius, normalize=True)
        else:
            # Use Gaussian placement (slower but smoother)
            self._log("    Using Gaussian placement...")
            for sigma, coords in sigma_groups.items():
                # Create a temporary heatmap for these coordinates
                temp = np.zeros(self.shape, dtype=np.float32)
                for coord in coords:
                    row, col, slice_idx = coord
                    if (0 <= row < self.shape[0] and 
                        0 <= col < self.shape[1] and 
                        0 <= slice_idx < self.shape[2]):
                        temp[row, col, slice_idx] = 1.0
                
                # Apply Gaussian filter with this sigma
                if sigma > 0:
                    gaussian_filter(temp, sigma=sigma, output=temp, mode='constant')
                
                # Add to result
                result += temp
        
        # Normalize result
        if result.max() > 0:
            result = result / result.max()
        
        return result


class CombinedSignal(BaseSignal):
    """Combined signal that averages multiple signals."""
    
    def __init__(self, shape: Tuple[int, int, int], points: List[List[int]],
                 signals: List[BaseSignal], weights: Optional[List[float]] = None,
                 use_gpu: bool = True, verbose: bool = True):
        """
        Initialize combined signal generator.
        
        Args:
            shape: 3D volume shape (rows, cols, slices)
            points: List of [row, col, slice] coordinates
            signals: List of signal generators
            weights: Optional list of weights for each signal
            use_gpu: Whether to attempt GPU acceleration
            verbose: Whether to print progress messages
        """
        super().__init__(shape, points, use_gpu, verbose)
        self.signals = signals
        self.weights = weights if weights is not None else [1.0] * len(signals)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def generate(self) -> np.ndarray:
        """Generate the combined signal."""
        self._log("  - Generating combined signal...")
        
        combined = np.zeros(self.shape, dtype=np.float32)
        
        for i, signal in enumerate(self.signals):
            signal.verbose = self.verbose
            result = signal.generate()
            combined += self.weights[i] * result
        
        return normalize_signal(combined)


# -----------------------------------------------------------------------------
# Factory / Generator Class
# -----------------------------------------------------------------------------

class GuidanceSignalGenerator:
    """
    Factory class for generating guidance signals.
    
    This class provides a unified interface for creating different types of
    guidance signals from scribble coordinates.
    """
    
    def __init__(self, ref_path: str, use_gpu: bool = True, verbose: bool = True):
        """
        Initialize the guidance signal generator.
        
        Args:
            ref_path: Path to reference NIfTI file
            use_gpu: Whether to attempt GPU acceleration
            verbose: Whether to print progress messages
        """
        self.ref_path = ref_path
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        # Load reference volume
        self.ref_img = nib.load(ref_path)
        self.ref_data = self.ref_img.get_fdata()
        self.shape = self.ref_data.shape
        
        if self.verbose:
            print(f"Reference volume shape: {self.shape} (rows, cols, slices)")
    
    def generate(self, signal_type: str, points: List[List[int]], **kwargs) -> np.ndarray:
        """
        Generate a guidance signal.
        
        Args:
            signal_type: Type of signal to generate ('gaussian', 'euclidean', 
                        'geodesic', 'adaptive', 'disk', 'combined')
            points: List of [row, col, slice] coordinates
            **kwargs: Additional parameters for the signal generator
        
        Returns:
            3D numpy array with the generated signal
        
        Examples:
            # Gaussian heatmap
            signal = generator.generate('gaussian', points, sigma=2.0)
            
            # Disk heatmap
            signal = generator.generate('disk', points, radius=3.0)
            
            # Geodesic distance
            signal = generator.generate('geodesic', points, lambda_val=0.5)
            
            # Combined signal
            signal = generator.generate('combined', points, 
                                       sigma=2.0, geodesic_lambda=0.5)
        """
        signal_type = signal_type.lower()
        
        if signal_type == 'gaussian':
            sigma = kwargs.get('sigma', 2.0)
            signal = GaussianHeatmap(self.shape, points, sigma=sigma,
                                     use_gpu=self.use_gpu, verbose=self.verbose)
            return signal.generate()
        
        elif signal_type == 'disk':
            radius = kwargs.get('radius', 3.0)
            signal = DiskHeatmap(self.shape, points, radius=radius,
                                 use_gpu=self.use_gpu, verbose=self.verbose)
            return signal.generate()
        
        elif signal_type == 'euclidean':
            invert = kwargs.get('invert', True)
            signal = EuclideanDistance(self.shape, points, invert=invert,
                                       use_gpu=self.use_gpu, verbose=self.verbose)
            return signal.generate()
        
        elif signal_type == 'geodesic':
            lambda_val = kwargs.get('lambda_val', 0.5)
            iterations = kwargs.get('iterations', 2)
            method = kwargs.get('method', 'raster_scan')
            debug_slices = kwargs.get('debug_slices', False)
            output_dir = kwargs.get('output_dir', None)
            
            signal = GeodesicDistance(
                self.shape, points, self.ref_data,
                lambda_val=lambda_val, iterations=iterations,
                method=method, debug_slices=debug_slices,
                output_dir=output_dir,
                use_gpu=self.use_gpu, verbose=self.verbose
            )
            return signal.generate()
        
        elif signal_type == 'adaptive':
            # First compute geodesic signal
            lambda_val = kwargs.get('geodesic_lambda', 0.5)
            iterations = kwargs.get('geodesic_iterations', 2)
            method = kwargs.get('geodesic_method', 'raster_scan')
            
            geodesic_signal = GeodesicDistance(
                self.shape, points, self.ref_data,
                lambda_val=lambda_val, iterations=iterations,
                method=method, debug_slices=False,
                output_dir=None,
                use_gpu=self.use_gpu, verbose=False
            ).generate()
            
            sigma_min = kwargs.get('sigma_min', 1.0)
            sigma_max = kwargs.get('sigma_max', 5.0)
            num_buckets = kwargs.get('num_buckets', 5)
            window_size = kwargs.get('window_size', 9)
            use_disk = kwargs.get('use_disk', True)
            
            signal = AdaptiveHeatmap(
                self.shape, points, geodesic_signal,
                sigma_min=sigma_min, sigma_max=sigma_max,
                num_buckets=num_buckets, window_size=window_size,
                use_disk=use_disk,
                use_gpu=self.use_gpu, verbose=self.verbose
            )
            return signal.generate()
        
        elif signal_type == 'combined':
            # Build individual signals
            sigma = kwargs.get('sigma', 2.0)
            radius = kwargs.get('radius', 3.0)
            lambda_val = kwargs.get('geodesic_lambda', 0.5)
            iterations = kwargs.get('geodesic_iterations', 2)
            method = kwargs.get('geodesic_method', 'raster_scan')
            sigma_min = kwargs.get('sigma_min', 1.0)
            sigma_max = kwargs.get('sigma_max', 5.0)
            num_buckets = kwargs.get('num_buckets', 5)
            window_size = kwargs.get('window_size', 9)
            use_disk = kwargs.get('use_disk', True)
            
            # Geodesic signal (needed for adaptive)
            geodesic_signal = GeodesicDistance(
                self.shape, points, self.ref_data,
                lambda_val=lambda_val, iterations=iterations,
                method=method, debug_slices=False,
                output_dir=None,
                use_gpu=self.use_gpu, verbose=False
            ).generate()
            
            signals = [
                GaussianHeatmap(self.shape, points, sigma=sigma,
                               use_gpu=self.use_gpu, verbose=False),
                DiskHeatmap(self.shape, points, radius=radius,
                           use_gpu=self.use_gpu, verbose=False),
                EuclideanDistance(self.shape, points, invert=True,
                                 use_gpu=self.use_gpu, verbose=False),
                AdaptiveHeatmap(self.shape, points, geodesic_signal,
                               sigma_min=sigma_min, sigma_max=sigma_max,
                               num_buckets=num_buckets, window_size=window_size,
                               use_disk=use_disk,
                               use_gpu=self.use_gpu, verbose=False)
            ]
            
            combined = CombinedSignal(self.shape, points, signals,
                                      use_gpu=self.use_gpu, verbose=self.verbose)
            return combined.generate()
        
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
    
    def save(self, signal: np.ndarray, output_path: str) -> None:
        """Save the signal as a NIfTI file."""
        output_img = nib.Nifti1Image(signal.astype(np.float32), 
                                     self.ref_img.affine, 
                                     self.ref_img.header)
        nib.save(output_img, output_path)
        if self.verbose:
            print(f"Guidance signal saved to: {output_path}")
    
    def debug_visualize(self, signal: np.ndarray, points: List[List[int]], 
                         output_path: str, signal_name: str = "Signal") -> None:
        """Save debug visualizations of the signal and reference slice."""
        save_debug_slice(signal, points, output_path, signal_name, slice_dim=2)
        save_reference_slice(self.ref_data, points, output_path, slice_dim=2)


# -----------------------------------------------------------------------------
# CLI Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D guidance signals from scribble coordinates.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate combined signal with debug visualization
  python guidance_signal.py --json clicks.json --ref ref.nii.gz --output guidance.nii.gz --debug

  # Generate only disk heatmap with radius 5
  python guidance_signal.py --json clicks.json --ref ref.nii.gz --output disk.nii.gz --signal disk --radius 5.0

  # Generate only geodesic signal with slice-level debugging
  python guidance_signal.py --json clicks.json --ref ref.nii.gz --output geodesic.nii.gz --signal geodesic --debug-slices

  # Generate adaptive heatmap with custom buckets
  python guidance_signal.py --json clicks.json --ref ref.nii.gz --output adaptive.nii.gz --signal adaptive --sigma-min 0.5 --sigma-max 9.0 --sigma-buckets 20 --window-size 15

  # Generate adaptive heatmap with Gaussian placement instead of disk
  python guidance_signal.py --json clicks.json --ref ref.nii.gz --output adaptive_gaussian.nii.gz --signal adaptive --no-disk

  # Custom geodesic parameters
  python guidance_signal.py --json clicks.json --ref ref.nii.gz --output guidance.nii.gz --geodesic-lambda 0.8 --geodesic-iterations 3
        """
    )

    # Required arguments
    parser.add_argument('--json', required=True, help='Path to JSON file with scribble coordinates')
    parser.add_argument('--ref', required=True, help='Path to reference NIfTI file (.nii.gz)')
    parser.add_argument('--output', required=True, help='Output path for NIfTI file')

    # Signal type
    parser.add_argument(
        '--signal',
        choices=['gaussian', 'euclidean', 'geodesic', 'adaptive', 'disk', 'combined'],
        default='combined',
        help='Type of guidance signal to generate (default: combined)'
    )

    # Debug options
    parser.add_argument('--debug', action='store_true', help='Save debug PNG of slice with most points')
    parser.add_argument('--debug-slices', action='store_true', help='Save debug images for each geodesic slice')

    # Gaussian parameters
    parser.add_argument('--sigma', type=float, default=2.0, help='Sigma for Gaussian heatmap (default: 2.0)')

    # Disk parameters
    parser.add_argument('--radius', type=float, default=3.0, help='Radius for disk heatmap (default: 3.0)')

    # Adaptive heatmap parameters
    parser.add_argument('--sigma-min', type=float, default=1.0,
                        help='Minimum sigma for adaptive heatmap (high geodesic distance, default: 1.0)')
    parser.add_argument('--sigma-max', type=float, default=5.0,
                        help='Maximum sigma for adaptive heatmap (low geodesic distance, default: 5.0)')
    parser.add_argument('--sigma-buckets', type=int, default=5,
                        help='Number of percentile buckets for adaptive heatmap (default: 5)')
    parser.add_argument('--window-size', type=int, default=9,
                        help='Window size around each seed for mean geodesic value (default: 9)')
    parser.add_argument('--no-disk', action='store_true',
                        help='Use Gaussian placement instead of disk for adaptive heatmap')

    # Hardware options
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')

    # GeodisTK parameters
    parser.add_argument(
        '--geodesic-method',
        choices=['raster_scan', 'fast_marching'],
        default='raster_scan',
        help='GeodisTK method (default: raster_scan)'
    )
    parser.add_argument(
        '--geodesic-lambda',
        type=float,
        default=0.5,
        help='Lambda for GeodisTK (0.0=Euclidean, 1.0=gradient-based, default: 0.5)'
    )
    parser.add_argument(
        '--geodesic-iterations',
        type=int,
        default=2,
        help='Iterations for raster_scan (2-4 typically, default: 2)'
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Print library status
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("LIBRARY STATUS")
    print("=" * 50)
    print(f"GeodisTK available: {GEODIS_AVAILABLE}")
    if GEODIS_AVAILABLE:
        print(f"  - geodesic2d_raster_scan: {GEODIS_2D_RASTER}")
        print(f"  - geodesic2d_fast_marching: {GEODIS_2D_FM}")
    print(f"CuPy available: {CUPY_AVAILABLE}")
    print("=" * 50 + "\n")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    try:
        tumor_points, background_points = parse_json_coordinates(args.json)
        print(f"Found {len(tumor_points)} tumor points and {len(background_points)} background points")
        print("Using ONLY tumor points for guidance signal generation")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return

    if not tumor_points:
        print("No tumor points found in JSON file. Exiting.")
        return

    print(f"Using {len(tumor_points)} tumor points for guidance signal")

    # -------------------------------------------------------------------------
    # Generate guidance signal
    # -------------------------------------------------------------------------
    use_gpu = not args.no_gpu and CUPY_AVAILABLE
    
    # Create generator
    generator = GuidanceSignalGenerator(args.ref, use_gpu=use_gpu, verbose=True)

    # Create debug directory if needed
    debug_dir = None
    if args.debug_slices and args.signal in ['geodesic', 'adaptive', 'combined']:
        debug_dir = os.path.join(os.path.dirname(args.output), 'geodesic_debug')
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug slices will be saved to: {debug_dir}")

    print(f"\nGenerating {args.signal} guidance signal...")

    # Generate signal
    if args.signal == 'gaussian':
        signal = generator.generate('gaussian', tumor_points, sigma=args.sigma)
        signal_name = "Gaussian"
    
    elif args.signal == 'disk':
        signal = generator.generate('disk', tumor_points, radius=args.radius)
        signal_name = f"Disk (r={args.radius})"
    
    elif args.signal == 'euclidean':
        signal = generator.generate('euclidean', tumor_points, invert=True)
        signal_name = "Euclidean"
    
    elif args.signal == 'geodesic':
        signal = generator.generate('geodesic', tumor_points,
                                   lambda_val=args.geodesic_lambda,
                                   iterations=args.geodesic_iterations,
                                   method=args.geodesic_method,
                                   debug_slices=args.debug_slices,
                                   output_dir=debug_dir)
        signal_name = "Geodesic"
    
    elif args.signal == 'adaptive':
        signal = generator.generate('adaptive', tumor_points,
                                   geodesic_lambda=args.geodesic_lambda,
                                   geodesic_iterations=args.geodesic_iterations,
                                   geodesic_method=args.geodesic_method,
                                   sigma_min=args.sigma_min,
                                   sigma_max=args.sigma_max,
                                   num_buckets=args.sigma_buckets,
                                   window_size=args.window_size,
                                   use_disk=not args.no_disk)
        signal_name = "Adaptive"
    
    else:  # combined
        signal = generator.generate('combined', tumor_points,
                                   sigma=args.sigma,
                                   radius=args.radius,
                                   geodesic_lambda=args.geodesic_lambda,
                                   geodesic_iterations=args.geodesic_iterations,
                                   geodesic_method=args.geodesic_method,
                                   sigma_min=args.sigma_min,
                                   sigma_max=args.sigma_max,
                                   num_buckets=args.sigma_buckets,
                                   window_size=args.window_size,
                                   use_disk=not args.no_disk)
        signal_name = "Combined"

    print(f"Signal range: [{signal.min():.4f}, {signal.max():.4f}]")

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    generator.save(signal, args.output)

    # -------------------------------------------------------------------------
    # Debug visualization
    # -------------------------------------------------------------------------
    if args.debug:
        print("\nGenerating debug visualization...")
        generator.debug_visualize(signal, tumor_points, args.output, signal_name)

    print("\nDone!")


if __name__ == "__main__":
    main()