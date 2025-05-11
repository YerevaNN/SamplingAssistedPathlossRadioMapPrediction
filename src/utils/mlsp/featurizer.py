import random

import math
import numpy as np
import torch
from numba import njit
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

from src.utils.mlsp.types import RadarSample


@njit
def _calculate_transmittance_loss_numpy(
    transmittance_matrix, x_ant, y_ant, n_angles=360 * 128 / 1, radial_step=1.0, max_walls=10
):
    """
    Numpy implementation for numba optimization.
    This function must stay as numpy for numba to work.
    """
    h, w = transmittance_matrix.shape
    dtheta = 2.0 * np.pi / n_angles
    output = np.zeros((h, w), dtype=transmittance_matrix.dtype)
    
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)
    max_dist = np.sqrt(w * w + h * h)
    
    for i in range(n_angles):
        cos_t = cos_vals[i]
        sin_t = sin_vals[i]
        
        sum_loss = 0.0
        last_val = None
        wall_count = 0
        r = 0.0
        
        while r <= max_dist:
            x = x_ant + r * cos_t
            y = y_ant + r * sin_t
            
            px = int(round(x))
            py = int(round(y))
            
            if px < 0 or px >= w or py < 0 or py >= h:
                if last_val is not None and last_val > 0:
                    sum_loss += last_val
                    wall_count += 1
                    if wall_count >= max_walls:
                        pass  # Already out of bounds, so we do nothing more
                break
            
            val = transmittance_matrix[py, px]
            
            if last_val is None:
                last_val = val
            
            if val != last_val:
                if last_val > 0 and val == 0:
                    sum_loss += last_val
                    wall_count += 1
                    if wall_count >= max_walls:
                        r_temp = r
                        while r_temp <= max_dist:
                            x_temp = x_ant + r_temp * cos_t
                            y_temp = y_ant + r_temp * sin_t
                            px_temp = int(round(x_temp))
                            py_temp = int(round(y_temp))
                            
                            if px_temp < 0 or px_temp >= w or py_temp < 0 or py_temp >= h:
                                break
                            
                            if output[py_temp, px_temp] == 0 or sum_loss < output[py_temp, px_temp]:
                                output[py_temp, px_temp] = sum_loss
                            r_temp += radial_step
                        break
                last_val = val
            
            if output[py, px] == 0 or (sum_loss < output[py, px]):
                output[py, px] = sum_loss
            
            r += radial_step
    
    return output


def calculate_transmittance_loss(
    transmittance_matrix, x_ant, y_ant, n_angles=360 * 128 / 1, radial_step=1.0, max_walls=10
):
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_transmittance_loss_numpy(transmittance_np, x_ant, y_ant, n_angles, radial_step, max_walls)
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


def calculate_fspl(
    dist_m,  # distance in meters (torch tensor)
    freq_MHz,  # frequency in MHz
    antenna_gain,  # shape=(360,) antenna gain in dBi [0..359]
    min_dist_m=0.125,  # clamp distance below this
):
    dist_clamped = torch.clamp(dist_m, min=min_dist_m)
    freq_tensor = torch.tensor(freq_MHz, device=torch.device('cpu'))
    fspl_db = 20.0 * torch.log10(dist_clamped) + 20.0 * torch.log10(freq_tensor) - 27.55
    pathloss_db = fspl_db - antenna_gain
    
    return pathloss_db


def calculate_pl_init(
    sample: RadarSample,
    distance,
    antenna_gain,
    transmittance,
):
    # Calculate free space path loss on CPU
    free_space_pathloss = calculate_fspl(
        dist_m=distance,
        freq_MHz=sample.freq_MHz,
        antenna_gain=antenna_gain,
    )
    
    # Calculate transmittance loss on CPU
    transmittance_loss = calculate_transmittance_loss(
        transmittance,
        sample.x_ant,
        sample.y_ant
    )
    
    pl_init = free_space_pathloss + transmittance_loss
    if sample.pl_clip != float("inf"):
        pl_init = torch.minimum(pl_init, sample.pl_clip)
    
    return pl_init


def calculate_antenna_gain(radiation_pattern, W, H, azimuth, x_ant, y_ant):
    """
    Calculate antenna gain across a grid based on radiation pattern and antenna orientation.
    Works with torch tensors.
    """
    x_grid = torch.arange(W, device=torch.device('cpu')).expand(H, W)
    y_grid = torch.arange(H, device=torch.device('cpu')).view(-1, 1).expand(H, W)
    angles = -(180 / torch.pi) * torch.atan2((y_ant - y_grid), (x_ant - x_grid)) + 180 + azimuth
    angles = torch.where(angles > 359, angles - 360, angles).to(torch.long)
    antenna_gain = radiation_pattern[angles]
    
    return antenna_gain


def normalize_input(input_tensor):
    min_antenna_gain = -55.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    normalized = input_tensor.clone()
    for i in range(2):
        
        normalized[i] = (normalized[i] / 255.0 - mean[i]) / std[i]
    normalized[2] = torch.log10(1 + normalized[2])
    normalized[3] = normalized[3] / min_antenna_gain
    normalized[4] = torch.log10(normalized[4]) - 1.9  # "magic shift"
    # normalized[5] = normalized[5] / 100.0  # this feature mask has values < 300
    return normalized

@njit
def select_indices(cand_idx, W, num_points, min_sep):
    sel = np.empty((num_points, 2), np.int64)
    count = 0
    min_sep_sq = min_sep * min_sep
    for idx in cand_idx:
        if count >= num_points:
            break
        r = idx // W
        c = idx - r * W
        ok = True
        for j in range(count):
            dr = r - sel[j, 0]
            dc = c - sel[j, 1]
            if dr * dr + dc * dc < min_sep_sq:
                ok = False
                break
        if ok:
            sel[count, 0] = r
            sel[count, 1] = c
            count += 1
    return sel, count


def add_points(matrix, x_ant, y_ant, num_points, alpha=2.0, min_sep=None, oversample=10):
    H, W = matrix.shape
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    dist = np.hypot(ys - y_ant, xs - x_ant).flatten()
    probs = dist ** alpha
    total = probs.sum()
    if total == 0:
        probs[:] = 1
        total = H * W
    probs /= total
    cand = np.random.choice(H * W, num_points * oversample, True, p=probs)
    cand = np.unique(cand)
    np.random.shuffle(cand)
    if min_sep is None:
        min_sep = 0.5 * math.sqrt(H * W / num_points)
    sel_arr, cnt = select_indices(cand.astype(np.int64), W, num_points, min_sep)
    selected = [(int(sel_arr[i, 0]), int(sel_arr[i, 1])) for i in range(cnt)]
    if cnt < num_points:
        rem = np.setdiff1d(np.arange(H * W), [r * W + c for r, c in selected], assume_unique=False)
        fill = np.random.choice(rem, num_points - cnt, False, p=probs[rem] / probs[rem].sum())
        for idx in fill:
            selected.append(divmod(int(idx), W))
    for r, c in selected:
        matrix[r, c] = 1
    return selected


def sparse_sampling(sample: RadarSample, training: bool, inference: bool, sparsity_range: tuple[float, float]):
    """
    Sparse sampling of the output pathloss values.
    """
    # select random pixels to keep
    # if training:
    #     sample.mask = torch.rand_like(sample.mask) > p
    # if not training or inference:
    # Exactly p% of the pixels are set to 0
    # noinspection PyTypeChecker
    p = max(0, random.uniform(*sparsity_range))
    total_pixels = sample.H * sample.W
    num_zeros = int(total_pixels * p)
    flat_mask = torch.ones(total_pixels, dtype=torch.bool)
    flat_mask[:num_zeros] = 0
    flat_mask = flat_mask[torch.randperm(total_pixels)]
    sampling_mask = flat_mask.view(sample.H, sample.W)
    
    # add a new channels to the input where the mask is False, take values from the output
    if inference:
        transmittance = sample.input_img[1]  # Second channel
        distance = sample.input_img[2]  # Third channel
        radiation_pattern = sample.radiation_pattern
        antenna_gain = calculate_antenna_gain(
            radiation_pattern,
            sample.W,
            sample.H,
            sample.azimuth,
            sample.x_ant,
            sample.y_ant
        )
        if sample.output_img != "":
            sparse_input = sample.output_img
        else:
            pl_init = calculate_pl_init(sample, distance, antenna_gain, transmittance)
            sparse_input = pl_init * (~sampling_mask)
    else:
        sparse_input = sample.output_img * (~sampling_mask)
    sample.input_img = torch.cat([sample.input_img, sparse_input.unsqueeze(0)], dim=0)
    
    return sample


def kriging(
    pl_init, sparse_sample,
    length_scale=20.0, length_scale_bounds=(1.0, 100.0),
    constant_value=1.0, constant_value_bounds=(1e-3, 1e3)
):
    h, w = pl_init.shape
    mask = sparse_sample != 0
    sample_indices = np.argwhere(mask)
    x_train = sample_indices.T
    y_train = sparse_sample[mask] - pl_init[mask]
    
    # Define the Gaussian Process model
    kernel = C(constant_value, constant_value_bounds) * RBF(length_scale, length_scale_bounds)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1.0)
    
    # Fit the GP model on residuals
    gp.fit(x_train, y_train)
    
    # Predict residuals for all pixels
    xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_pred = np.column_stack([xx.ravel(), yy.ravel()])
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    
    # Create the adjusted pathloss map
    residual_map = y_pred.reshape((h, w))
    adjusted_map = pl_init + residual_map
    return adjusted_map


def interpolate_difference(pl_init: np.ndarray, sparse_sample: np.ndarray, method="linear") -> np.ndarray:
    """
    Interpolates the difference between pl_init and sparse_sample using linear interpolation.

    Parameters:
        pl_init (np.ndarray): Initial pathloss estimate (H x W).
        sparse_sample (np.ndarray): Sparse ground truth measurements (H x W), with 0 indicating missing data.
        method (str): Interpolation method, one of 'linear', 'nearest', or 'cubic'.

    Returns:
        np.ndarray: Updated pathloss map after interpolation-based correction.
    """
    # Identify valid measurement positions
    try:
        mask = sparse_sample > 0  # or ~np.isnan(sparse_sample) if using NaNs
        
        coords = np.column_stack(np.nonzero(mask)).T  # (N, 2) -> (row, col)
        diff_values = (sparse_sample - pl_init)[mask]  # Difference at valid points
        
        # Generate full grid
        h, w = pl_init.shape
        grid_x, grid_y = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # Interpolate
        interpolated_diff = griddata(coords, diff_values, grid_points, method=method, fill_value=0)
        interpolated_diff = interpolated_diff.reshape(pl_init.shape)
        
        # Apply correction
        corrected_map = pl_init + interpolated_diff
        return corrected_map
    except Exception as ex:
        return pl_init


def extrapolate_difference(pl_init: np.ndarray, sparse_sample: np.ndarray, neighbors=10) -> np.ndarray:
    mask = (sparse_sample > 0) & np.isfinite(sparse_sample)
    coords = np.array(np.nonzero(mask))
    diff_values = (sparse_sample - pl_init)[mask]
    
    H, W = pl_init.shape
    grid_x, grid_y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    
    rbf = RBFInterpolator(coords, diff_values, neighbors=neighbors, smoothing=0.1)
    extrapolated_diff = rbf(grid_points).reshape(H, W)
    
    return pl_init + extrapolated_diff


def featurizer(sample: RadarSample) -> torch.Tensor:
    reflectance = sample.input_img[0]  # First channel
    transmittance = sample.input_img[1]  # Second channel
    distance = sample.input_img[2]  # Third channel
    sparse_sample = sample.input_img[3]  # Fourth channel
    radiation_pattern = sample.radiation_pattern
    antenna_gain = calculate_antenna_gain(
        radiation_pattern,
        sample.W,
        sample.H,
        sample.azimuth,
        sample.x_ant,
        sample.y_ant
    )
    
    pl_init = calculate_pl_init(sample, distance, antenna_gain, transmittance)
    # pl_adjusted = interpolate_difference(pl_init, sparse_sample, method="linear")
    # Build input tensor: [8, H, W] - Already in correct (C, H, W) format
    
    input_tensor = torch.zeros((8, sample.H, sample.W), dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[0] = reflectance  # reflectance
    input_tensor[1] = transmittance  # transmittance
    input_tensor[2] = distance  # distance
    input_tensor[3] = antenna_gain
    input_tensor[4] = torch.full((sample.H, sample.W), sample.freq_MHz, dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[5] = pl_init
    input_tensor = normalize_input(input_tensor)
    mask = sample.mask
    input_tensor[6] = mask
    input_tensor[7] = sparse_sample  # sparse sample
    
    return input_tensor


_calculate_transmittance_loss_numpy(np.array([[1]]), 0, 0)
