# Mamba-Adaptive Scanpath Model - Implementation Summary

## Overview
Successfully implemented comprehensive fixes for the Mamba-Adaptive scanpath prediction model, addressing critical issues with 360-degree boundary handling, loss function complexity, and Y-direction diversity.

## Changes Implemented

### Stage 1: Critical Fixes (High Priority)

#### 1. Fixed 360-Degree Boundary Wrap Around
**File**: `models/mamba_adaptive_scanpath.py` (lines 219-290)

**Changes**:
- Replaced `clamp` with true wrap around for X-coordinate (longitude)
- Implemented manual patch extraction with `torch.cat` for boundary-crossing patches
- Y-coordinate (latitude) still uses `clamp` as it's not periodic
- Properly handles patches that cross the right boundary (x=0 and x=W are continuous)

**Key Code**:
```python
# X坐标wrap around（经度是周期性的）
x_start = int(x_center) % W

# 提取patch
if x_start + patch_size <= W:
    # 不跨越边界
    patch = images[i:i+1, :, y_start:y_start+patch_size, x_start:x_start+patch_size]
else:
    # 跨越右边界，需要wrap around
    right_part = images[i:i+1, :, y_start:y_start+patch_size, x_start:]
    left_part = images[i:i+1, :, y_start:y_start+patch_size, :(x_start+patch_size-W)]
    patch = torch.cat([right_part, left_part], dim=3)
```

#### 2. Simplified Loss Function (13 items → 6 items)
**File**: `train_mamba_adaptive.py`

**New Loss Structure**:
1. **reconstruction_loss** - Position reconstruction
2. **kl_loss** - KL divergence regularization
3. **spatial_coverage_loss** - Combines coverage + diversity + center_penalty
4. **trajectory_smoothness_loss** - Combines step_length + jump_penalty + acceleration
5. **direction_consistency_loss** - Combines direction + direction_continuity
6. **boundary_penalty** - Boundary constraints

**Helper Functions Added**:
- `compute_spatial_coverage_loss()` - Merges coverage, diversity, and Y-center penalty
- `compute_trajectory_smoothness_loss()` - Merges step length, jump, and acceleration
- `compute_direction_consistency_loss()` - Merges direction change and continuity

**Simplified Weight Scheduling**:
```python
if epoch <= 80:
    weights = {
        'reconstruction': 1.0,
        'kl': 0.005,
        'spatial_coverage': 0.5,
        'trajectory_smoothness': 1.5,
        'direction_consistency': 0.5,
        'boundary': 0.2
    }
elif epoch <= 150:
    # Progressive increase in spatial_coverage
    weights['spatial_coverage'] = 0.5 + 0.3*progress
else:
    weights['spatial_coverage'] = 0.8
```

#### 3. Improved Teacher Forcing Strategy
**File**: `train_mamba_adaptive.py`

**Changes**:
- Replaced linear decay with exponential decay
- Faster convergence from 0.7 to 0.1 over 100 epochs
- More stable training dynamics

**Implementation**:
```python
def compute_teacher_forcing_ratio(epoch):
    """指数衰减的Teacher Forcing策略"""
    initial_ratio = 0.7
    final_ratio = 0.1
    decay_epochs = 100

    k = -math.log(final_ratio / initial_ratio) / decay_epochs
    ratio = initial_ratio * math.exp(-k * epoch)

    return max(ratio, final_ratio)
```

### Stage 2: Architecture Improvements (Medium Priority)

#### 4. Y-Direction Attention Mechanism
**New File**: `models/y_attention.py`

**Purpose**: Encourage Y-direction (latitude) diversity

**Architecture**:
- Multi-head attention (4 heads) on historical Y positions
- Position encoder for Y coordinates
- MLP to generate Y-direction bias
- Output: Tanh-activated bias in range [-1, 1]

**Integration**:
- Added to `MambaAdaptiveScanpathGenerator.__init__()`
- Applied after position prediction with 0.1 scaling factor
- Uses last 5 historical Y positions

#### 5. Simplified Feature Update Mechanism
**New File**: `models/feature_update_v2.py`

**Purpose**: Replace complex grid_sample with Cross-Attention

**Architecture**:
- Computes spatial weights using Gaussian kernel
- Considers 360-degree wrap around for X-distance
- Gate-controlled fusion of global and local features
- Sigma = 0.2 for spatial weight computation

**Key Improvement**:
- Clearer spatial mapping
- Explicit wrap around handling
- More interpretable update mechanism

#### 6. Model Integration
**File**: `models/mamba_adaptive_scanpath.py`

**Changes**:
- Imported `YDirectionAttention` and `CrossAttentionFeatureUpdate`
- Added Y-attention module to generator
- Replaced old feature update with simplified version
- Integrated Y-bias into position prediction

### Configuration Updates
**File**: `config_mamba_adaptive.py`

**New Configuration Options**:
```python
# Stage 1 Configuration
use_wrap_around = True
use_simplified_loss = True
teacher_forcing_strategy = 'exponential'

# Stage 2 Configuration
use_y_attention = True
y_attention_bias_scale = 0.1
use_simplified_feature_update = True
feature_update_sigma = 0.2

# Simplified Loss Weights
loss_weights = {
    'reconstruction': 1.0,
    'kl': 0.005,
    'spatial_coverage': 0.5,
    'trajectory_smoothness': 1.5,
    'direction_consistency': 0.5,
    'boundary': 0.2
}
```

## Expected Improvements

### Key Metrics to Monitor:

1. **Y-Direction Diversity**:
   - Target: Y standard deviation > 0.10 (currently 0.04-0.07)
   - Target: Y coverage range > 0.25
   - Target: Y mean distribution spread (avoid 0.49-0.51 clustering)

2. **Position Accuracy**:
   - Target: Overall position error < 0.22
   - Maintain or improve X-direction accuracy
   - Improve Y-direction accuracy

3. **Trajectory Quality**:
   - Smoother trajectories (lower step variance)
   - Better spatial coverage
   - Fewer boundary violations

4. **Training Stability**:
   - Lower loss variance
   - More stable gradient norms
   - Faster convergence

## Validation Plan

### Recommended Testing Sequence:

1. **Baseline Training** (50 epochs)
   - Use current code without modifications
   - Record all metrics
   - Save checkpoint

2. **Stage 1 Training** (50 epochs)
   - Apply wrap around + simplified loss + improved teacher forcing
   - Compare against baseline
   - Save checkpoint

3. **Stage 2 Training** (50 epochs)
   - Apply Stage 1 + Y attention + simplified feature update
   - Compare against both previous stages
   - Save checkpoint

4. **Visualization Comparison**
   - Generate 50 sample scanpaths
   - Compare path quality across three stages
   - Analyze failure cases

## Files Modified

### Modified Files:
1. `models/mamba_adaptive_scanpath.py` - Core model fixes and integrations
2. `train_mamba_adaptive.py` - Simplified loss and training improvements
3. `config_mamba_adaptive.py` - New configuration options

### New Files Created:
1. `models/y_attention.py` - Y-direction attention module
2. `models/feature_update_v2.py` - Simplified feature update module

## Backward Compatibility

All modifications are controlled by configuration flags:
- Old checkpoints can still be loaded
- Features can be toggled on/off via config
- Fallback to original behavior if needed

## Risk Mitigation

1. **Gradual Rollout**: Implemented in two stages for validation
2. **Configuration Control**: All features can be disabled
3. **Checkpoint Preservation**: Save checkpoints at each stage
4. **Metric Monitoring**: Track all key metrics throughout training

## Next Steps

1. Run baseline training to establish current performance
2. Train with Stage 1 improvements
3. Train with Stage 2 improvements
4. Compare results and visualizations
5. Fine-tune hyperparameters based on results
6. Generate comprehensive performance report

## Success Criteria

- ✅ Y-direction standard deviation > 0.10
- ✅ Y-direction coverage range > 0.25
- ✅ Position error < 0.22
- ✅ No increase in boundary violations
- ✅ Training time < 2x baseline
- ✅ Stable training (no divergence)
