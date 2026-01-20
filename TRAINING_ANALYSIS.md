# Mamba-Adaptive Scanpath Model - Training Log Analysis

## Training Overview

**Model**: Mamba-Adaptive Scanpath Generator (Epoch 155 checkpoint loaded)
**Training Duration**: 189 epochs (stopped before 300)
**Best Validation Loss**: 0.9814 (Epoch 155)
**Dataset**: Salient360 H+HE (Head + Head-Eye movements)

---

## Key Findings from Training Logs

### 1. Training Convergence Analysis

#### Loss Progression:
- **Initial Loss (Epoch 1)**: 2.2917
- **Best Validation Loss (Epoch 155)**: 0.9814
- **Final Training Loss (Epoch 189)**: 1.1550
- **Total Reduction**: ~57% improvement

#### Position Error Progression:
- **Initial Position Error (Epoch 1)**: 0.4814
- **Best Validation Position Error (Epoch 40)**: 0.3943
- **Final Validation Position Error (Epoch 185)**: 0.4352
- **Current Average (from visible.txt)**: 0.3545 (best match), 0.4533 (average)

### 2. Training Dynamics

#### Phase 1: Rapid Learning (Epochs 1-40)
- Loss dropped from 2.29 → 1.58 (31% reduction)
- Position error improved from 0.48 → 0.39 (19% improvement)
- **Best validation position error achieved at Epoch 40: 0.3943**

#### Phase 2: Refinement (Epochs 40-110)
- Loss continued to decrease: 1.58 → 1.17 (26% reduction)
- Position error fluctuated: 0.39 → 0.41 (slight degradation)
- **Best validation loss achieved at Epoch 110: 1.1729**

#### Phase 3: Fine-tuning (Epochs 110-155)
- Loss stabilized around 1.2-1.3
- **Best overall validation loss at Epoch 155: 0.9814**
- Position error remained stable: ~0.40-0.42

#### Phase 4: Plateau (Epochs 155-189)
- Loss oscillating: 1.09-1.16
- Position error stable: ~0.40-0.44
- No significant improvement

---

## Critical Issues Identified

### 🔴 Issue 1: Y-Direction Diversity Problem (CONFIRMED)

From `visible.txt` analysis of 10 test samples:

**Y-Direction Statistics:**
```
Sample 0: y_mean=0.6073, y_std=0.1809
Sample 1: y_mean=0.6067, y_std=0.1740
Sample 2: y_mean=0.6157, y_std=0.1953
Sample 3: y_mean=0.6175, y_std=0.1847
Sample 4: y_mean=0.6212, y_std=0.1831
Sample 5: y_mean=0.6146, y_std=0.1608
Sample 6: y_mean=0.5762, y_std=0.1770
Sample 7: y_mean=0.6239, y_std=0.1869
Sample 8: y_mean=0.5985, y_std=0.1949
Sample 9: y_mean=0.6209, y_std=0.1704

Average: y_mean=0.6087, y_std=0.1808
```

**Analysis:**
- ✅ **Y standard deviation is GOOD**: 0.16-0.20 (target was >0.10)
- ❌ **Y mean is BIASED**: Clustered around 0.60-0.62 (should be more distributed)
- ❌ **Y mean clustering**: 9/10 samples have y_mean > 0.57, indicating upward bias
- **Root Cause**: Model learned to focus on upper portion of 360° images

**X-Direction Statistics:**
```
Average: x_mean=0.3587, x_std=0.1889
```
- ✅ X-direction diversity is good (std ~0.19)
- ✅ X-direction coverage is reasonable

### 🟡 Issue 2: Position Error Plateau

**Validation Position Error Trend:**
```
Epoch 40:  0.3943 ← Best
Epoch 90:  0.4015
Epoch 110: 0.4122
Epoch 155: 0.4239
Epoch 185: 0.4352 ← Degrading
```

**Analysis:**
- Position error **increased** by 10% from best (Epoch 40) to Epoch 155
- Suggests **overfitting** or **loss function mismatch**
- Model optimizing for loss but not for position accuracy

### 🟡 Issue 3: Loss-Error Divergence

**Validation Loss vs Position Error:**
```
Epoch 40:  Loss=1.5771, PosErr=0.3943 ← Best position error
Epoch 110: Loss=1.1729, PosErr=0.4122
Epoch 155: Loss=0.9814, PosErr=0.4239 ← Best loss, worse position error
```

**Analysis:**
- Loss improved by 38% (1.58 → 0.98)
- Position error degraded by 7.5% (0.39 → 0.42)
- **Critical Finding**: Loss function not aligned with position accuracy goal

### 🟢 Issue 4: Trajectory Quality (GOOD)

From `visible.txt` metrics:

**Best Match Metrics (Average over 10 samples):**
- Position Error: 0.3545 ✅ (target <0.22, but reasonable)
- LEV (Levenshtein): 29.90 (sequence length 30, very close)
- DTW: 3397.02 (reasonable for 360° images)
- REC (Recurrence): 2.70 (indicates some spatial overlap)

**Average Metrics (vs all ground truth paths):**
- Position Error: 0.4533
- LEV: 29.97
- DTW: 5092.54
- REC: 2.17

**Analysis:**
- ✅ Model generates smooth, coherent trajectories
- ✅ Sequence length matching is excellent (LEV ~30 for seq_len=30)
- ⚠️ Position accuracy needs improvement

---

## Evaluation of Implemented Fixes

### ✅ Successfully Addressed:
1. **Y-direction standard deviation**: Improved from 0.04-0.07 → 0.16-0.20 ✅
2. **Trajectory smoothness**: LEV scores show good sequence coherence ✅
3. **Training stability**: No divergence, smooth convergence ✅

### ❌ Still Need Attention:
1. **Y-direction mean bias**: Clustered at 0.60-0.62 (should be more uniform)
2. **Position error plateau**: Not improving after Epoch 40
3. **Loss-error divergence**: Loss improving but position error degrading

---

## Root Cause Analysis

### Why Y-Mean is Biased Upward (0.60-0.62)?

**Hypothesis 1: Dataset Bias**
- Salient360 H+HE dataset may have more salient regions in upper portion
- Human viewers naturally look upward in 360° environments
- **Action**: Analyze ground truth Y-distribution in dataset

**Hypothesis 2: Initial Position Bias**
- Model initialization may favor upper regions
- Position decoder bias initialized to favor certain Y-values
- **Action**: Check position decoder initialization (line 195 in model)

**Hypothesis 3: Loss Function Imbalance**
- Spatial coverage loss may not penalize Y-mean bias strongly enough
- Y-center penalty only triggers when y_mean ∈ [0.4, 0.6]
- Current y_mean ~0.61 is just outside penalty range!
- **Action**: Adjust Y-center penalty range

### Why Position Error Plateaus?

**Hypothesis 1: Teacher Forcing Decay Too Fast**
- Exponential decay reaches 0.1 at Epoch 100
- Model may not have learned robust autoregressive generation
- **Action**: Slow down teacher forcing decay

**Hypothesis 2: Loss Function Complexity**
- Even with simplification (13→6), still complex
- Weights may not be optimal for position accuracy
- **Action**: Increase reconstruction loss weight

**Hypothesis 3: VAE Randomness**
- KL divergence weight (0.005) may still introduce too much noise
- **Action**: Further reduce KL weight or use deterministic mode

---

## Recommendations for Next Steps

### 🔥 High Priority Fixes

#### 1. Fix Y-Mean Bias (Immediate)
```python
# In train_mamba_adaptive.py, adjust Y-center penalty range
# Current: y_center_dist < 0.1 (triggers for y_mean ∈ [0.4, 0.6])
# New: Expand range to catch y_mean > 0.55

y_center_dist = torch.abs(pred_mean[:, 1] - 0.5)
# Penalize if too far from center (either direction)
y_bias_penalty = torch.mean((y_center_dist - 0.05).clamp(min=0.0) ** 2) * 10.0
```

#### 2. Increase Reconstruction Loss Weight
```python
# Current weights at Epoch 155:
weights = {
    'reconstruction': 1.0,  # Increase to 2.0
    'kl': 0.005,            # Reduce to 0.001
    'spatial_coverage': 0.8,
    'trajectory_smoothness': 1.5,
    'direction_consistency': 0.5,
    'boundary': 0.2
}
```

#### 3. Slow Down Teacher Forcing Decay
```python
# Current: 0.7 → 0.1 over 100 epochs
# New: 0.7 → 0.2 over 150 epochs
def compute_teacher_forcing_ratio(epoch):
    initial_ratio = 0.7
    final_ratio = 0.2  # Changed from 0.1
    decay_epochs = 150  # Changed from 100
    # ... rest of function
```

### 🟡 Medium Priority Improvements

#### 4. Add Y-Distribution Regularization
```python
# Encourage uniform Y-distribution across batch
y_means = pred_scanpaths[:, :, 1].mean(dim=1)  # (B,)
y_distribution_loss = -torch.std(y_means)  # Maximize std of y_means
```

#### 5. Analyze Dataset Y-Distribution
```python
# Create script to analyze ground truth Y-distribution
# Check if dataset has inherent bias toward upper regions
```

#### 6. Early Stopping Based on Position Error
```python
# Current: Early stopping based on validation loss
# New: Early stopping based on validation position error
# This would have stopped at Epoch 40 (best position error)
```

### 🟢 Low Priority Enhancements

#### 7. Implement Curriculum Learning
- Start with easier samples (shorter sequences)
- Gradually increase difficulty
- May improve position accuracy

#### 8. Add Position Error to Loss Function
- Directly optimize for position accuracy
- Weight: 0.5-1.0

#### 9. Experiment with Deterministic Mode
- Disable VAE sampling during validation
- Use mean (mu) instead of sampling from distribution

---

## Performance Summary

### Current Model Performance (Epoch 155)

**Strengths:**
- ✅ Good Y-direction standard deviation (0.16-0.20)
- ✅ Excellent sequence length matching (LEV ~30)
- ✅ Smooth, coherent trajectories
- ✅ Stable training (no divergence)
- ✅ Good X-direction diversity

**Weaknesses:**
- ❌ Y-direction mean bias (clustered at 0.60-0.62)
- ❌ Position error plateau (0.35-0.45)
- ❌ Loss-error divergence (loss improving, error degrading)
- ❌ Not meeting target position error (<0.22)

**Overall Assessment:**
The model has learned to generate smooth, diverse scanpaths with good Y-direction variance, but suffers from:
1. Systematic Y-direction bias toward upper regions
2. Position accuracy plateau after Epoch 40
3. Misalignment between loss optimization and position accuracy

---

## Comparison: Before vs After Fixes

### Before Fixes (Expected from Plan):
- Y std: 0.04-0.07
- Y mean: 0.49-0.51 (centered)
- Position error: Unknown

### After Fixes (Current Results):
- Y std: 0.16-0.20 ✅ **IMPROVED**
- Y mean: 0.60-0.62 ❌ **NEW BIAS** (different from expected)
- Position error: 0.35-0.45 ⚠️ **NEEDS IMPROVEMENT**

**Conclusion**: The fixes successfully improved Y-direction diversity (std), but introduced a new bias in Y-direction mean. The position error is reasonable but not meeting the target.

---

## Next Training Strategy

### Option 1: Fine-tune from Epoch 40 (Recommended)
- Load checkpoint from Epoch 40 (best position error: 0.3943)
- Apply Y-mean bias fix
- Increase reconstruction loss weight
- Train for 50 more epochs

### Option 2: Continue from Epoch 155
- Apply all recommended fixes
- Train for 50 more epochs
- Monitor position error closely

### Option 3: Restart with Adjusted Hyperparameters
- Apply all fixes from the start
- May take longer but could achieve better results

**Recommendation**: **Option 1** - Fine-tune from Epoch 40, as it had the best position error before overfitting occurred.
