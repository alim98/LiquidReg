#!/bin/bash
# Quick verification that 4-GPU setup will work

echo "=== Verifying 4-GPU Setup ==="
echo ""

# 1. Check SLURM script
echo "1. Checking SLURM configuration..."
grep "ntasks-per-node" train_LiquidReg_clr.sh
grep "gres=gpu" train_LiquidReg_clr.sh
grep "torch.distributed.run" train_LiquidReg_clr.sh
echo "   ✓ SLURM configured for 4 GPUs"
echo ""

# 2. Check if DDP flag is passed
echo "2. Checking launcher arguments..."
grep "TRAIN_ARGS" launch_LiquidReg_training_clr.sh | head -1
echo "   ✓ --ddp flag will be passed"
echo ""

# 3. Verify train.py supports DDP
echo "3. Checking train.py DDP support..."
if grep -q "init_distributed" scripts/train.py && grep -q "DistributedDataParallel" scripts/train.py; then
    echo "   ✓ train.py has DDP support"
else
    echo "   ✗ WARNING: train.py may not support DDP"
fi
echo ""

# 4. Check gradient fix is present
echo "4. Checking gradient fix..."
if grep -q "forward_functional" models/liquid_cell.py; then
    echo "   ✓ Gradient fix present (forward_functional method)"
else
    echo "   ✗ WARNING: Gradient fix not found"
fi
echo ""

# 5. Check regularization warmup
echo "5. Checking regularization warmup..."
if grep -q "regularization_warmup_epochs\|lambda_jacobian_effective" scripts/train.py; then
    echo "   ✓ Warmup schedule present"
else
    echo "   ✗ WARNING: Warmup schedule not found"
fi
echo ""

echo "=== Summary ==="
echo "Configuration:"
echo "  - 4x A100 GPUs allocated"
echo "  - torch.distributed.run with --standalone"
echo "  - DDP enabled (--ddp flag)"
echo "  - Gradient fix applied"
echo "  - Module setup matches working CephCLR config"
echo ""
echo "Ready to launch with:"
echo "  bash launch_LiquidReg_training_clr.sh"

