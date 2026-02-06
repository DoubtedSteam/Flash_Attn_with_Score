"""
Benchmark script for Flash Attention with Scores
Compares Flash Attention, PyTorch SDPA, and naive implementations
"""

import sys
import os
import torch
import argparse
import contextlib
from typing import Dict, List, Tuple, Optional

# Add current directory to path so modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    import triton
    from triton.testing import do_bench
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    raise ImportError("Triton is not available. Please install: pip install triton")

# Import directly from modules in the same directory
from attention_with_scores import attention_with_scores
from attention_with_row_sum import attention_with_row_sum
from attention_with_col_sum import attention_with_col_sum
from attention_with_col_sum_sequential import attention_with_col_sum_sequential
from attention_with_col_softmax_sum import attention_with_softmax_col_sum
from attention_cross_token_softmax_sum import attention_cross_token_softmax_sum, attention_cross_token_softmax_sum_buffered
from attention_cross_token_qk_sum import attention_cross_token_qk_sum
from reference_implementations import naive_attention_with_scores, pytorch_sdpa_attention_with_scores


def generate_config_space(
    B: int, H: int, M: int, N: int, D: int, causal: bool
) -> List[Tuple[int, int, int, int]]:
    """
    Generate configuration search space
    
    Returns:
        List of (BLOCK_M, BLOCK_N, num_stages, num_warps) tuples
    """
    configs = []
    
    # BLOCK_M candidates
    block_m_candidates = [32, 64, 128]
    if M >= 2048:
        block_m_candidates.append(256)
    
    # BLOCK_N candidates
    block_n_candidates = [32, 64, 128]
    if N >= 2048:
        block_n_candidates.append(256)
    
    # num_stages candidates (usually 1-5)
    num_stages_candidates = [1, 2, 3, 4, 5]
    
    # num_warps candidates (usually 1, 2, 4, 8, 16)
    num_warps_candidates = [1, 2, 4, 8]
    if D >= 128:
        num_warps_candidates.append(16)
    
    # Generate all combinations
    for block_m in block_m_candidates:
        if block_m > M:
            continue
        for block_n in block_n_candidates:
            if block_n > N:
                continue
            for num_stages in num_stages_candidates:
                for num_warps in num_warps_candidates:
                    configs.append((block_m, block_n, num_stages, num_warps))
    
    return configs


@contextlib.contextmanager
def override_get_fwd_config(config: Tuple[int, int, int, int]):
    """
    Temporarily override get_fwd_config function to use specified configuration
    Since all modules import get_fwd_config from flash module, we only need to modify flash module
    """
    import flash as flash_module
    
    # Save original function
    original_get_fwd_config = flash_module.get_fwd_config
    
    # Create new function
    def new_get_fwd_config(*args, **kwargs):
        return config
    
    # Temporarily replace
    flash_module.get_fwd_config = new_get_fwd_config
    
    try:
        yield
    finally:
        # Restore original function
        flash_module.get_fwd_config = original_get_fwd_config


def pad_sequence_to_multiple(seq_len: int, block_size: int) -> int:
    """
    Pad sequence length to be divisible by block_size

    Args:
        seq_len: Original sequence length
        block_size: Block size (typically 64 or 128)

    Returns:
        Padded sequence length that is divisible by block_size
    """
    if seq_len % block_size == 0:
        return seq_len
    return ((seq_len + block_size - 1) // block_size) * block_size


def search_best_config(
    implementation: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    config_space: Optional[List[Tuple[int, int, int, int]]] = None,
    num_trials: int = 3,
    warmup: int = 5,
    timeout_ms: float = 1000.0,
    verbose: bool = True,
) -> Tuple[Tuple[int, int, int, int], float]:
    """
    Search for optimal kernel configuration
    
    Args:
        implementation: Implementation type ("flash", "row_sum", "col_sum", "col_sum_sequential")
        q, k, v: Input tensors
        causal: Whether to use causal mask
        dropout_p: Dropout probability
        config_space: Configuration search space, if None then auto-generate
        num_trials: Number of trials per configuration
        warmup: Warmup iterations
        timeout_ms: Timeout for single configuration (ms)
        verbose: Whether to print detailed information
    
    Returns:
        (best_config, best_time_ms): Optimal configuration and execution time
    """
    B, H, M, D = q.shape
    N = k.shape[2]
    
    if config_space is None:
        config_space = generate_config_space(B, H, M, N, D, causal)
    
    if verbose:
        print(f"  Search space size: {len(config_space)} configurations")
    
    # Create test function
    def create_test_fn(config):
        def test_fn():
            with override_get_fwd_config(config):
                if implementation == "flash":
                    return attention_with_scores(q, k, v, causal=causal, dropout_p=dropout_p)
                elif implementation == "row_sum":
                    return attention_with_row_sum(q, k, v, causal=causal, dropout_p=dropout_p)
                elif implementation == "col_sum":
                    return attention_with_col_sum(q, k, v, causal=causal, dropout_p=dropout_p)
                elif implementation == "col_sum_sequential":
                    return attention_with_col_sum_sequential(q, k, v, causal=causal, dropout_p=dropout_p)
                elif implementation == "cross_token_qk":
                    split = getattr(create_test_fn, 'split', 768)  # Default split value
                    return attention_cross_token_qk_sum(q, k, v, split=split, causal=causal, dropout_p=dropout_p)
                else:
                    raise ValueError(f"Unsupported implementation type: {implementation}")
        return test_fn
    
    best_config = None
    best_time = float('inf')
    valid_configs = []
    
    for idx, config in enumerate(config_space):
        BLOCK_M, BLOCK_N, num_stages, num_warps = config
        
        # Check if configuration is reasonable
        if BLOCK_M > M or BLOCK_N > N:
            continue
        
        try:
            test_fn = create_test_fn(config)
            
            # Warmup
            for _ in range(warmup):
                try:
                    _ = test_fn()
                    torch.cuda.synchronize()
                except Exception:
                    break
            
            # Test performance
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            times = []
            for trial in range(num_trials):
                start_event.record()
                result = test_fn()
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_event.elapsed_time(end_event)
                if elapsed_ms < timeout_ms:
                    times.append(elapsed_ms)
            
            if len(times) > 0:
                avg_time = sum(times) / len(times)
                valid_configs.append((config, avg_time))
                
                if avg_time < best_time:
                    best_time = avg_time
                    best_config = config
                
                if verbose and (idx + 1) % 10 == 0:
                    print(f"  Tested {idx + 1}/{len(config_space)} configurations, current best: {best_time:.3f}ms (BLOCK_M={best_config[0]}, BLOCK_N={best_config[1]}, stages={best_config[2]}, warps={best_config[3]})")
        
        except Exception as e:
            if verbose and idx < 5:  # Only print first few errors
                print(f"  Configuration {config} failed: {str(e)[:100]}")
            continue
    
    if best_config is None:
        raise RuntimeError("No valid configuration found")
    
    if verbose:
        print(f"  Found {len(valid_configs)} valid configurations")
        print(f"  Optimal configuration: BLOCK_M={best_config[0]}, BLOCK_N={best_config[1]}, "
              f"num_stages={best_config[2]}, num_warps={best_config[3]}, "
              f"time={best_time:.3f}ms")
    
    return best_config, best_time


def benchmark_attention_with_scores(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    causal: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.float16,
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
    implementation: str = "flash",
    enable_config_search: bool = False,
    config_search_trials: int = 3,
    config_search_warmup: int = 5,
    custom_config: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, float]:
    """
    Benchmark attention_with_scores implementations
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    try:
        # Create input tensors with padded dimensions
        q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Configuration search
        best_config = None
        if enable_config_search and implementation == "flash":
            print(f"  Searching optimal configuration for {implementation}...")
            import math
            sm_scale = 1.0 / math.sqrt(head_dim)
            try:
                best_config, best_time = search_best_config(
                    implementation=implementation,
                    q=q,
                    k=k,
                    v=v,
                    causal=causal,
                    dropout_p=dropout_p,
                    config_space=None,
                    num_trials=config_search_trials,
                    warmup=config_search_warmup,
                    timeout_ms=1000.0,
                    verbose=True,
                )
                print(f"  Configuration search completed, optimal config: BLOCK_M={best_config[0]}, BLOCK_N={best_config[1]}, num_stages={best_config[2]}, num_warps={best_config[3]}")
            except Exception as e:
                print(f"  Configuration search failed: {e}, using default configuration")
                import traceback
                traceback.print_exc()
                best_config = None
        
        # Use custom config if provided
        if custom_config is not None:
            best_config = custom_config
        
        # Select implementation
        if implementation == "flash":
            if best_config is not None:
                def fn():
                    with override_get_fwd_config(best_config):
                        return attention_with_scores(q, k, v, causal=causal, dropout_p=dropout_p)
            else:
                def fn():
                    return attention_with_scores(q, k, v, causal=causal, dropout_p=dropout_p)
        elif implementation == "naive":
            def fn():
                return naive_attention_with_scores(q, k, v, causal=causal, dropout_p=dropout_p)
        elif implementation == "pytorch_sdpa":
            def fn():
                return pytorch_sdpa_attention_with_scores(q, k, v, causal=causal, dropout_p=dropout_p)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        all_run_times = []
        total_runs_to_perform = num_runs + 2
        
        for run_idx in range(total_runs_to_perform):
            # Clear cache before each run
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # JIT compilation warmup for Flash Attention
            if implementation == "flash":
                for _ in range(5):
                    _ = fn()
                torch.cuda.synchronize()
            
            run_ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
            if run_ms != float("inf") and run_ms > 0:
                all_run_times.append(run_ms)
        
        # Use last num_runs results for statistics
        if len(all_run_times) >= num_runs:
            times_for_calculation = all_run_times[-num_runs:]
        else:
            times_for_calculation = all_run_times
        
        if len(times_for_calculation) == 0:
            ms = float("inf")
            avg_ms = float("inf")
            min_ms = float("inf")
            max_ms = float("inf")
        else:
            avg_ms = sum(times_for_calculation) / len(times_for_calculation)
            sorted_times = sorted(times_for_calculation)
            min_ms = sorted_times[0]
            max_ms = sorted_times[-1]
            median_ms = sorted_times[len(sorted_times) // 2]
            ms = median_ms
        
        # Calculate throughput
        total_tokens = batch_size * num_heads_q * seq_len_q
        throughput = total_tokens / (ms / 1000) if ms != float("inf") else 0.0
        
        # Calculate TFLOPs
        # Attention: 2 * B * H_q * M * N * D (Q @ K^T) + 2 * B * H_q * M * N * D (attn @ V)
        # Score: 2 * B * H_q * M * N * D (Q @ K^T)
        # Total: 4 * B * H_q * M * N * D
        flops = 4 * batch_size * num_heads_q * seq_len_q * seq_len_k * head_dim
        tflops = (flops / (ms / 1000)) / 1e12 if ms != float("inf") else 0.0
        
        return {
            "avg_time_ms": avg_ms,
            "median_time_ms": ms,
            "min_time_ms": min_ms,
            "max_time_ms": max_ms,
            "throughput_tokens_per_sec": throughput,
            "tflops": tflops,
        }
    
    except torch.cuda.OutOfMemoryError:
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }


def benchmark_row_sum(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    causal: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.float16,
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark attention_with_row_sum implementation
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        def fn():
            return attention_with_row_sum(q, k, v, causal=causal, dropout_p=dropout_p)
        
        all_run_times = []
        total_runs_to_perform = num_runs + 2
        
        for run_idx in range(total_runs_to_perform):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # JIT compilation warmup
            for _ in range(5):
                _ = fn()
            torch.cuda.synchronize()
            
            run_ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
            if run_ms != float("inf") and run_ms > 0:
                all_run_times.append(run_ms)
        
        if len(all_run_times) >= num_runs:
            times_for_calculation = all_run_times[-num_runs:]
        else:
            times_for_calculation = all_run_times
        
        if len(times_for_calculation) == 0:
            ms = float("inf")
            avg_ms = float("inf")
            min_ms = float("inf")
            max_ms = float("inf")
        else:
            avg_ms = sum(times_for_calculation) / len(times_for_calculation)
            sorted_times = sorted(times_for_calculation)
            min_ms = sorted_times[0]
            max_ms = sorted_times[-1]
            median_ms = sorted_times[len(sorted_times) // 2]
            ms = median_ms
        
        total_tokens = batch_size * num_heads_q * seq_len_q
        throughput = total_tokens / (ms / 1000) if ms != float("inf") else 0.0
        
        # FLOPS: same as attention_with_scores
        flops = 4 * batch_size * num_heads_q * seq_len_q * seq_len_k * head_dim
        tflops = (flops / (ms / 1000)) / 1e12 if ms != float("inf") else 0.0
        
        return {
            "avg_time_ms": avg_ms,
            "median_time_ms": ms,
            "min_time_ms": min_ms,
            "max_time_ms": max_ms,
            "throughput_tokens_per_sec": throughput,
            "tflops": tflops,
        }
    
    except torch.cuda.OutOfMemoryError:
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }


def benchmark_col_sum(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    causal: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.float16,
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark attention_with_col_sum implementation
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        def fn():
            return attention_with_col_sum(q, k, v, causal=causal, dropout_p=dropout_p)
        
        all_run_times = []
        total_runs_to_perform = num_runs + 2
        
        for run_idx in range(total_runs_to_perform):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # JIT compilation warmup
            for _ in range(5):
                _ = fn()
            torch.cuda.synchronize()
            
            run_ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
            if run_ms != float("inf") and run_ms > 0:
                all_run_times.append(run_ms)
        
        if len(all_run_times) >= num_runs:
            times_for_calculation = all_run_times[-num_runs:]
        else:
            times_for_calculation = all_run_times
        
        if len(times_for_calculation) == 0:
            ms = float("inf")
            avg_ms = float("inf")
            min_ms = float("inf")
            max_ms = float("inf")
        else:
            avg_ms = sum(times_for_calculation) / len(times_for_calculation)
            sorted_times = sorted(times_for_calculation)
            min_ms = sorted_times[0]
            max_ms = sorted_times[-1]
            median_ms = sorted_times[len(sorted_times) // 2]
            ms = median_ms
        
        total_tokens = batch_size * num_heads_q * seq_len_q
        throughput = total_tokens / (ms / 1000) if ms != float("inf") else 0.0
        
        # FLOPS: same as attention_with_scores
        flops = 4 * batch_size * num_heads_q * seq_len_q * seq_len_k * head_dim
        tflops = (flops / (ms / 1000)) / 1e12 if ms != float("inf") else 0.0
        
        return {
            "avg_time_ms": avg_ms,
            "median_time_ms": ms,
            "min_time_ms": min_ms,
            "max_time_ms": max_ms,
            "throughput_tokens_per_sec": throughput,
            "tflops": tflops,
        }
    
    except torch.cuda.OutOfMemoryError:
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }


def benchmark_col_sum_sequential(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    causal: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.float16,
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark attention_with_col_sum_sequential implementation
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        def fn():
            return attention_with_col_sum_sequential(q, k, v, causal=causal, dropout_p=dropout_p)
        
        all_run_times = []
        total_runs_to_perform = num_runs + 2
        
        for run_idx in range(total_runs_to_perform):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # JIT compilation warmup
            for _ in range(5):
                _ = fn()
            torch.cuda.synchronize()
            
            run_ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
            if run_ms != float("inf") and run_ms > 0:
                all_run_times.append(run_ms)
        
        if len(all_run_times) >= num_runs:
            times_for_calculation = all_run_times[-num_runs:]
        else:
            times_for_calculation = all_run_times
        
        if len(times_for_calculation) == 0:
            ms = float("inf")
            avg_ms = float("inf")
            min_ms = float("inf")
            max_ms = float("inf")
        else:
            avg_ms = sum(times_for_calculation) / len(times_for_calculation)
            sorted_times = sorted(times_for_calculation)
            min_ms = sorted_times[0]
            max_ms = sorted_times[-1]
            median_ms = sorted_times[len(sorted_times) // 2]
            ms = median_ms
        
        total_tokens = batch_size * num_heads_q * seq_len_q
        throughput = total_tokens / (ms / 1000) if ms != float("inf") else 0.0
        
        # FLOPS: same as attention_with_scores
        flops = 4 * batch_size * num_heads_q * seq_len_q * seq_len_k * head_dim
        tflops = (flops / (ms / 1000)) / 1e12 if ms != float("inf") else 0.0
        
        return {
            "avg_time_ms": avg_ms,
            "median_time_ms": ms,
            "min_time_ms": min_ms,
            "max_time_ms": max_ms,
            "throughput_tokens_per_sec": throughput,
            "tflops": tflops,
        }
    
    except torch.cuda.OutOfMemoryError:
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }


def benchmark_col_softmax_sum(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    causal: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.float16,
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark attention_with_softmax_col_sum implementation
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        def fn():
            return attention_with_softmax_col_sum(q, k, v, causal=causal, dropout_p=dropout_p)

        all_run_times = []
        total_runs_to_perform = num_runs + 2

        for run_idx in range(total_runs_to_perform):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # JIT compilation warmup
            for _ in range(5):
                _ = fn()
            torch.cuda.synchronize()

            run_ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
            if run_ms != float("inf") and run_ms > 0:
                all_run_times.append(run_ms)

        if len(all_run_times) >= num_runs:
            times_for_calculation = all_run_times[-num_runs:]
        else:
            times_for_calculation = all_run_times

        if len(times_for_calculation) == 0:
            ms = float("inf")
            avg_ms = float("inf")
            min_ms = float("inf")
            max_ms = float("inf")
        else:
            avg_ms = sum(times_for_calculation) / len(times_for_calculation)
            sorted_times = sorted(times_for_calculation)
            min_ms = sorted_times[0]
            max_ms = sorted_times[-1]
            median_ms = sorted_times[len(sorted_times) // 2]
            ms = median_ms

        total_tokens = batch_size * num_heads_q * seq_len_q
        throughput = total_tokens / (ms / 1000) if ms != float("inf") else 0.0

        # FLOPS: same as attention_with_scores
        flops = 4 * batch_size * num_heads_q * seq_len_q * seq_len_k * head_dim
        tflops = (flops / (ms / 1000)) / 1e12 if ms != float("inf") else 0.0

        return {
            "avg_time_ms": avg_ms,
            "median_time_ms": ms,
            "min_time_ms": min_ms,
            "max_time_ms": max_ms,
            "throughput_tokens_per_sec": throughput,
            "tflops": tflops,
        }

    except torch.cuda.OutOfMemoryError:
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }


def benchmark_cross_token_softmax(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    split: int,
    causal: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.float16,
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark attention_cross_token_softmax_sum implementation
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        def fn():
            return attention_cross_token_softmax_sum(q, k, v, split=split, causal=causal, dropout_p=dropout_p)
        
        all_run_times = []
        total_runs_to_perform = num_runs + 2
        
        for run_idx in range(total_runs_to_perform):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # JIT compilation warmup
            for _ in range(5):
                _ = fn()
            torch.cuda.synchronize()
            
            run_ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
            if run_ms != float("inf") and run_ms > 0:
                all_run_times.append(run_ms)
        
        if len(all_run_times) >= num_runs:
            # Use last num_runs results for statistics
            run_times = sorted(all_run_times[-num_runs:])
        else:
            run_times = sorted(all_run_times) if all_run_times else [float("inf")]
        
        median_time_ms = run_times[len(run_times) // 2] if run_times else float("inf")
        avg_time_ms = sum(run_times) / len(run_times) if run_times else float("inf")
        min_time_ms = run_times[0] if run_times else float("inf")
        max_time_ms = run_times[-1] if run_times else float("inf")
        
        total_tokens = batch_size * num_heads_q * seq_len_q
        throughput = total_tokens / (median_time_ms / 1000) if median_time_ms != float("inf") else 0.0
        
        # Calculate TFLOPs (only matmul operations)
        flops = 4 * batch_size * num_heads_q * seq_len_q * seq_len_k * head_dim
        tflops = (flops / (median_time_ms / 1000)) / 1e12 if median_time_ms != float("inf") else 0.0
        
        return {
            "avg_time_ms": avg_time_ms,
            "median_time_ms": median_time_ms,
            "min_time_ms": min_time_ms,
            "max_time_ms": max_time_ms,
            "throughput_tokens_per_sec": throughput,
            "tflops": tflops,
        }
    
    except torch.cuda.OutOfMemoryError:
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }




def benchmark_cross_token_softmax_buffered(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    split: int,
    causal: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.float16,
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark attention_cross_token_softmax_sum_buffered implementation (with QK^T buffering)
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        def fn():
            return attention_cross_token_softmax_sum_buffered(q, k, v, split=split, causal=causal, dropout_p=dropout_p)

        all_run_times = []
        total_runs_to_perform = num_runs + 2

        for run_idx in range(total_runs_to_perform):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # JIT compilation warmup
            for _ in range(5):
                _ = fn()
            torch.cuda.synchronize()

            run_ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
            if run_ms != float("inf") and run_ms > 0:
                all_run_times.append(run_ms)

        if len(all_run_times) >= num_runs:
            # Use last num_runs results for statistics
            run_times = sorted(all_run_times[-num_runs:])
        else:
            run_times = sorted(all_run_times) if all_run_times else [float("inf")]

        median_time_ms = run_times[len(run_times) // 2] if run_times else float("inf")
        avg_time_ms = sum(run_times) / len(run_times) if run_times else float("inf")
        min_time_ms = run_times[0] if run_times else float("inf")
        max_time_ms = run_times[-1] if run_times else float("inf")

        total_tokens = batch_size * num_heads_q * seq_len_q
        throughput = total_tokens / (median_time_ms / 1000) if median_time_ms != float("inf") else 0.0

        # Calculate TFLOPs (only matmul operations)
        flops = 4 * batch_size * num_heads_q * seq_len_q * seq_len_k * head_dim
        tflops = (flops / (median_time_ms / 1000)) / 1e12 if median_time_ms != float("inf") else 0.0

        return {
            "avg_time_ms": avg_time_ms,
            "median_time_ms": median_time_ms,
            "min_time_ms": min_time_ms,
            "max_time_ms": max_time_ms,
            "throughput_tokens_per_sec": throughput,
            "tflops": tflops,
        }

    except torch.cuda.OutOfMemoryError:
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }
    except Exception as e:
        print(f"Error in benchmark_cross_token_softmax_buffered: {e}")
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }


def benchmark_cross_token_qk(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    split: int = 768,
    causal: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.float16,
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark attention_cross_token_qk_sum implementation
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        def fn():
            return attention_cross_token_qk_sum(q, k, v, split=split, causal=causal, dropout_p=dropout_p)
        
        all_run_times = []
        total_runs_to_perform = num_runs + 2
        
        for run_idx in range(total_runs_to_perform):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # JIT compilation warmup
            for _ in range(5):
                _ = fn()
            torch.cuda.synchronize()
            
            run_ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
            if run_ms != float("inf") and run_ms > 0:
                all_run_times.append(run_ms)
        
        if len(all_run_times) >= num_runs:
            times_for_calculation = all_run_times[-num_runs:]
        else:
            times_for_calculation = all_run_times
        
        if len(times_for_calculation) == 0:
            ms = float("inf")
            avg_ms = float("inf")
            min_ms = float("inf")
            max_ms = float("inf")
        else:
            avg_ms = sum(times_for_calculation) / len(times_for_calculation)
            sorted_times = sorted(times_for_calculation)
            min_ms = sorted_times[0]
            max_ms = sorted_times[-1]
            median_ms = sorted_times[len(sorted_times) // 2]
            ms = median_ms
        
        total_tokens = batch_size * num_heads_q * seq_len_q
        throughput = total_tokens / (ms / 1000) if ms != float("inf") else 0.0
        
        # FLOPS: same as attention_with_scores
        flops = 4 * batch_size * num_heads_q * seq_len_q * seq_len_k * head_dim
        tflops = (flops / (ms / 1000)) / 1e12 if ms != float("inf") else 0.0
        
        return {
            "avg_time_ms": avg_ms,
            "median_time_ms": ms,
            "min_time_ms": min_ms,
            "max_time_ms": max_ms,
            "throughput_tokens_per_sec": throughput,
            "tflops": tflops,
        }
    
    except torch.cuda.OutOfMemoryError:
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "avg_time_ms": float("inf"),
            "median_time_ms": float("inf"),
            "min_time_ms": float("inf"),
            "max_time_ms": float("inf"),
            "throughput_tokens_per_sec": 0.0,
            "tflops": 0.0,
        }


def _safe_isfinite_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Create a mask for finite values that works with large tensors (> INT_MAX elements).
    Uses separate checks for nan and inf to avoid the nonzero() limitation.
    """
    return ~torch.isnan(x) & ~torch.isinf(x)


def _safe_finite_stats(x: torch.Tensor, mask: torch.Tensor = None) -> tuple[float, float, int]:
    """
    Compute max, mean, and count of finite values in a tensor without using boolean indexing.
    This avoids the nonzero() limitation for large tensors (> INT_MAX elements).
    
    Returns:
        (max_value, mean_value, count)
    """
    if mask is None:
        mask = _safe_isfinite_mask(x)
    
    # First, replace inf and nan values with 0 to ensure sum is finite
    # Use torch.where to replace non-finite values with -inf for max, 0 for sum
    # This avoids boolean indexing which triggers nonzero()
    x_clean = torch.where(torch.isfinite(x), x, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    x_max = torch.where(mask, x_clean, torch.tensor(float('-inf'), device=x.device, dtype=x.dtype))
    x_sum = torch.where(mask, x_clean, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    
    # Count finite values using sum of mask (converted to int)
    count = mask.sum().item()
    
    if count > 0:
        max_val = x_max.max().item()
        sum_val = x_sum.sum().item()
        # Check if sum is finite before dividing
        if torch.isfinite(torch.tensor(sum_val)):
            mean_val = sum_val / count
        else:
            # If sum is inf, use a large but finite value
            mean_val = 1e10
        return max_val, mean_val, count
    else:
        return 0.0, 0.0, 0


def verify_correctness(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    causal: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    tolerance: float = 1e-2,
    split: int = 768,
) -> Dict[str, Dict[str, float]]:
    """
    Verify correctness of all implementations and return error metrics
    
    Returns:
        Dictionary with error metrics for each implementation:
        {
            'flash_with_scores': {
                'output_abs_error': float,
                'output_rel_error': float,
                'scores_abs_error': float,
                'scores_rel_error': float,
            },
            ...
        }
    """
    print(f"  Verifying correctness...")

    # Pad sequence lengths to be divisible by typical block size (64)
    BLOCK_SIZE = 64
    seq_len_q_padded = pad_sequence_to_multiple(seq_len_q, BLOCK_SIZE)
    seq_len_k_padded = pad_sequence_to_multiple(seq_len_k, BLOCK_SIZE)

    # Print padding info if sequences were padded
    if seq_len_q_padded != seq_len_q or seq_len_k_padded != seq_len_k:
        print(f"  Padding for verification: seq_q {seq_len_q} -> {seq_len_q_padded}, seq_k {seq_len_k} -> {seq_len_k_padded}")

    # Create input tensors with fixed seed for reproducibility
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads_q, seq_len_q_padded, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads_k, seq_len_k_padded, head_dim, dtype=dtype, device=device)
    
    results = {}
    errors = {}
    
    # 1. Naive implementation (reference)
    try:
        torch.manual_seed(42)
        q_ref = q.clone()
        k_ref = k.clone()
        v_ref = v.clone()
        output_naive, scores_naive = naive_attention_with_scores(q_ref, k_ref, v_ref, causal=causal)
        results['naive'] = {'output': output_naive, 'scores': scores_naive}
    except Exception as e:
        print(f"    ✗ Naive implementation failed: {e}")
        return {}
    
    # 2. Flash Attention with Scores
    try:
        torch.manual_seed(42)
        q_flash = q.clone()
        k_flash = k.clone()
        v_flash = v.clone()
        output_flash, scores_flash = attention_with_scores(q_flash, k_flash, v_flash, causal=causal)
        results['flash_with_scores'] = {'output': output_flash, 'scores': scores_flash}
    except Exception as e:
        print(f"    ✗ + Scores failed: {e}")
    
    # 3. Flash Attention with Row Sum
    try:
        torch.manual_seed(42)
        q_row = q.clone()
        k_row = k.clone()
        v_row = v.clone()
        output_row, row_sum = attention_with_row_sum(q_row, k_row, v_row, causal=causal)
        results['flash_row_sum'] = {'output': output_row, 'row_sum': row_sum}
    except Exception as e:
        print(f"    ✗ + Row Sum failed: {e}")
    
    # 4. Flash Attention with Col Sum
    try:
        torch.manual_seed(42)
        q_col = q.clone()
        k_col = k.clone()
        v_col = v.clone()
        output_col, col_sum = attention_with_col_sum(q_col, k_col, v_col, causal=causal)
        results['flash_col_sum'] = {'output': output_col, 'col_sum': col_sum}
    except Exception as e:
        print(f"    ✗ + Col Sum failed: {e}")
    
    # 5. Flash Attention with Col Sum Sequential
    try:
        torch.manual_seed(42)
        q_col_seq = q.clone()
        k_col_seq = k.clone()
        v_col_seq = v.clone()
        output_col_seq, col_sum_seq = attention_with_col_sum_sequential(q_col_seq, k_col_seq, v_col_seq, causal=causal)
        results['flash_col_sum_sequential'] = {'output': output_col_seq, 'col_sum': col_sum_seq}
    except Exception as e:
        print(f"    ✗ + Col Sum Sequential failed: {e}")

    # 5b. Flash Attention with Col Softmax Sum
    try:
        torch.manual_seed(42)
        q_col_softmax = q.clone()
        k_col_softmax = k.clone()
        v_col_softmax = v.clone()
        output_col_softmax, col_softmax_sum = attention_with_softmax_col_sum(q_col_softmax, k_col_softmax, v_col_softmax, causal=causal)
        results['flash_col_softmax_sum'] = {'output': output_col_softmax, 'col_softmax_sum': col_softmax_sum}
    except Exception as e:
        print(f"    ✗ + Col Softmax Sum failed: {e}")

    # 6. Flash Attention with Col Sum Softmax
    try:
        torch.manual_seed(42)
        q_col_softmax = q.clone()
        k_col_softmax = k.clone()
        v_col_softmax = v.clone()
        output_col_softmax, cross_token_softmax = attention_cross_token_softmax_sum(q_col_softmax, k_col_softmax, v_col_softmax, split=split, causal=causal)
        results['flash_cross_token_softmax'] = {'output': output_col_softmax, 'cross_token_softmax': cross_token_softmax}
    except Exception as e:
        print(f"    ✗ + Cross-Token Softmax Sum failed: {e}")

    # 6b. Flash Attention with Col Sum Softmax (Buffered)
    try:
        torch.manual_seed(42)
        q_col_softmax_buf = q.clone()
        k_col_softmax_buf = k.clone()
        v_col_softmax_buf = v.clone()
        output_col_softmax_buf, cross_token_softmax_buf = attention_cross_token_softmax_sum_buffered(q_col_softmax_buf, k_col_softmax_buf, v_col_softmax_buf, split=split, causal=causal)
        results['flash_cross_token_softmax_buffered'] = {'output': output_col_softmax_buf, 'cross_token_softmax': cross_token_softmax_buf}
    except Exception as e:
        print(f"    ✗ + Cross-Token Softmax Sum (Buffered) failed: {e}")

    # 7. Flash Attention with Split QK Sum
    try:
        if split < seq_len_k and split < seq_len_q:
            torch.manual_seed(42)
            q_split = q.clone()
            k_split = k.clone()
            v_split = v.clone()
            output_split, cross_token_qk = attention_cross_token_qk_sum(q_split, k_split, v_split, split=split, causal=causal)
            results['flash_cross_token_qk'] = {'output': output_split, 'cross_token_qk': cross_token_qk}
    except Exception as e:
        print(f"    ✗ + Cross-Token QK Sum failed: {e}")
    
    # Calculate errors for each implementation
    reference_output = results['naive']['output']
    reference_scores = results['naive']['scores']
    
    for impl_name, impl_result in results.items():
        if impl_name == 'naive':
            continue
        
        impl_output = impl_result['output']
        
        # Calculate output errors
        output_diff = (impl_output - reference_output).abs()
        
        # Filter out inf and nan values (use safe method for large tensors)
        valid_mask = _safe_isfinite_mask(output_diff) & _safe_isfinite_mask(reference_output)
        count = valid_mask.sum().item()
        if count > 0:
            output_abs_error, output_mean_abs_error, _ = _safe_finite_stats(output_diff, valid_mask)
            
            # Calculate relative error (only for non-zero reference values)
            ref_abs = reference_output.abs()
            # Use a larger epsilon for very small values to avoid inf
            ref_abs_safe = torch.clamp(ref_abs, min=1e-6)
            rel_error_tensor = output_diff / ref_abs_safe
            # Replace any inf/nan values that might have been created before clamp
            rel_error_tensor = torch.where(torch.isfinite(rel_error_tensor), rel_error_tensor, torch.tensor(1e10, device=rel_error_tensor.device, dtype=rel_error_tensor.dtype))
            # Clamp relative error to avoid inf (max relative error = 1e10)
            rel_error_tensor = torch.clamp(rel_error_tensor, max=1e10)
            rel_error_mask = valid_mask & (ref_abs > 1e-6) & _safe_isfinite_mask(rel_error_tensor)
            rel_error_count = rel_error_mask.sum().item()
            
            if rel_error_count > 0:
                output_rel_error, output_mean_rel_error, _ = _safe_finite_stats(rel_error_tensor, rel_error_mask)
            else:
                # If all reference values are too small, use absolute error
                output_rel_error = min(output_abs_error, 1e10)
                output_mean_rel_error = min(output_mean_abs_error, 1e10)
        else:
            # All values are inf or nan
            output_abs_error = float('inf')
            output_mean_abs_error = float('inf')
            output_rel_error = 1e10  # Clamp to max relative error
            output_mean_rel_error = 1e10
        
        errors[impl_name] = {
            'output_max_abs_error': output_abs_error,
            'output_mean_abs_error': output_mean_abs_error,
            'output_max_rel_error': output_rel_error,
            'output_mean_rel_error': output_mean_rel_error,
        }
        
        # Calculate scores errors if available
        if 'scores' in impl_result:
            impl_scores = impl_result['scores']
            
            # Handle causal mask differences (naive uses -inf, flash uses 0)
            if causal:
                inf_mask_ref = torch.isinf(reference_scores) & (reference_scores < 0)
                zero_mask_flash = (impl_scores == 0.0)
                valid_mask = ~inf_mask_ref & ~zero_mask_flash
                
                if valid_mask.sum() > 0:
                    scores_diff_masked = torch.where(
                        valid_mask,
                        (impl_scores - reference_scores).abs(),
                        torch.zeros_like(impl_scores)
                    )
                    # Filter out inf and nan (use safe method for large tensors)
                    finite_mask = _safe_isfinite_mask(scores_diff_masked)
                    count = finite_mask.sum().item()
                    if count > 0:
                        scores_abs_error, scores_mean_abs_error, _ = _safe_finite_stats(scores_diff_masked, finite_mask)
                    else:
                        scores_abs_error = 0.0
                        scores_mean_abs_error = 0.0
                    
                    # Calculate relative error (only for non-zero reference values)
                    ref_scores_abs = reference_scores.abs()
                    ref_scores_abs_safe = torch.clamp(ref_scores_abs, min=1e-6)
                    rel_error_tensor = scores_diff_masked / ref_scores_abs_safe
                    # Replace any inf/nan values that might have been created before clamp
                    rel_error_tensor = torch.where(torch.isfinite(rel_error_tensor), rel_error_tensor, torch.tensor(1e10, device=rel_error_tensor.device, dtype=rel_error_tensor.dtype))
                    # Clamp relative error to avoid inf (max relative error = 1e10)
                    rel_error_tensor = torch.clamp(rel_error_tensor, max=1e10)
                    rel_error_mask = valid_mask & (ref_scores_abs > 1e-6) & _safe_isfinite_mask(rel_error_tensor)
                    rel_error_count = rel_error_mask.sum().item()
                    
                    if rel_error_count > 0:
                        scores_rel_error, scores_mean_rel_error, _ = _safe_finite_stats(rel_error_tensor, rel_error_mask)
                    else:
                        # If all reference values are too small, use absolute error
                        scores_rel_error = min(scores_abs_error, 1e10)
                        scores_mean_rel_error = min(scores_mean_abs_error, 1e10)
                else:
                    scores_abs_error = 0.0
                    scores_mean_abs_error = 0.0
                    scores_rel_error = 0.0
                    scores_mean_rel_error = 0.0
            else:
                scores_diff = (impl_scores - reference_scores).abs()
                # Filter out inf and nan (use safe method for large tensors)
                valid_mask = _safe_isfinite_mask(scores_diff) & _safe_isfinite_mask(reference_scores)
                count = valid_mask.sum().item()
                if count > 0:
                    scores_abs_error, scores_mean_abs_error, _ = _safe_finite_stats(scores_diff, valid_mask)
                    
                    # Calculate relative error (only for non-zero reference values)
                    ref_scores_abs = reference_scores.abs()
                    ref_scores_abs_safe = torch.clamp(ref_scores_abs, min=1e-6)
                    rel_error_tensor = scores_diff / ref_scores_abs_safe
                    # Replace any inf/nan values that might have been created before clamp
                    rel_error_tensor = torch.where(torch.isfinite(rel_error_tensor), rel_error_tensor, torch.tensor(1e10, device=rel_error_tensor.device, dtype=rel_error_tensor.dtype))
                    # Clamp relative error to avoid inf (max relative error = 1e10)
                    rel_error_tensor = torch.clamp(rel_error_tensor, max=1e10)
                    rel_error_mask = valid_mask & (ref_scores_abs > 1e-6) & _safe_isfinite_mask(rel_error_tensor)
                    rel_error_count = rel_error_mask.sum().item()
                    
                    if rel_error_count > 0:
                        scores_rel_error, scores_mean_rel_error, _ = _safe_finite_stats(rel_error_tensor, rel_error_mask)
                    else:
                        scores_rel_error = min(scores_abs_error, 1e10)
                        scores_mean_rel_error = min(scores_mean_abs_error, 1e10)
                else:
                    scores_abs_error = float('inf')
                    scores_mean_abs_error = float('inf')
                    scores_rel_error = 1e10  # Clamp to max relative error
                    scores_mean_rel_error = 1e10
            
            errors[impl_name].update({
                'scores_max_abs_error': scores_abs_error,
                'scores_mean_abs_error': scores_mean_abs_error,
                'scores_max_rel_error': scores_rel_error,
                'scores_mean_rel_error': scores_mean_rel_error,
            })
        
        # Calculate row_sum errors if available
        if 'row_sum' in impl_result:
            impl_row_sum = impl_result['row_sum']
            reference_scores_for_sum = reference_scores.clone()
            reference_scores_for_sum = torch.where(
                torch.isinf(reference_scores_for_sum) & (reference_scores_for_sum < 0),
                torch.zeros_like(reference_scores_for_sum),
                reference_scores_for_sum
            )
            reference_row_sum = reference_scores_for_sum.sum(dim=-1)
            
            row_sum_diff = (impl_row_sum - reference_row_sum).abs()
            # Filter out inf and nan (use safe method for large tensors)
            valid_mask = _safe_isfinite_mask(row_sum_diff) & _safe_isfinite_mask(reference_row_sum)
            count = valid_mask.sum().item()
            if count > 0:
                row_sum_abs_error, row_sum_mean_abs_error, _ = _safe_finite_stats(row_sum_diff, valid_mask)
                
                # Calculate relative error (only for non-zero reference values)
                ref_row_sum_abs = reference_row_sum.abs()
                ref_row_sum_abs_safe = torch.clamp(ref_row_sum_abs, min=1e-6)
                rel_error_tensor = row_sum_diff / ref_row_sum_abs_safe
                # Replace any inf/nan values that might have been created before clamp
                rel_error_tensor = torch.where(torch.isfinite(rel_error_tensor), rel_error_tensor, torch.tensor(1e10, device=rel_error_tensor.device, dtype=rel_error_tensor.dtype))
                # Clamp relative error to avoid inf (max relative error = 1e10)
                rel_error_tensor = torch.clamp(rel_error_tensor, max=1e10)
                rel_error_mask = valid_mask & (ref_row_sum_abs > 1e-6) & _safe_isfinite_mask(rel_error_tensor)
                rel_error_count = rel_error_mask.sum().item()
                
                if rel_error_count > 0:
                    row_sum_rel_error, row_sum_mean_rel_error, _ = _safe_finite_stats(rel_error_tensor, rel_error_mask)
                else:
                    row_sum_rel_error = min(row_sum_abs_error, 1e10)
                    row_sum_mean_rel_error = min(row_sum_mean_abs_error, 1e10)
            else:
                row_sum_abs_error = float('inf')
                row_sum_mean_abs_error = float('inf')
                row_sum_rel_error = 1e10  # Clamp to max relative error
                row_sum_mean_rel_error = 1e10
            
            errors[impl_name].update({
                'row_sum_max_abs_error': row_sum_abs_error,
                'row_sum_mean_abs_error': row_sum_mean_abs_error,
                'row_sum_max_rel_error': row_sum_rel_error,
                'row_sum_mean_rel_error': row_sum_mean_rel_error,
            })
        
        # Calculate col_sum errors if available
        if 'col_sum' in impl_result:
            impl_col_sum = impl_result['col_sum']
            reference_scores_for_col_sum = reference_scores.clone()
            reference_scores_for_col_sum = torch.where(
                torch.isinf(reference_scores_for_col_sum) & (reference_scores_for_col_sum < 0),
                torch.zeros_like(reference_scores_for_col_sum),
                reference_scores_for_col_sum
            )
            reference_col_sum = reference_scores_for_col_sum.sum(dim=-2)
            
            col_sum_diff = (impl_col_sum - reference_col_sum).abs()
            # Filter out inf and nan (use safe method for large tensors)
            valid_mask = _safe_isfinite_mask(col_sum_diff) & _safe_isfinite_mask(reference_col_sum)
            count = valid_mask.sum().item()
            if count > 0:
                col_sum_abs_error, col_sum_mean_abs_error, _ = _safe_finite_stats(col_sum_diff, valid_mask)
                
                # Calculate relative error (only for non-zero reference values)
                ref_col_sum_abs = reference_col_sum.abs()
                ref_col_sum_abs_safe = torch.clamp(ref_col_sum_abs, min=1e-6)
                rel_error_tensor = col_sum_diff / ref_col_sum_abs_safe
                # Replace any inf/nan values that might have been created before clamp
                rel_error_tensor = torch.where(torch.isfinite(rel_error_tensor), rel_error_tensor, torch.tensor(1e10, device=rel_error_tensor.device, dtype=rel_error_tensor.dtype))
                # Clamp relative error to avoid inf (max relative error = 1e10)
                rel_error_tensor = torch.clamp(rel_error_tensor, max=1e10)
                rel_error_mask = valid_mask & (ref_col_sum_abs > 1e-6) & _safe_isfinite_mask(rel_error_tensor)
                rel_error_count = rel_error_mask.sum().item()
                
                if rel_error_count > 0:
                    col_sum_rel_error, col_sum_mean_rel_error, _ = _safe_finite_stats(rel_error_tensor, rel_error_mask)
                else:
                    col_sum_rel_error = min(col_sum_abs_error, 1e10)
                    col_sum_mean_rel_error = min(col_sum_mean_abs_error, 1e10)
            else:
                col_sum_abs_error = float('inf')
                col_sum_mean_abs_error = float('inf')
                col_sum_rel_error = 1e10  # Clamp to max relative error
                col_sum_mean_rel_error = 1e10
            
            errors[impl_name].update({
                'col_sum_max_abs_error': col_sum_abs_error,
                'col_sum_mean_abs_error': col_sum_mean_abs_error,
                'col_sum_max_rel_error': col_sum_rel_error,
                'col_sum_mean_rel_error': col_sum_mean_rel_error,
            })
        
        # Calculate cross_token_softmax errors if available
        if 'cross_token_softmax' in impl_result:
            impl_cross_token_softmax = impl_result['cross_token_softmax']
            # Reference: compute softmax attention weights and sum over queries
            reference_scores_for_cross_token_softmax = reference_scores.clone()
            # Apply softmax to reference scores
            # Handle causal mask: replace -inf with a large negative value before softmax
            # Use a value that's safe for float16: -65504 is the min for fp16, use -6e4 to be safe
            dtype = reference_scores_for_cross_token_softmax.dtype
            if dtype == torch.float16:
                mask_value = -6e4  # Safe for float16 (min is ~-65504)
            else:
                mask_value = -1e10  # Safe for float32/float64

            reference_scores_for_cross_token_softmax = torch.where(
                torch.isinf(reference_scores_for_cross_token_softmax) & (reference_scores_for_cross_token_softmax < 0),
                torch.full_like(reference_scores_for_cross_token_softmax, mask_value),
                reference_scores_for_cross_token_softmax
            )
            # Compute softmax along the last dimension (over keys for each query)
            reference_attention_weights = torch.softmax(reference_scores_for_cross_token_softmax, dim=-1)

            # Apply split logic: only queries >= split contribute to keys < split
            # reference_attention_weights shape: (B, H, M, N)
            # We want: sum over queries[split:] for keys[:split]
            reference_attention_weights_split = reference_attention_weights[:, :, split:, :split]
            # Sum over queries dimension (dim=-2) to get column-wise sum
            reference_cross_token_softmax = reference_attention_weights_split.sum(dim=-2)

            cross_token_softmax_diff = (impl_cross_token_softmax - reference_cross_token_softmax).abs()
            # Filter out inf and nan (use safe method for large tensors)
            valid_mask = _safe_isfinite_mask(cross_token_softmax_diff) & _safe_isfinite_mask(reference_cross_token_softmax)
            count = valid_mask.sum().item()
            if count > 0:
                cross_token_softmax_abs_error, cross_token_softmax_mean_abs_error, _ = _safe_finite_stats(cross_token_softmax_diff, valid_mask)
                
                # Calculate relative error (only for non-zero reference values)
                ref_cross_token_softmax_abs = reference_cross_token_softmax.abs()
                ref_cross_token_softmax_abs_safe = torch.clamp(ref_cross_token_softmax_abs, min=1e-6)
                rel_error_tensor = cross_token_softmax_diff / ref_cross_token_softmax_abs_safe
                # Replace any inf/nan values that might have been created before clamp
                rel_error_tensor = torch.where(torch.isfinite(rel_error_tensor), rel_error_tensor, torch.tensor(1e10, device=rel_error_tensor.device, dtype=rel_error_tensor.dtype))
                # Clamp relative error to avoid inf (max relative error = 1e10)
                rel_error_tensor = torch.clamp(rel_error_tensor, max=1e10)
                rel_error_mask = valid_mask & (ref_cross_token_softmax_abs > 1e-6) & _safe_isfinite_mask(rel_error_tensor)
                rel_error_count = rel_error_mask.sum().item()
                
                if rel_error_count > 0:
                    cross_token_softmax_rel_error, cross_token_softmax_mean_rel_error, _ = _safe_finite_stats(rel_error_tensor, rel_error_mask)
                else:
                    cross_token_softmax_rel_error = min(cross_token_softmax_abs_error, 1e10)
                    cross_token_softmax_mean_rel_error = min(cross_token_softmax_mean_abs_error, 1e10)
            else:
                cross_token_softmax_abs_error = float('inf')
                cross_token_softmax_mean_abs_error = float('inf')
                cross_token_softmax_rel_error = 1e10  # Clamp to max relative error
                cross_token_softmax_mean_rel_error = 1e10
            
            errors[impl_name].update({
                'cross_token_softmax_max_abs_error': cross_token_softmax_abs_error,
                'cross_token_softmax_mean_abs_error': cross_token_softmax_mean_abs_error,
                'cross_token_softmax_max_rel_error': cross_token_softmax_rel_error,
                'cross_token_softmax_mean_rel_error': cross_token_softmax_mean_rel_error,
            })
        
        # Calculate cross_token_qk errors if available
        if 'cross_token_qk' in impl_result:
            impl_cross_token_qk = impl_result['cross_token_qk']
            reference_scores_for_split = reference_scores.clone()
            reference_scores_for_split = torch.where(
                torch.isinf(reference_scores_for_split) & (reference_scores_for_split < 0),
                torch.zeros_like(reference_scores_for_split),
                reference_scores_for_split
            )
            reference_cross_token_qk = reference_scores_for_split[:, :, split:, :split].sum(dim=-2)
            
            cross_token_qk_diff = (impl_cross_token_qk - reference_cross_token_qk).abs()
            # Filter out inf and nan (use safe method for large tensors)
            valid_mask = _safe_isfinite_mask(cross_token_qk_diff) & _safe_isfinite_mask(reference_cross_token_qk)
            count = valid_mask.sum().item()
            if count > 0:
                cross_token_qk_abs_error, cross_token_qk_mean_abs_error, _ = _safe_finite_stats(cross_token_qk_diff, valid_mask)
                
                # Calculate relative error (only for non-zero reference values)
                ref_cross_token_qk_abs = reference_cross_token_qk.abs()
                ref_cross_token_qk_abs_safe = torch.clamp(ref_cross_token_qk_abs, min=1e-6)
                rel_error_tensor = cross_token_qk_diff / ref_cross_token_qk_abs_safe
                # Replace any inf/nan values that might have been created before clamp
                rel_error_tensor = torch.where(torch.isfinite(rel_error_tensor), rel_error_tensor, torch.tensor(1e10, device=rel_error_tensor.device, dtype=rel_error_tensor.dtype))
                # Clamp relative error to avoid inf (max relative error = 1e10)
                rel_error_tensor = torch.clamp(rel_error_tensor, max=1e10)
                rel_error_mask = valid_mask & (ref_cross_token_qk_abs > 1e-6) & _safe_isfinite_mask(rel_error_tensor)
                rel_error_count = rel_error_mask.sum().item()
                
                if rel_error_count > 0:
                    cross_token_qk_rel_error, cross_token_qk_mean_rel_error, _ = _safe_finite_stats(rel_error_tensor, rel_error_mask)
                else:
                    cross_token_qk_rel_error = min(cross_token_qk_abs_error, 1e10)
                    cross_token_qk_mean_rel_error = min(cross_token_qk_mean_abs_error, 1e10)
            else:
                cross_token_qk_abs_error = float('inf')
                cross_token_qk_mean_abs_error = float('inf')
                cross_token_qk_rel_error = 1e10  # Clamp to max relative error
                cross_token_qk_mean_rel_error = 1e10
            
            errors[impl_name].update({
                'cross_token_qk_max_abs_error': cross_token_qk_abs_error,
                'cross_token_qk_mean_abs_error': cross_token_qk_mean_abs_error,
                'cross_token_qk_max_rel_error': cross_token_qk_rel_error,
                'cross_token_qk_mean_rel_error': cross_token_qk_mean_rel_error,
            })
    
    # Print error summary in a formatted table
    print(f"    Correctness Verification Results (relative to Naive implementation):")
    print(f"    {'=' * 120}")
    
    for impl_name, error_metrics in errors.items():
        impl_display_name = {
            'flash_with_scores': '+ Scores',
            'flash_row_sum': '+ Row Sum',
            'flash_col_sum': '+ Col Sum',
            'flash_col_sum_sequential': '+ Col Sum (Sequential)',
            'flash_cross_token_softmax': '+ Cross-Token Softmax Sum',
            'flash_cross_token_softmax_buffered': '+ Cross-Token Softmax Sum (Buffered)',
            'flash_cross_token_qk': '+ Cross-Token QK Sum',
        }.get(impl_name, impl_name)
        
        print(f"    {impl_display_name}:")
        print(f"      Output:")
        # Format errors, handling inf and nan
        output_max_abs = error_metrics['output_max_abs_error']
        output_mean_abs = error_metrics['output_mean_abs_error']
        output_max_rel = error_metrics['output_max_rel_error']
        output_mean_rel = error_metrics['output_mean_rel_error']
        
        abs_str_max = f"{output_max_abs:.6e}" if torch.isfinite(torch.tensor(output_max_abs)) else "inf"
        abs_str_mean = f"{output_mean_abs:.6e}" if torch.isfinite(torch.tensor(output_mean_abs)) else "inf"
        # Format relative error, showing ">1e10" if clamped
        if not torch.isfinite(torch.tensor(output_max_rel)):
            rel_str_max = "inf"
        elif output_max_rel >= 1e10:
            rel_str_max = ">1e10"
        else:
            rel_str_max = f"{output_max_rel:.6e}"
        if not torch.isfinite(torch.tensor(output_mean_rel)):
            rel_str_mean = "inf"
        elif output_mean_rel >= 1e10:
            rel_str_mean = ">1e10"
        else:
            rel_str_mean = f"{output_mean_rel:.6e}"
        
        print(f"        Max Absolute Error: {abs_str_max}  |  Mean Absolute Error: {abs_str_mean}")
        print(f"        Max Relative Error: {rel_str_max}  |  Mean Relative Error: {rel_str_mean}")
        
        if 'scores_max_abs_error' in error_metrics:
            print(f"      Scores:")
            scores_max_abs = error_metrics['scores_max_abs_error']
            scores_mean_abs = error_metrics['scores_mean_abs_error']
            scores_max_rel = error_metrics['scores_max_rel_error']
            scores_mean_rel = error_metrics['scores_mean_rel_error']
            
            scores_abs_str_max = f"{scores_max_abs:.6e}" if torch.isfinite(torch.tensor(scores_max_abs)) else "inf"
            scores_abs_str_mean = f"{scores_mean_abs:.6e}" if torch.isfinite(torch.tensor(scores_mean_abs)) else "inf"
            # Format relative error, showing ">1e10" if clamped
            if not torch.isfinite(torch.tensor(scores_max_rel)):
                scores_rel_str_max = "inf"
            elif scores_max_rel >= 1e10:
                scores_rel_str_max = ">1e10"
            else:
                scores_rel_str_max = f"{scores_max_rel:.6e}"
            if not torch.isfinite(torch.tensor(scores_mean_rel)):
                scores_rel_str_mean = "inf"
            elif scores_mean_rel >= 1e10:
                scores_rel_str_mean = ">1e10"
            else:
                scores_rel_str_mean = f"{scores_mean_rel:.6e}"
            
            print(f"        Max Absolute Error: {scores_abs_str_max}  |  Mean Absolute Error: {scores_abs_str_mean}")
            print(f"        Max Relative Error: {scores_rel_str_max}  |  Mean Relative Error: {scores_rel_str_mean}")
        
        if 'row_sum_max_abs_error' in error_metrics:
            print(f"      Row Sum:")
            row_sum_max_abs = error_metrics['row_sum_max_abs_error']
            row_sum_mean_abs = error_metrics['row_sum_mean_abs_error']
            row_sum_max_rel = error_metrics['row_sum_max_rel_error']
            row_sum_mean_rel = error_metrics['row_sum_mean_rel_error']
            
            row_sum_abs_str_max = f"{row_sum_max_abs:.6e}" if torch.isfinite(torch.tensor(row_sum_max_abs)) else "inf"
            row_sum_abs_str_mean = f"{row_sum_mean_abs:.6e}" if torch.isfinite(torch.tensor(row_sum_mean_abs)) else "inf"
            # Format relative error, showing ">1e10" if clamped
            if not torch.isfinite(torch.tensor(row_sum_max_rel)):
                row_sum_rel_str_max = "inf"
            elif row_sum_max_rel >= 1e10:
                row_sum_rel_str_max = ">1e10"
            else:
                row_sum_rel_str_max = f"{row_sum_max_rel:.6e}"
            if not torch.isfinite(torch.tensor(row_sum_mean_rel)):
                row_sum_rel_str_mean = "inf"
            elif row_sum_mean_rel >= 1e10:
                row_sum_rel_str_mean = ">1e10"
            else:
                row_sum_rel_str_mean = f"{row_sum_mean_rel:.6e}"
            
            print(f"        Max Absolute Error: {row_sum_abs_str_max}  |  Mean Absolute Error: {row_sum_abs_str_mean}")
            print(f"        Max Relative Error: {row_sum_rel_str_max}  |  Mean Relative Error: {row_sum_rel_str_mean}")
        
        if 'col_sum_max_abs_error' in error_metrics:
            print(f"      Col Sum:")
            col_sum_max_abs = error_metrics['col_sum_max_abs_error']
            col_sum_mean_abs = error_metrics['col_sum_mean_abs_error']
            col_sum_max_rel = error_metrics['col_sum_max_rel_error']
            col_sum_mean_rel = error_metrics['col_sum_mean_rel_error']
            
            col_sum_abs_str_max = f"{col_sum_max_abs:.6e}" if torch.isfinite(torch.tensor(col_sum_max_abs)) else "inf"
            col_sum_abs_str_mean = f"{col_sum_mean_abs:.6e}" if torch.isfinite(torch.tensor(col_sum_mean_abs)) else "inf"
            # Format relative error, showing ">1e10" if clamped
            if not torch.isfinite(torch.tensor(col_sum_max_rel)):
                col_sum_rel_str_max = "inf"
            elif col_sum_max_rel >= 1e10:
                col_sum_rel_str_max = ">1e10"
            else:
                col_sum_rel_str_max = f"{col_sum_max_rel:.6e}"
            if not torch.isfinite(torch.tensor(col_sum_mean_rel)):
                col_sum_rel_str_mean = "inf"
            elif col_sum_mean_rel >= 1e10:
                col_sum_rel_str_mean = ">1e10"
            else:
                col_sum_rel_str_mean = f"{col_sum_mean_rel:.6e}"
            
            print(f"        Max Absolute Error: {col_sum_abs_str_max}  |  Mean Absolute Error: {col_sum_abs_str_mean}")
            print(f"        Max Relative Error: {col_sum_rel_str_max}  |  Mean Relative Error: {col_sum_rel_str_mean}")
        
        if 'cross_token_softmax_max_abs_error' in error_metrics:
            print(f"      Cross-Token Softmax Sum:")
            cross_token_softmax_max_abs = error_metrics['cross_token_softmax_max_abs_error']
            cross_token_softmax_mean_abs = error_metrics['cross_token_softmax_mean_abs_error']
            cross_token_softmax_max_rel = error_metrics['cross_token_softmax_max_rel_error']
            cross_token_softmax_mean_rel = error_metrics['cross_token_softmax_mean_rel_error']
            
            cross_token_softmax_abs_str_max = f"{cross_token_softmax_max_abs:.6e}" if torch.isfinite(torch.tensor(cross_token_softmax_max_abs)) else "inf"
            cross_token_softmax_abs_str_mean = f"{cross_token_softmax_mean_abs:.6e}" if torch.isfinite(torch.tensor(cross_token_softmax_mean_abs)) else "inf"
            # Format relative error, showing ">1e10" if clamped
            if not torch.isfinite(torch.tensor(cross_token_softmax_max_rel)):
                cross_token_softmax_rel_str_max = "inf"
            elif cross_token_softmax_max_rel >= 1e10:
                cross_token_softmax_rel_str_max = ">1e10"
            else:
                cross_token_softmax_rel_str_max = f"{cross_token_softmax_max_rel:.6e}"
            if not torch.isfinite(torch.tensor(cross_token_softmax_mean_rel)):
                cross_token_softmax_rel_str_mean = "inf"
            elif cross_token_softmax_mean_rel >= 1e10:
                cross_token_softmax_rel_str_mean = ">1e10"
            else:
                cross_token_softmax_rel_str_mean = f"{cross_token_softmax_mean_rel:.6e}"
            
            print(f"        Max Absolute Error: {cross_token_softmax_abs_str_max}  |  Mean Absolute Error: {cross_token_softmax_abs_str_mean}")
            print(f"        Max Relative Error: {cross_token_softmax_rel_str_max}  |  Mean Relative Error: {cross_token_softmax_rel_str_mean}")
        
        if 'cross_token_qk_max_abs_error' in error_metrics:
            print(f"      Cross-Token QK Sum:")
            cross_token_qk_max_abs = error_metrics['cross_token_qk_max_abs_error']
            cross_token_qk_mean_abs = error_metrics['cross_token_qk_mean_abs_error']
            cross_token_qk_max_rel = error_metrics['cross_token_qk_max_rel_error']
            cross_token_qk_mean_rel = error_metrics['cross_token_qk_mean_rel_error']
            
            cross_token_qk_abs_str_max = f"{cross_token_qk_max_abs:.6e}" if torch.isfinite(torch.tensor(cross_token_qk_max_abs)) else "inf"
            cross_token_qk_abs_str_mean = f"{cross_token_qk_mean_abs:.6e}" if torch.isfinite(torch.tensor(cross_token_qk_mean_abs)) else "inf"
            # Format relative error, showing ">1e10" if clamped
            if not torch.isfinite(torch.tensor(cross_token_qk_max_rel)):
                cross_token_qk_rel_str_max = "inf"
            elif cross_token_qk_max_rel >= 1e10:
                cross_token_qk_rel_str_max = ">1e10"
            else:
                cross_token_qk_rel_str_max = f"{cross_token_qk_max_rel:.6e}"
            if not torch.isfinite(torch.tensor(cross_token_qk_mean_rel)):
                cross_token_qk_rel_str_mean = "inf"
            elif cross_token_qk_mean_rel >= 1e10:
                cross_token_qk_rel_str_mean = ">1e10"
            else:
                cross_token_qk_rel_str_mean = f"{cross_token_qk_mean_rel:.6e}"
            
            print(f"        Max Absolute Error: {cross_token_qk_abs_str_max}  |  Mean Absolute Error: {cross_token_qk_abs_str_mean}")
            print(f"        Max Relative Error: {cross_token_qk_rel_str_max}  |  Mean Relative Error: {cross_token_qk_rel_str_mean}")
        
        print()
    
    print(f"    {'=' * 120}")
    
    return errors


def run_benchmark_suite(
    test_configs: List[Dict],
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
    include_sum_ops: bool = True,
    enable_config_search: bool = False,
    config_search_trials: int = 3,
    config_search_warmup: int = 5,
    verify_correctness_flag: bool = True,
    tolerance: float = 1e-2,
) -> None:
    """
    Run benchmark suite and output results
    """
    print("=" * 140)
    print("Flash Attention with Scores - Performance Benchmark")
    print("=" * 140)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using Triton do_bench: warmup={warmup_ms}ms, rep={rep_ms}ms, runs={num_runs + 2} (using last {num_runs})")
    print("=" * 140)
    print()
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        config_name = config.get('name', 'Unnamed')
        print(f"Test {i}/{len(test_configs)}: {config_name}")
        print(f"  Config: batch={config['batch_size']}, "
              f"heads_q={config['num_heads_q']}, heads_k={config['num_heads_k']}, "
              f"seq_q={config['seq_len_q']}, seq_k={config['seq_len_k']}, "
              f"head_dim={config['head_dim']}, causal={config.get('causal', True)}")
        
        causal = config.get('causal', True)
        dtype = config.get('dtype', torch.float16)
        
        # Verify correctness if enabled
        if verify_correctness_flag:
            try:
                errors = verify_correctness(
                    batch_size=config['batch_size'],
                    num_heads_q=config['num_heads_q'],
                    num_heads_k=config['num_heads_k'],
                    seq_len_q=config['seq_len_q'],
                    seq_len_k=config['seq_len_k'],
                    head_dim=config['head_dim'],
                    causal=causal,
                    dtype=dtype,
                    device=device,
                    tolerance=tolerance,
                    split=config.get('split', 768),
                )
            except Exception as e:
                print(f"  ⚠ Correctness verification failed: {e}")
                import traceback
                traceback.print_exc()
        
        try:
            # Test naive implementation
            print("  [Naive (PyTorch Native)]")
            naive_result = benchmark_attention_with_scores(
                batch_size=config['batch_size'],
                num_heads_q=config['num_heads_q'],
                num_heads_k=config['num_heads_k'],
                seq_len_q=config['seq_len_q'],
                seq_len_k=config['seq_len_k'],
                head_dim=config['head_dim'],
                causal=causal,
                dtype=dtype,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
                num_runs=num_runs,
                device=device,
                implementation="naive"
            )
            
            # Test PyTorch SDPA
            print("  [PyTorch SDPA + Scores]")
            pytorch_result = benchmark_attention_with_scores(
                batch_size=config['batch_size'],
                num_heads_q=config['num_heads_q'],
                num_heads_k=config['num_heads_k'],
                seq_len_q=config['seq_len_q'],
                seq_len_k=config['seq_len_k'],
                head_dim=config['head_dim'],
                causal=causal,
                dtype=dtype,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
                num_runs=num_runs,
                device=device,
                implementation="pytorch_sdpa"
            )
            
            # Test Flash Attention
            print("  [+ Scores]")
            flash_result = benchmark_attention_with_scores(
                batch_size=config['batch_size'],
                num_heads_q=config['num_heads_q'],
                num_heads_k=config['num_heads_k'],
                seq_len_q=config['seq_len_q'],
                seq_len_k=config['seq_len_k'],
                head_dim=config['head_dim'],
                causal=causal,
                dtype=dtype,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
                num_runs=num_runs,
                device=device,
                implementation="flash",
                enable_config_search=enable_config_search,
                config_search_trials=config_search_trials,
                config_search_warmup=config_search_warmup,
            )
            
            # Calculate speedup
            if flash_result["median_time_ms"] != float("inf") and naive_result["median_time_ms"] != float("inf"):
                flash_speedup = naive_result["median_time_ms"] / flash_result["median_time_ms"]
            else:
                flash_speedup = 0.0
            
            if pytorch_result["median_time_ms"] != float("inf") and naive_result["median_time_ms"] != float("inf"):
                pytorch_speedup = naive_result["median_time_ms"] / pytorch_result["median_time_ms"]
            else:
                pytorch_speedup = 0.0
            
            results.append({
                "config": config,
                "naive": naive_result,
                "pytorch_sdpa": pytorch_result,
                "flash": flash_result,
                "flash_speedup": flash_speedup,
                "pytorch_speedup": pytorch_speedup,
            })
            
            print(f"    Results: Naive={naive_result['median_time_ms']:.3f}ms, "
                  f"PyTorch SDPA={pytorch_result['median_time_ms']:.3f}ms, "
                  f"Flash={flash_result['median_time_ms']:.3f}ms")
            print(f"    Speedup: PyTorch SDPA={pytorch_speedup:.2f}, Flash={flash_speedup:.2f}")
            
            # Benchmark row_sum and col_sum if requested
            if include_sum_ops:
                # Compute split value for benchmarks that need it
                split = config.get('split', 768)

                print("  [+ Row Sum]")
                row_sum_result = benchmark_row_sum(
                    batch_size=config['batch_size'],
                    num_heads_q=config['num_heads_q'],
                    num_heads_k=config['num_heads_k'],
                    seq_len_q=config['seq_len_q'],
                    seq_len_k=config['seq_len_k'],
                    head_dim=config['head_dim'],
                    causal=causal,
                    dtype=dtype,
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                    num_runs=num_runs,
                    device=device,
                )
                
                print("  [+ Col Sum (Reverse-Order)]")
                col_sum_result = benchmark_col_sum(
                    batch_size=config['batch_size'],
                    num_heads_q=config['num_heads_q'],
                    num_heads_k=config['num_heads_k'],
                    seq_len_q=config['seq_len_q'],
                    seq_len_k=config['seq_len_k'],
                    head_dim=config['head_dim'],
                    causal=causal,
                    dtype=dtype,
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                    num_runs=num_runs,
                    device=device,
                )
                
                print("  [+ Col Sum (Sequential)]")
                col_sum_sequential_result = benchmark_col_sum_sequential(
                    batch_size=config['batch_size'],
                    num_heads_q=config['num_heads_q'],
                    num_heads_k=config['num_heads_k'],
                    seq_len_q=config['seq_len_q'],
                    seq_len_k=config['seq_len_k'],
                    head_dim=config['head_dim'],
                    causal=causal,
                    dtype=dtype,
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                    num_runs=num_runs,
                    device=device,
                )

                print("  [+ Col Softmax Sum (Sequential)]")
                col_softmax_sum_result = benchmark_col_softmax_sum(
                    batch_size=config['batch_size'],
                    num_heads_q=config['num_heads_q'],
                    num_heads_k=config['num_heads_k'],
                    seq_len_q=config['seq_len_q'],
                    seq_len_k=config['seq_len_k'],
                    head_dim=config['head_dim'],
                    causal=causal,
                    dtype=dtype,
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                    num_runs=num_runs,
                    device=device,
                )

                print("  [+ Cross-Token Softmax Sum]")
                cross_token_softmax_result = benchmark_cross_token_softmax(
                    batch_size=config['batch_size'],
                    num_heads_q=config['num_heads_q'],
                    num_heads_k=config['num_heads_k'],
                    seq_len_q=config['seq_len_q'],
                    seq_len_k=config['seq_len_k'],
                    head_dim=config['head_dim'],
                    split=split,
                    causal=causal,
                    dtype=dtype,
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                    num_runs=num_runs,
                    device=device,
                )

                print("  [+ Cross-Token Softmax Sum (Buffered)]")
                cross_token_softmax_buffered_result = benchmark_cross_token_softmax_buffered(
                    batch_size=config['batch_size'],
                    num_heads_q=config['num_heads_q'],
                    num_heads_k=config['num_heads_k'],
                    seq_len_q=config['seq_len_q'],
                    seq_len_k=config['seq_len_k'],
                    head_dim=config['head_dim'],
                    split=split,
                    causal=causal,
                    dtype=dtype,
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                    num_runs=num_runs,
                    device=device,
                )

                # Benchmark cross_token_qk if split is specified and valid
                if split < config['seq_len_k'] and split < config['seq_len_q']:
                    print(f"  [+ Cross-Token QK Sum (split={split})]")
                    cross_token_qk_result = benchmark_cross_token_qk(
                        batch_size=config['batch_size'],
                        num_heads_q=config['num_heads_q'],
                        num_heads_k=config['num_heads_k'],
                        seq_len_q=config['seq_len_q'],
                        seq_len_k=config['seq_len_k'],
                        head_dim=config['head_dim'],
                        split=split,
                        causal=causal,
                        dtype=dtype,
                        warmup_ms=warmup_ms,
                        rep_ms=rep_ms,
                        num_runs=num_runs,
                        device=device,
                    )
                else:
                    print(f"  [+ Cross-Token QK Sum (split={split})] - Skipped (split >= seq_len)")
                    cross_token_qk_result = {
                        "avg_time_ms": float("inf"),
                        "median_time_ms": float("inf"),
                        "min_time_ms": float("inf"),
                        "max_time_ms": float("inf"),
                        "throughput_tokens_per_sec": 0.0,
                        "tflops": 0.0,
                    }
                
                # Calculate speedups relative to naive
                naive_time = naive_result["median_time_ms"]
                row_sum_speedup = naive_time / row_sum_result["median_time_ms"] if naive_time != float("inf") and row_sum_result["median_time_ms"] != float("inf") else 0.0
                col_sum_speedup = naive_time / col_sum_result["median_time_ms"] if naive_time != float("inf") and col_sum_result["median_time_ms"] != float("inf") else 0.0
                col_sum_sequential_speedup = naive_time / col_sum_sequential_result["median_time_ms"] if naive_time != float("inf") and col_sum_sequential_result["median_time_ms"] != float("inf") else 0.0
                col_softmax_sum_speedup = naive_time / col_softmax_sum_result["median_time_ms"] if naive_time != float("inf") and col_softmax_sum_result["median_time_ms"] != float("inf") else 0.0
                cross_token_softmax_speedup = naive_time / cross_token_softmax_result["median_time_ms"] if naive_time != float("inf") and cross_token_softmax_result["median_time_ms"] != float("inf") else 0.0
                cross_token_softmax_buffered_speedup = naive_time / cross_token_softmax_buffered_result["median_time_ms"] if naive_time != float("inf") and cross_token_softmax_buffered_result["median_time_ms"] != float("inf") else 0.0
                cross_token_qk_speedup = naive_time / cross_token_qk_result["median_time_ms"] if naive_time != float("inf") and cross_token_qk_result["median_time_ms"] != float("inf") else 0.0
                
                print(f"    Row Sum: {row_sum_result['median_time_ms']:.3f}ms ({row_sum_speedup:.2f} vs Naive), "
                      f"Col Sum (Reverse): {col_sum_result['median_time_ms']:.3f}ms ({col_sum_speedup:.2f} vs Naive)")
                print(f"    Col Sum (Sequential): {col_sum_sequential_result['median_time_ms']:.3f}ms ({col_sum_sequential_speedup:.2f} vs Naive), "
                      f"Col Softmax Sum: {col_softmax_sum_result['median_time_ms']:.3f}ms ({col_softmax_sum_speedup:.2f} vs Naive)")
                print(f"    Cross-Token Softmax: {cross_token_softmax_result['median_time_ms']:.3f}ms ({cross_token_softmax_speedup:.2f} vs Naive), "
                      f"Cross-Token Softmax (Buffered): {cross_token_softmax_buffered_result['median_time_ms']:.3f}ms ({cross_token_softmax_buffered_speedup:.2f} vs Naive)")
                print(f"    Cross-Token QK: {cross_token_qk_result['median_time_ms']:.3f}ms ({cross_token_qk_speedup:.2f} vs Naive)")
                
                results[-1]["row_sum"] = row_sum_result
                results[-1]["col_sum"] = col_sum_result
                results[-1]["col_sum_sequential"] = col_sum_sequential_result
                results[-1]["col_softmax_sum"] = col_softmax_sum_result
                results[-1]["cross_token_softmax"] = cross_token_softmax_result
                results[-1]["cross_token_softmax_buffered"] = cross_token_softmax_buffered_result
                results[-1]["cross_token_qk"] = cross_token_qk_result
                results[-1]["row_sum_speedup"] = row_sum_speedup
                results[-1]["col_sum_speedup"] = col_sum_speedup
                results[-1]["col_sum_sequential_speedup"] = col_sum_sequential_speedup
                results[-1]["col_softmax_sum_speedup"] = col_softmax_sum_speedup
                results[-1]["cross_token_softmax_speedup"] = cross_token_softmax_speedup
                results[-1]["cross_token_softmax_buffered_speedup"] = cross_token_softmax_buffered_speedup
                results[-1]["cross_token_qk_speedup"] = cross_token_qk_speedup
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Output summary table (transposed: methods as rows, configs as columns)
    print("\n" + "=" * 200)
    print("Performance Summary (All operators, speedup relative to Naive)")
    print("=" * 200)
    
    # Collect all data first
    method_names = []
    method_data = {}
    
    if include_sum_ops:
        method_names = [
            "Naive (PyTorch)",
            "PyTorch SDPA",
            "+ Scores",
            "+ Row Sum",
            "+ Col Sum (Reverse)",
            "+ Col Sum (Sequential)",
            "+ Col Softmax Sum (Sequential)",
            "+ Cross-Token Softmax Sum",
            "+ Cross-Token Softmax Sum (Buffered)",
            "+ Cross-Token QK Sum"
        ]
        
        # Initialize data structure
        for method in method_names:
            method_data[method] = {"times": [], "speedups": []}
        
        # Collect data from results
        for result in results:
            config = result["config"]
            # Convert sequence length to K (1024 = 1K)
            seq_k = config['seq_len_q'] // 1024
            config_label = f"{seq_k}K"
            
            naive_time = result["naive"]["median_time_ms"]
            pytorch_time = result["pytorch_sdpa"]["median_time_ms"]
            flash_time = result["flash"]["median_time_ms"]
            
            pytorch_speedup = result["pytorch_speedup"]
            flash_speedup = result["flash_speedup"]
            
            method_data["Naive (PyTorch)"]["times"].append((config_label, naive_time))
            method_data["Naive (PyTorch)"]["speedups"].append((config_label, "-"))
            
            method_data["PyTorch SDPA"]["times"].append((config_label, pytorch_time))
            method_data["PyTorch SDPA"]["speedups"].append((config_label, pytorch_speedup))
            
            method_data["+ Scores"]["times"].append((config_label, flash_time))
            method_data["+ Scores"]["speedups"].append((config_label, flash_speedup))
            
            # Always collect row_sum and col_sum data, even if missing (will show N/A)
            row_sum_time = result.get("row_sum", {}).get("median_time_ms", float("inf"))
            col_sum_time = result.get("col_sum", {}).get("median_time_ms", float("inf"))
            col_sum_sequential_time = result.get("col_sum_sequential", {}).get("median_time_ms", float("inf"))
            col_softmax_sum_time = result.get("col_softmax_sum", {}).get("median_time_ms", float("inf"))
            cross_token_softmax_time = result.get("cross_token_softmax", {}).get("median_time_ms", float("inf"))
            cross_token_softmax_buffered_time = result.get("cross_token_softmax_buffered", {}).get("median_time_ms", float("inf"))
            cross_token_qk_time = result.get("cross_token_qk", {}).get("median_time_ms", float("inf"))

            row_sum_speedup = result.get("row_sum_speedup", 0.0)
            col_sum_speedup = result.get("col_sum_speedup", 0.0)
            col_sum_sequential_speedup = result.get("col_sum_sequential_speedup", 0.0)
            col_softmax_sum_speedup = result.get("col_softmax_sum_speedup", 0.0)
            cross_token_softmax_speedup = result.get("cross_token_softmax_speedup", 0.0)
            cross_token_softmax_buffered_speedup = result.get("cross_token_softmax_buffered_speedup", 0.0)
            cross_token_qk_speedup = result.get("cross_token_qk_speedup", 0.0)
            
            method_data["+ Row Sum"]["times"].append((config_label, row_sum_time))
            method_data["+ Row Sum"]["speedups"].append((config_label, row_sum_speedup))
            
            method_data["+ Col Sum (Reverse)"]["times"].append((config_label, col_sum_time))
            method_data["+ Col Sum (Reverse)"]["speedups"].append((config_label, col_sum_speedup))
            
            method_data["+ Col Sum (Sequential)"]["times"].append((config_label, col_sum_sequential_time))
            method_data["+ Col Sum (Sequential)"]["speedups"].append((config_label, col_sum_sequential_speedup))

            method_data["+ Col Softmax Sum (Sequential)"]["times"].append((config_label, col_softmax_sum_time))
            method_data["+ Col Softmax Sum (Sequential)"]["speedups"].append((config_label, col_softmax_sum_speedup))

            method_data["+ Cross-Token Softmax Sum"]["times"].append((config_label, cross_token_softmax_time))
            method_data["+ Cross-Token Softmax Sum"]["speedups"].append((config_label, cross_token_softmax_speedup))

            method_data["+ Cross-Token Softmax Sum (Buffered)"]["times"].append((config_label, cross_token_softmax_buffered_time))
            method_data["+ Cross-Token Softmax Sum (Buffered)"]["speedups"].append((config_label, cross_token_softmax_buffered_speedup))

            method_data["+ Cross-Token QK Sum"]["times"].append((config_label, cross_token_qk_time))
            method_data["+ Cross-Token QK Sum"]["speedups"].append((config_label, cross_token_qk_speedup))
        
        # Print header
        config_labels = [label for label, _ in method_data["Naive (PyTorch)"]["times"]]
        header = f"{'Method':<45} "
        for label in config_labels:
            header += f"{label + ' (ms)':<12} {'speed up':<12} "
        print(header)
        print("-" * 200)
        
        # Print each method as a row
        for method in method_names:
            row = f"{method:<45} "
            for i, config_label in enumerate(config_labels):
                time_val = method_data[method]["times"][i][1]
                speedup_val = method_data[method]["speedups"][i][1]
                
                if time_val == float("inf"):
                    time_str = "OOM/Error"
                else:
                    time_str = f"{time_val:.3f}"
                
                if speedup_val == "-":
                    speedup_str = "-"
                elif speedup_val == 0.0:
                    speedup_str = "N/A"
                else:
                    speedup_str = f"{speedup_val:.2f}x"
                
                row += f"{time_str:<12} {speedup_str:<12} "
            print(row)
    else:
        method_names = [
            "Naive (PyTorch)",
            "PyTorch SDPA",
            "+ Scores"
        ]
        
        # Initialize data structure
        for method in method_names:
            method_data[method] = {"times": [], "speedups": []}
        
        # Collect data from results
        for result in results:
            config = result["config"]
            # Convert sequence length to K (1024 = 1K)
            seq_k = config['seq_len_q'] // 1024
            config_label = f"{seq_k}K"
            
            naive_time = result["naive"]["median_time_ms"]
            pytorch_time = result["pytorch_sdpa"]["median_time_ms"]
            flash_time = result["flash"]["median_time_ms"]
            
            pytorch_speedup = result["pytorch_speedup"]
            flash_speedup = result["flash_speedup"]
            
            method_data["Naive (PyTorch)"]["times"].append((config_label, naive_time))
            method_data["Naive (PyTorch)"]["speedups"].append((config_label, "-"))
            
            method_data["PyTorch SDPA"]["times"].append((config_label, pytorch_time))
            method_data["PyTorch SDPA"]["speedups"].append((config_label, pytorch_speedup))
            
            method_data["+ Scores"]["times"].append((config_label, flash_time))
            method_data["+ Scores"]["speedups"].append((config_label, flash_speedup))
        
        # Print header
        config_labels = [label for label, _ in method_data["Naive (PyTorch)"]["times"]]
        header = f"{'Method':<45} "
        for label in config_labels:
            header += f"{label + ' (ms)':<12} {label + ' (speedup)':<12} "
        print(header)
        print("-" * 200)
        
        # Print each method as a row
        for method in method_names:
            row = f"{method:<45} "
            for i, config_label in enumerate(config_labels):
                time_val = method_data[method]["times"][i][1]
                speedup_val = method_data[method]["speedups"][i][1]
                
                if time_val == float("inf"):
                    time_str = "OOM/Error"
                else:
                    time_str = f"{time_val:.3f}"
                
                if speedup_val == "-":
                    speedup_str = "-"
                elif speedup_val == 0.0:
                    speedup_str = "N/A"
                else:
                    speedup_str = f"{speedup_val:.2f}x"
                
                row += f"{time_str:<12} {speedup_str:<12} "
            print(row)
    
    print("=" * 200)


def main():
    parser = argparse.ArgumentParser(description="Flash Attention with Scores Benchmark")
    parser.add_argument("--warmup-ms", type=int, default=50, help="Warmup time (ms)")
    parser.add_argument("--rep-ms", type=int, default=200, help="Repetition time (ms)")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs to use for statistics")
    parser.add_argument("--device", type=str, default="cuda", help="Device name")
    parser.add_argument("--no-sum-ops", action="store_true", 
                        help="Disable row_sum and col_sum benchmarks (enabled by default)")
    parser.add_argument("--enable-config-search", action="store_true",
                        help="Enable configuration search for optimal BLOCK_M, BLOCK_N, num_stages, num_warps")
    parser.add_argument("--config-search-trials", type=int, default=3,
                        help="Number of trials per configuration during search")
    parser.add_argument("--config-search-warmup", type=int, default=5,
                        help="Number of warmup iterations per configuration during search")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip correctness verification")
    parser.add_argument("--tolerance", type=float, default=1e-2,
                        help="Tolerance for correctness verification (default: 1e-2)")
    parser.add_argument("--split", type=int, default=768,
                        help="Split position for cross-token operators (default: 768)")

    args = parser.parse_args()
    
    # Qwen3 8B configuration
    # Add split parameter to each config
    test_configs = [
        {
            "name": "Qwen3 8B - 1K",
            "batch_size": 1,
            "num_heads_q": 32,
            "num_heads_k": 32,
            "seq_len_q": 1024,
            "seq_len_k": 1024,
            "head_dim": 128,
            "causal": True,
            "dtype": torch.float16,
            "split": args.split,
        },
        {
            "name": "Qwen3 8B - 2K",
            "batch_size": 1,
            "num_heads_q": 32,
            "num_heads_k": 32,
            "seq_len_q": 2048,
            "seq_len_k": 2048,
            "head_dim": 128,
            "causal": True,
            "dtype": torch.float16,
            "split": args.split,
        },
        {
            "name": "Qwen3 8B - 4K",
            "batch_size": 1,
            "num_heads_q": 32,
            "num_heads_k": 32,
            "seq_len_q": 4096,
            "seq_len_k": 4096,
            "head_dim": 128,
            "causal": True,
            "dtype": torch.float16,
            "split": args.split,
        },
        {
            "name": "Qwen3 8B - 8K",
            "batch_size": 1,
            "num_heads_q": 32,
            "num_heads_k": 32,
            "seq_len_q": 8192,
            "seq_len_k": 8192,
            "head_dim": 128,
            "causal": True,
            "dtype": torch.float16,
            "split": args.split,
        },
    ]
    
    run_benchmark_suite(
        test_configs=test_configs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        num_runs=args.num_runs,
        device=args.device,
        include_sum_ops=not args.no_sum_ops,
        enable_config_search=args.enable_config_search,
        config_search_trials=args.config_search_trials,
        config_search_warmup=args.config_search_warmup,
        verify_correctness_flag=not args.no_verify,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()

