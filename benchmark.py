"""
Benchmark script for Flash Attention with Scores
Compares Flash Attention, PyTorch SDPA, and naive implementations
"""

import sys
import os
import torch
import argparse
from typing import Dict, List

# Add parent directory to path so flash_attention_with_scores is imported as a package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    import triton
    from triton.testing import do_bench
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    raise ImportError("Triton is not available. Please install: pip install triton")

from flash_attention_with_scores import (
    attention_with_scores,
    attention_with_row_sum,
    attention_with_col_sum,
    attention_with_col_sum_sequential,
)
from reference_implementations import naive_attention_with_scores, pytorch_sdpa_attention_with_scores


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
    implementation: str = "flash"
) -> Dict[str, float]:
    """
    Benchmark attention_with_scores implementations
    """
    if device != "cuda":
        raise ValueError("do_bench only supports CUDA devices")
    
    try:
        # Create input tensors
        q = torch.randn(batch_size, num_heads_q, seq_len_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k, head_dim, dtype=dtype, device=device)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Select implementation
        if implementation == "flash":
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
    
    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k, head_dim, dtype=dtype, device=device)
        
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
    
    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k, head_dim, dtype=dtype, device=device)
        
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
    
    try:
        q = torch.randn(batch_size, num_heads_q, seq_len_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads_k, seq_len_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads_k, seq_len_k, head_dim, dtype=dtype, device=device)
        
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


def run_benchmark_suite(
    test_configs: List[Dict],
    warmup_ms: int = 50,
    rep_ms: int = 200,
    num_runs: int = 3,
    device: str = "cuda",
    include_sum_ops: bool = True,
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
            print("  [Flash Attention + Scores]")
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
                implementation="flash"
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
                print("  [Flash Attention + Row Sum]")
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
                
                print("  [Flash Attention + Col Sum (Reverse-Order)]")
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
                
                print("  [Flash Attention + Col Sum (Sequential)]")
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
                
                # Calculate speedups relative to naive
                naive_time = naive_result["median_time_ms"]
                row_sum_speedup = naive_time / row_sum_result["median_time_ms"] if naive_time != float("inf") and row_sum_result["median_time_ms"] != float("inf") else 0.0
                col_sum_speedup = naive_time / col_sum_result["median_time_ms"] if naive_time != float("inf") and col_sum_result["median_time_ms"] != float("inf") else 0.0
                col_sum_sequential_speedup = naive_time / col_sum_sequential_result["median_time_ms"] if naive_time != float("inf") and col_sum_sequential_result["median_time_ms"] != float("inf") else 0.0
                
                print(f"    Row Sum: {row_sum_result['median_time_ms']:.3f}ms ({row_sum_speedup:.2f} vs Naive), "
                      f"Col Sum (Reverse): {col_sum_result['median_time_ms']:.3f}ms ({col_sum_speedup:.2f} vs Naive), "
                      f"Col Sum (Sequential): {col_sum_sequential_result['median_time_ms']:.3f}ms ({col_sum_sequential_speedup:.2f} vs Naive)")
                
                results[-1]["row_sum"] = row_sum_result
                results[-1]["col_sum"] = col_sum_result
                results[-1]["col_sum_sequential"] = col_sum_sequential_result
                results[-1]["row_sum_speedup"] = row_sum_speedup
                results[-1]["col_sum_speedup"] = col_sum_speedup
                results[-1]["col_sum_sequential_speedup"] = col_sum_sequential_speedup
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Output summary table
    print("\n" + "=" * 200)
    print("Performance Summary (All operators, speedup relative to Naive)")
    print("=" * 200)
    if include_sum_ops:
        print(f"{'Batch':<6} {'Seq':<6} {'HeadsQ':<7} {'HeadsK':<7} {'HeadDim':<8} "
              f"{'Naive(ms)':<11} {'SDPA(ms)':<11} {'Flash(ms)':<11} {'RowSum(ms)':<12} {'ColSum-R(ms)':<14} {'ColSum-S(ms)':<14} "
              f"{'SDPA':<8} {'Flash':<8} {'RowSum':<9} {'ColSum-R':<11} {'ColSum-S':<11} {'Throughput(M/s)':<15}")
        print("-" * 200)
        
        for result in results:
            config = result["config"]
            naive_time = result["naive"]["median_time_ms"]
            pytorch_time = result["pytorch_sdpa"]["median_time_ms"]
            flash_time = result["flash"]["median_time_ms"]
            throughput = result["flash"]["throughput_tokens_per_sec"] / 1e6
            
            if "row_sum" in result and "col_sum" in result and "col_sum_sequential" in result:
                row_sum_time = result["row_sum"]["median_time_ms"]
                col_sum_time = result["col_sum"]["median_time_ms"]
                col_sum_sequential_time = result["col_sum_sequential"]["median_time_ms"]
                
                # Calculate speedups relative to naive
                pytorch_speedup = result["pytorch_speedup"]
                flash_speedup = result["flash_speedup"]
                row_sum_speedup = result.get("row_sum_speedup", 0.0)
                col_sum_speedup = result.get("col_sum_speedup", 0.0)
                col_sum_sequential_speedup = result.get("col_sum_sequential_speedup", 0.0)
                
                if naive_time == float("inf") or pytorch_time == float("inf") or flash_time == float("inf") or row_sum_time == float("inf") or col_sum_time == float("inf") or col_sum_sequential_time == float("inf"):
                    print(f"{config['batch_size']:<6} {config['seq_len_q']:<6} {config['num_heads_q']:<7} "
                          f"{config['num_heads_k']:<7} {config['head_dim']:<8} "
                          f"{'OOM/Error':<11} {'OOM/Error':<11} {'OOM/Error':<11} {'OOM/Error':<12} {'OOM/Error':<14} {'OOM/Error':<14} "
                          f"{'-':<8} {'-':<8} {'-':<9} {'-':<11} {'-':<11} {'-':<15}")
                else:
                    print(f"{config['batch_size']:<6} {config['seq_len_q']:<6} {config['num_heads_q']:<7} "
                          f"{config['num_heads_k']:<7} {config['head_dim']:<8} "
                          f"{naive_time:<11.3f} {pytorch_time:<11.3f} {flash_time:<11.3f} {row_sum_time:<12.3f} {col_sum_time:<14.3f} {col_sum_sequential_time:<14.3f} "
                          f"{pytorch_speedup:<8.2f} {flash_speedup:<8.2f} {row_sum_speedup:<9.2f} {col_sum_speedup:<11.2f} {col_sum_sequential_speedup:<11.2f} {throughput:<15.2f}")
            else:
                # Fallback if sum ops not available
                pytorch_speedup = result["pytorch_speedup"]
                flash_speedup = result["flash_speedup"]
                if naive_time == float("inf") or pytorch_time == float("inf") or flash_time == float("inf"):
                    print(f"{config['batch_size']:<6} {config['seq_len_q']:<6} {config['num_heads_q']:<7} "
                          f"{config['num_heads_k']:<7} {config['head_dim']:<8} "
                          f"{'OOM/Error':<11} {'OOM/Error':<11} {'OOM/Error':<11} {'N/A':<12} {'N/A':<14} {'N/A':<14} "
                          f"{'-':<8} {'-':<8} {'-':<9} {'-':<11} {'-':<11} {'-':<15}")
                else:
                    print(f"{config['batch_size']:<6} {config['seq_len_q']:<6} {config['num_heads_q']:<7} "
                          f"{config['num_heads_k']:<7} {config['head_dim']:<8} "
                          f"{naive_time:<11.3f} {pytorch_time:<11.3f} {flash_time:<11.3f} {'N/A':<12} {'N/A':<14} {'N/A':<14} "
                          f"{pytorch_speedup:<8.2f}x {flash_speedup:<8.2f}x {'-':<9} {'-':<11} {'-':<11} {throughput:<15.2f}")
    else:
        print(f"{'Batch':<6} {'Seq':<6} {'HeadsQ':<7} {'HeadsK':<7} {'HeadDim':<8} "
              f"{'Naive(ms)':<10} {'PyTorch(ms)':<14} {'Flash(ms)':<12} "
              f"{'PyTorch Speedup':<16} {'Flash Speedup':<14} {'Throughput(M/s)':<15}")
        print("-" * 140)
        
        for result in results:
            config = result["config"]
            naive_time = result["naive"]["median_time_ms"]
            pytorch_time = result["pytorch_sdpa"]["median_time_ms"]
            flash_time = result["flash"]["median_time_ms"]
            pytorch_speedup = result["pytorch_speedup"]
            flash_speedup = result["flash_speedup"]
            throughput = result["flash"]["throughput_tokens_per_sec"] / 1e6
            
            if naive_time == float("inf") or pytorch_time == float("inf") or flash_time == float("inf"):
                print(f"{config['batch_size']:<6} {config['seq_len_q']:<6} {config['num_heads_q']:<7} "
                      f"{config['num_heads_k']:<7} {config['head_dim']:<8} "
                      f"{'OOM/Error':<10} {'OOM/Error':<14} {'OOM/Error':<12} "
                      f"{'-':<16} {'-':<14} {'-':<15}")
            else:
                print(f"{config['batch_size']:<6} {config['seq_len_q']:<6} {config['num_heads_q']:<7} "
                      f"{config['num_heads_k']:<7} {config['head_dim']:<8} "
                      f"{naive_time:<10.3f} {pytorch_time:<14.3f} {flash_time:<12.3f} "
                      f"{pytorch_speedup:<16.2f} {flash_speedup:<14.2f} {throughput:<15.2f}")
    
    print("=" * 200 if include_sum_ops else "=" * 140)


def main():
    parser = argparse.ArgumentParser(description="Flash Attention with Scores Benchmark")
    parser.add_argument("--warmup-ms", type=int, default=50, help="Warmup time (ms)")
    parser.add_argument("--rep-ms", type=int, default=200, help="Repetition time (ms)")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs to use for statistics")
    parser.add_argument("--device", type=str, default="cuda", help="Device name")
    parser.add_argument("--no-sum-ops", action="store_true", 
                        help="Disable row_sum and col_sum benchmarks (enabled by default)")
    
    args = parser.parse_args()
    
    # Qwen3 8B configuration
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
        },
    ]
    
    run_benchmark_suite(
        test_configs=test_configs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        num_runs=args.num_runs,
        device=args.device,
        include_sum_ops=not args.no_sum_ops,
    )


if __name__ == "__main__":
    main()

