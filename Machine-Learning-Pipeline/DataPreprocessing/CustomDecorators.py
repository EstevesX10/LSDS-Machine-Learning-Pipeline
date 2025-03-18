import time
import io
import cProfile
from functools import wraps
import psutil
import pstats

def timeit(func):
    """
    # Description
        -> A decorator to measure wall-clock time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save the start time
        start = time.time()

        # Compute the output of the function
        result = func(*args, **kwargs)

        # Print the execution time
        print(f"[{func.__name__}] executed in {(time.time() - start): .3e}(s)")
        
        # Return the result of the function
        return result
    return wrapper

def resourceProfiler(func):
    """
    # Description
        -> A decorator to measure memory usage, CPU usage, wall-clock time, 
        and function call profiling (cProfile) for the decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()

        # Record initial values
        mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
        cpu_times_before = process.cpu_times()
        start_time = time.time()

        # Enable cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        # Execute the function
        result = func(*args, **kwargs)

        # Disable cProfile
        profiler.disable()
        end_time = time.time()

        # Record final values
        mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
        cpu_times_after = process.cpu_times()

        # Compute differences
        wall_time = end_time - start_time
        mem_diff = mem_after - mem_before
        user_cpu_time = cpu_times_after.user - cpu_times_before.user
        system_cpu_time = cpu_times_after.system - cpu_times_before.system

        # Print summary
        print("=" * 40)
        print(f"Function: {func.__name__}")
        print(f"Wall time       : {wall_time:.4f} s")
        print(f"User CPU time   : {user_cpu_time:.4f} s")
        print(f"System CPU time : {system_cpu_time:.4f} s")
        print(f"Memory before   : {mem_before:.2f} MB")
        print(f"Memory after    : {mem_after:.2f} MB")
        print(f"Memory diff     : {mem_diff:.2f} MB")
        print("=" * 40)

        # Format cProfile stats, removing absolute paths
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()

        print("cProfile Stats (Cumulative):")
        print(s.getvalue())

        return result
    return wrapper