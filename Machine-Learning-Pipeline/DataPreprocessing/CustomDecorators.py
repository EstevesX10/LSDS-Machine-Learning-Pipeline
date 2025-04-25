import time
import sys
import time
import psutil
from functools import wraps

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

def profileResources(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        stats = {}     # key -> {'cum_time','user_cpu','kernel_cpu','calls','mem'}
        children = {}  # parent_key -> set(child_key)
        stack = []     # (key, t0, u0, k0, m0, depth)
        process = psutil.Process()

        def _profiler(frame, event, arg):
            if event == 'call':
                code = frame.f_code
                key = (code.co_name, code.co_filename, code.co_firstlineno)
                depth = len(stack)
                if stack:
                    parent = stack[-1][0]
                    children.setdefault(parent, set()).add(key)
                stats.setdefault(key, {'cum_time':0.0,'user_cpu':0.0,'kernel_cpu':0.0,'calls':0,'mem':0.0})
                times = process.cpu_times()
                stack.append((key,
                              time.perf_counter(),
                              times.user,
                              times.system,
                              process.memory_info().rss,
                              depth))
            elif event == 'return' and stack:
                key, t0, u0, k0, m0, depth = stack.pop()
                dt = time.perf_counter() - t0
                times = process.cpu_times()
                du = times.user - u0
                dk = times.system - k0
                mem_mb = (process.memory_info().rss - m0) / (1024*1024)
                rec = stats[key]
                rec['cum_time']   += dt
                rec['user_cpu']   += du
                rec['kernel_cpu'] += dk
                rec['calls']      += 1
                rec['mem']        += mem_mb  # allow negative

        # Profiling start
        sys.setprofile(_profiler)
        t_start = time.perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            total_time = time.perf_counter() - t_start
            sys.setprofile(None)
            util_end = psutil.cpu_percent(interval=None, percpu=True)

            # Create the hierarchy calls
            rows = []
            def traverse(key, depth=0):
                rec = stats.get(key)
                if rec is None:
                    return
                rows.append({
                    'key': key,
                    'depth': depth,
                    'cum_time': rec['cum_time'],
                    'user_cpu': rec['user_cpu'],
                    'kernel_cpu': rec['kernel_cpu'],
                    'mem': rec['mem'],
                    'calls': rec['calls']
                })
                for child in sorted(children.get(key, []), key=lambda k: stats[k]['cum_time'], reverse=True):
                    traverse(child, depth+1)

            root = (func.__name__, func.__code__.co_filename, func.__code__.co_firstlineno)
            traverse(root)

            # Define the header columns
            header_cols = [
                'Function',
                'Total Call Count',
                'CPU Time',
                'User CPU Time',
                'Kernel CPU Time',
                'Memory Change (MB)'
            ]
            header = ' | '.join(f"{c:>25}" if i>0 else f"{c:<30}" for i,c in enumerate(header_cols))
            sep = '-' * len(header)

            print(f"\n[Total Time]: {total_time:.4f} s")
            core_utilization_str = "(" + ", ".join([f"{elem}%" for elem in util_end]) + ")"
            print(f"[CPU utilization per core]: {core_utilization_str}")
            print(sep)
            print(header)
            print(sep)
            for r in rows:
                name, _, lineno = r['key']
                indent = '  ' * r['depth']
                label = f"{indent}{name} (L{lineno})"
                
                # Compute percentages
                time_pct = (r['cum_time'] / total_time) * 100 if total_time else 0
                user_pct = (r['user_cpu'] / total_time) * 100 if total_time else 0
                kern_pct = (r['kernel_cpu'] / total_time) * 100 if total_time else 0
                
                # Format using exponentital notation
                mem_val = r['mem']
                if abs(mem_val) < 0.01:
                    mem_str = f"{mem_val:.2e}"
                else:
                    mem_str = f"{mem_val:.2f}"
                if mem_val >= 0:
                    mem_str = "(+) " + mem_str
                else:
                    mem_str = "(-) " + mem_str
                vals = [
                    label,
                    str(r['calls']),
                    f"{r['cum_time']:.4f} s ({time_pct:.1f}%)",
                    f"{r['user_cpu']:.4f} s ({user_pct:.1f}%)",
                    f"{r['kernel_cpu']:.4f} s ({kern_pct:.1f}%)",
                    mem_str
                ]
                row = ' | '.join(f"{v:>25}" if i>0 else f"{v:<30}" for i,v in enumerate(vals))
                print(row)
            print(sep)
    return wrapper