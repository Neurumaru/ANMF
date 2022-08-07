import time


def progress(idx, max_idx, start_time):
    print(f'\r{idx}/{max_idx} | TOTAL:{(time.time() - start_time) / (idx + 1) * max_idx:.2f}s | TIME:{time.time() - start_time:.2f}s | LEFT:{(time.time() - start_time) / (idx + 1) * (max_idx - idx - 1):.2f}s     ', end='')