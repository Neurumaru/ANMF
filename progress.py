import time


def progress(idx, max_idx, start_time):
    if idx == 0:
        print(f'\r{idx}/{max_idx} | TOTAL:{0:.2f}s | TIME:{0:.2f}s | LEFT:{0:.2f}s', end='')
    else:
        print(f'\r{idx}/{max_idx} | TOTAL:{(time.time() - start_time) / idx * max_idx:.2f}s | TIME:{time.time() - start_time:.2f}s | LEFT:{(time.time() - start_time) / idx * (max_idx - idx):.2f}s', end='')