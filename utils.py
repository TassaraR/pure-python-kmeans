import time


def measure_runtime(data, algorithm, name='', verbose=False):
    start_time = time.time()
    algorithm.fit(data)
    runtime = time.time() - start_time
    if verbose:
        print(f'{name} took: {runtime:.2f} Seconds to run')
    return algorithm, runtime

