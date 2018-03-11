"""
Script for profiling the ik-rvea runs

As we are using joblib, we cannot profile scripts directly, they must be used as modules


USAGE

For memory_profiler::

    pip install memory_profiler
    mprof run -T 1 -C
    mprof plot

"""
if __name__ == '__main__':
    import iKRVEA
    iKRVEA.main(ikrvea_path='../ik-rvea', clear=False)
