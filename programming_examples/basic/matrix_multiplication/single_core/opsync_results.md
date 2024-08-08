# Comparing Sync Overheads

See also `opsync_trace_old.json` and `opsync_trace_new.json`.

Compilation &amp; benchmarking command:
```
M=4096 K=64 N=64 m=64 k=64 n=64 dtype_out=i16 runargs="--trace_sz 131072 --iters 100 --warmup 10 --verify 0 -v 1" trace_size=131072 make run
```

New: as of this branch at commit `57096aeeef58b87e34a5000095c65b81839e87c8` + changes to enable tracing
Old: as of `main` at commit `b8a489095b594bd3fdff65487cd2b5c92050a907` + changes to enable tracing

## New -- Interleaved Syncs

```
Avg NPU matmul time: 882.32us.
Avg NPU gflops: 38.0298

Min NPU matmul time: 621us.
Max NPU gflops: 54.0329

Max NPU matmul time: 2956us.
Min NPU gflops: 11.3513

PASS!
```

## New -- Interleaved Syncs -- Running `./keep_busy` in Parallel

```
Avg NPU matmul time: 754.39us.
Avg NPU gflops: 44.4789

Min NPU matmul time: 631us.
Max NPU gflops: 53.1766

Max NPU matmul time: 1452us.
Min NPU gflops: 23.1091

PASS!
```

## Old -- Non-Interleaved Syncs

```
Avg NPU matmul time: 1462.49us.
Avg NPU gflops: 22.9434

Min NPU matmul time: 828us.
Max NPU gflops: 40.5247

Max NPU matmul time: 3590us.
Min NPU gflops: 9.34664

PASS!
```

## Old -- Non-Interleaved Syncs -- Running `./keep_busy` in Parallel

```
Avg NPU matmul time: 924.14us.
Avg NPU gflops: 36.3088

Min NPU matmul time: 761us.
Max NPU gflops: 44.0926

Max NPU matmul time: 1884us.
Min NPU gflops: 17.8102

PASS!
```
