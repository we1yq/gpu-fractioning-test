# GPU Fractioning Benchmark

比较 GPU Fractioning、传统 CUDA MPS 和 MIG 的性能开销、资源利用率与隔离性。

## 目录

```
bench/      合成负载，输出延迟分布 JSON
k8s/        Pod 模板（fractioning / mps / mig）
run.sh      跑实验矩阵的一格：部署 Pod、等待完成、收集结果
analyze/    汇总 P50/P99 和干扰系数（作图待第 5 步补充）
```

## 负载

| 负载 | 目标 | 默认规模 | 显存占用 |
|---|---|---|---|
| `gemm` | SM 竞争 | 4096 × 4096 | 约 192 MiB |
| `membw` | 显存带宽 | 每个数组 680 MiB | 约 2 GiB |
| `smallkernel` | 调度开销 | 256 × 256 | 很小 |
| `hostcopy` | PCIe | 每个方向 256 MiB | 约 256 MiB |
| `alloccycle` | 分配路径 | 每次 64 MiB | 约 64 MiB |

默认规模要能放进最小的切片（`1g.5gb`），这样所有配置跑的都是同一负载。
正式实验前，先单独校准一次 `--size`。

## 集群访问

在服务器上运行一次，之后 `run.sh` 不需要 `sudo`：

```bash
./k8s/setup-access.sh
```

它会安装 `k8s/rbac.yaml`，并写入 `~/.kube/config`，已有该文件时会拒绝覆盖。
这个账号只能操作 `gpu-bench` 里的 Pod、Pod 日志和 ConfigMap，并只读
`GpuFractioningConfig`。恢复或关闭 Fractioning 仍需管理员权限。

## 运行

先确认同事已结束 GPU 工作，再加 `--gpu-confirmed`。长时间运行放在 `tmux` 里，
SSH 断开时 `run.sh` 会退出。

```bash
# 只渲染 manifest，不接触集群
./run.sh --dry-run --mode fractioning --image IMAGE --workload gemm -n 2

# 同构并发
./run.sh --gpu-confirmed --mode fractioning --compute time-slicing \
  --image IMAGE --workload gemm -n 2 --reps 5

# 异构干扰：Pod 0 是受害者，开环 100 请求/秒，其余 3 个是吵闹邻居
./run.sh --gpu-confirmed --mode fractioning --compute sm-sharing \
  --image IMAGE --workload smallkernel --rate 100 --noisy gemm -n 4 --reps 5

# MIG：分区必须已存在
./run.sh --gpu-confirmed --mode mig --mig-profile 3g.20gb \
  --image IMAGE --workload gemm -n 2 --reps 5

python3 analyze/summarize.py results -o results/summary.csv
```

完整矩阵用 `matrix.sh`，在 `tmux` 里运行。用相同的 `--results` 重跑就能续跑：

```bash
./matrix.sh --list
./matrix.sh --gpu-confirmed --image IMAGE --results results/v2 --reps 1
python3 analyze/summarize.py results/v2 -o results/v2-summary.csv
python3 analyze/capacity.py results/v2
```

实验组和判定标准见 [docs/experiment-design.md](docs/experiment-design.md)。C 组（传统 MPS）
要切换集群模式，由 `k8s/run-legacy-group.sh` 完成，退出时自动恢复，必须在 `tmux` 里运行。

`run.sh` 只创建自己的 Pod 和一个 ConfigMap，不会改动 `GpuFractioningConfig`、
device plugin、Helm Release 或 MIG 分区。代码通过 ConfigMap 挂载，
镜像只需要带 PyTorch。

## 测量方法

- **同步计时**：每步之后调用 `torch.cuda.synchronize()`，否则测到的只是提交耗时。
- **同时开始**：同一轮的 Pod 共用 `--start-at`，汇总时检查实际测量窗口的重叠度，
  迟到或重叠不足 95% 的整轮会被剔除。
- **闭环与开环**：闭环（默认）测饱和吞吐；`--rate` 开启泊松到达的开环模式，
  延迟从计划到达时刻算起，避免 coordinated omission。受害者的 P99 应使用开环。
- **相同输入**：种子只由轮次和 Pod 序号决定，各配置看到的输入相同。
- **独占 CPU**：`synchronize()` 会占满一个核，模板里 CPU 的 request 等于 limit，
  避免把 CPU 争用误判成 GPU 争用。
- **干扰系数**：相对同一配置的 n=1 基线。跨模式的开销比较，用各模式 n=1 的结果。
- **额外退化**：`p50_excess` 扣除公平份额，n 个 Pod 平分 GPU 时为 1.0，超出部分才是
  开销或争用。
- **定位干扰来源**：交叉矩阵让每个负载既当受害者又当邻居。比如 `hostcopy` 只被
  `hostcopy` 拖慢，说明争用在 PCIe。`time-slicing` 下 Pod 轮流占用 GPU，这种区分
  主要在 `sm-sharing` 下有效。
- **遥测**：每轮保存 `dmon.txt`（SM%、内存控制器、PCIe、时钟、功耗）和
  `clocks.csv`（降频原因）。`dcgmi` 的 SM/DRAM 指标在 MPS 开启时读取失败，
  内存控制器忙碌度只是显存带宽的代理指标。
- **同一张卡**：每个 Pod 记录 GPU UUID，不一致的整轮会被剔除。
- **CPU 位置**：Pod 固定到 GPU 所在的 NUMA 0，线程数等于 CPU 配额，避免跨插槽和
  CFS 限流影响延迟。

## 结果

每个 Pod 输出一个 JSON：`summary`（吞吐、P50/P90/P99/P99.9）、`histogram`
（对数分桶）、环境信息和显存峰值。显存不足时 `status` 为 `oom`，退出码为 2，
这种情况算实验结果，不算错误。
