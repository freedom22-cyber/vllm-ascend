# IQuest-Q1 部署包总览 · 202611111

2026-11-11 的部署快照集合。本目录下每个子目录是一个**可独立拉起**的部署包，
覆盖两类形态：**PD 分离（Mooncake 池化）** 与 **TP16 混部（单实例跨机）**。

> 本文件只做索引和横向对比；每个包的可执行口径、逐条坑与实测数据，以各自
> `README.md` 为准（本目录所有脚本都是**全字面值**，没有 `env.sh` 变量层）。

---

## 1. 包索引

| 目录 | 形态 | 规模 | 推理入口 | 权重 | 状态（2026-11-11 12:30） |
|---|---|---|---|---|---|
| `pd-mooncacke-mxfp8-1p1d-2m/` | PD 分离 1P1D | 2 机（每侧 TP8+EP） | `http://10.222.46.181:1999` | `IQuest-Q1-W8A8C8-mxfp8` | **运行中**（11:44 拉起，proxy/healthcheck 200） |
| `pd-mooncacke-mxfp8-1p1d-4m/` | PD 分离 1P1D，DP2 | 4 机（每侧 2×TP8，DP=2） | `http://10.222.43.66:8999` | `IQuest-Q1-W8A8C8-mxfp8` | 未运行（机器被外部 4m 部署占用） |
| `vllm-iquest-bf16-1p1d-4m-20261110/` | PD 分离 1P1D，TP16 | 4 机（每侧 2 机 TP16） | `http://10.222.43.66:1999` | bf16 `IQuest-Q1` | 未运行（同上，机器冲突） |
| `vllm-iquest-bf16-tp16-mix-2m/` | 混部（非 PD） | 2 机，一个 TP16 实例 | `http://10.222.39.253:8005` | bf16 `IQuest-Q1` | 机器上有现场实例在服务（非本包脚本所建） |

> `vllm-iquest-bf16-1p1d-4m-20261110` 与 `pd-mooncacke-mxfp8-1p1d-4m` **共用同一批 4 台机**，
> 二者互斥；`pd-mooncacke-mxfp8-1p1d-2m` 已于 11-11 迁到 46.181/47.74，不再占用那 4 台。

---

## 2. 机器与端口矩阵

> ⚠️ **主机名不可靠**：`sjsbg-00004`、`sjsbg-00002` 等 hostname 在多台机器上重复出现，
> 判断机器**一律以 IP 为准**。

### 2.1 2m PD（`pd-mooncacke-mxfp8-1p1d-2m`）

| IP | 角色 | 容器 | 端口 |
|---|---|---|---|
| `10.222.46.181` | prefill + proxy + mooncake_master | `pd-mooncacke-p` | API `7100`、proxy `1999`、KV producer `28010`、master `50088` |
| `10.222.47.74` | decode + MTP | `pd-mooncacke-d` | API `7100`、KV consumer `29010` |

### 2.2 4m PD（两个 4m 包共用机器，互斥）

| IP | `pd-mooncacke-mxfp8-1p1d-4m` | `vllm-iquest-bf16-1p1d-4m-20261110` |
|---|---|---|
| `10.222.43.66` | prefill DP rank0，API `8000`，proxy `8999`，master `50090`，DP `16500` | prefill TP16 rank0，API `7100`，proxy `1999`，rendezvous `29563` |
| `10.222.44.142` | prefill DP rank1，API `8000` | prefill TP16 rank1（headless） |
| `10.222.42.240` | decode DP rank0，API `8000`，DP `16600` | decode TP16 rank0，API `7100`，master `50088`，rendezvous `29563` |
| `10.222.41.88` | decode DP rank1，API `8000` | decode TP16 rank1（headless） |

两边都用 KV 端口 `28010`(P)/`29010`(D) 和容器名 `iquest-mxfp8-p/d`（mxfp8 包）。

### 2.3 TP16 混部（`vllm-iquest-bf16-tp16-mix-2m`）

| IP | 角色 | 容器（本包口径） | 端口 |
|---|---|---|---|
| `10.222.39.253` | head，TP16 rank0 | `q1-tp16-mix-node1` | API `8005`、rendezvous `29563` |
| `10.222.34.2` | worker，TP16 rank1 | `q1-tp16-mix-node2` | headless（不监听） |

---

## 3. 怎么选

| 目标 | 用哪个包 |
|---|---|
| 要 PD 分离 + 低 TPOT（当前默认） | `pd-mooncacke-mxfp8-1p1d-2m` |
| 要 16 卡 EP 组 / 更大吞吐，愿意占 4 台机 | `pd-mooncacke-mxfp8-1p1d-4m` |
| 要 bf16 基线、复现 512k/TP16 历史数据 | `vllm-iquest-bf16-1p1d-4m-20261110` |
| 不要 PD、单入口、最小部署面 | `vllm-iquest-bf16-tp16-mix-2m` |

选型依据与实测对比见 `../../docs/IQuest-Q1_512k_性能测试报告.md` 与
`../../docs/IQuest-Q1_512k_经验沉淀.md`。

---

## 4. 通用约定（所有包一致）

1. **脚本全字面值**：IP/端口/路径直接写在脚本里，没有 `env.sh` 变量层；
   换机器时改 `docker_start.sh` 的 IP 常量和各角色脚本的 `local_ip` / `--host`。
2. **挂载口径**：
   - `本目录 → /pd`（容器内跑 `/pd/...` 的脚本）；`/pd` 必须是**真实 bind mount**，
     不要复用旧容器——旧容器的 bind 可能指向已删除/已移动的旧路径（看到的是旧快照）。
   - `vllm-ascend 代码仓 → /workspace`，例如
     `/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/vllm-ascend`。
     2m PD 包当前用的是该仓的 `releases/v0.30.0` worktree `pd-mooncacke-20261110`
     （同提交 `0b2829930` + 3 个本地部署补丁）。
   - `/mnt/sfs_turbo` 整盘 + 模型目录（通常 `:ro`）。
3. **不要在 `docker run` 里加 `-e PYTHONPATH=/workspace`**：会覆盖镜像注入的 acl 路径，
   spawn worker 报 `No module named 'acl'`。`PYTHONPATH` 必须在角色脚本里**追加**：
   `export PYTHONPATH="${PYTHONPATH:-}:/workspace"`。
4. **启动顺序**：`master → decode → prefill → proxy`（混部包为 `head → worker`）。
   - master 必须先起（引擎初始化即连池）；
   - **proxy 必须最后起**：P/D 脚本开头的残留清理块会 `pkill -9 python`，proxy 也是 python。
5. **日志落 SFS**：`docker exec -d` 会丢弃屏输出，拉起时一律重定向到
   `<包>/logs/{prefill,decode,proxy,mooncake}/...`。
6. **探活端点**：引擎 `/health`；PD proxy 是 **`/healthcheck`**（`/health` 会 404）。
7. **推理只走 proxy**：各引擎的 API 端口是 `kv_producer` / `kv_consumer` 角色，
   直连发请求不走 PD 编排，只适合 `/health`。
8. **NPU 独占**：一张 davinci 同一时刻只能被一个容器真正占用。起新站前先
   `npu-smi info -t proc-mem -i <id>` 确认 "No process in device."。

---

## 5. 共同坑（实踩）

| 症状 | 根因 | 处置 |
|---|---|---|
| 引擎没起，日志只有 `mkdir: ... File exists` | `/pd/logs/<role>` 是悬空软链 | 容器内 `ln -sfn <本目录>/logs/<role> /pd/logs/<role>`；脚本已容错 `mkdir -p … \|\| echo WARN` |
| `ModuleNotFoundError: No module named 'acl'` | `PYTHONPATH` 被整体覆盖（export 或 `docker run -e`） | 改追加式，且 `docker run` 不要带 `-e PYTHONPATH` |
| 有 `vllm serve` 进程但端口不监听、NPU 不占卡 | 上次启动死在 spawn 阶段的残留僵尸（“假绿”） | `pkill -KILL vllm; pkill -KILL VLLM`，确认 `proc-mem` 干净再重启 |
| 端口莫名被占 | 同机多容器共享 host 网络端口表 | 起之前 `ss -tlnp`；用 `/proc/<pid>/cgroup` 确认 pid 归属 |
| 清理命令把会话杀了 | 内联 `pkill -f 'vllm serve'` 命中执行它的 shell | 用脚本文件执行，或按 comm 匹配（`pkill -9 VLLM`，不带 `-f`） |
| 容器里的 `/pd` 内容很旧 | 容器是旧的，bind 源目录已被移动/删除（docker 按 inode 保持旧视图） | 用 `docker_start.sh` 重建容器，不要复用旧容器 |
| 日志时间差 8 小时 | 容器内 UTC，宿主 CST+8 | 换算后再对时间线 |
| decode 起不来，报 "no config.json" | MTP 草稿 `/models/IQuest-Q1/mtp` 未挂载 | 按各包 `docker_start.sh` 的挂载清单补齐 |

---

## 6. 常用命令

```bash
ROOT=/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/202611111

# ---- 2m PD（当前主推）----
TS=$(date +%Y%m%d_%H%M%S); D=$ROOT/pd-mooncacke-mxfp8-1p1d-2m
ssh root@10.222.46.181 'docker exec -d pd-mooncacke-p bash /pd/mooncake/run_master.sh'
ssh root@10.222.47.74  "docker exec -d pd-mooncacke-d bash -c 'bash /pd/decode/d.sh  >> $D/logs/decode/engine-$TS.log 2>&1'"
ssh root@10.222.46.181 "docker exec -d pd-mooncacke-p bash -c 'bash /pd/prefill/p.sh >> $D/logs/prefill/engine-$TS.log 2>&1'"
ssh root@10.222.46.181 "docker exec -d pd-mooncacke-p bash -c 'bash /pd/run_proxy.sh   >> $D/logs/proxy/proxy-stdout.log 2>&1'"

# 探活
curl -s -o /dev/null -w 'P %{http_code}\n' http://10.222.46.181:7100/health
curl -s -o /dev/null -w 'D %{http_code}\n' http://10.222.47.74:7100/health
curl -s -o /dev/null -w 'X %{http_code}\n' http://10.222.46.181:1999/healthcheck

# 冒烟（走 proxy）
curl -s http://10.222.46.181:1999/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":512,"temperature":0}'

# 停止（各机容器内，按 comm 匹配）
docker exec pd-mooncacke-p bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
docker exec pd-mooncacke-d bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
docker exec pd-mooncacke-p pkill -f load_balance_proxy_server_example
docker exec pd-mooncacke-p pkill -f '[m]ooncake_master'
```

其余包的拉起/停止命令见各自 `README.md` 的第二节（拓扑与端口）与启动章节。

---

## 7. 状态快照（2026-11-11 12:30，仅供当天参考）

| 站点 | 观测结果 |
|---|---|
| 2m PD（46.181 / 47.74） | **全绿**：P/D `/health` 200、proxy `/healthcheck` 200、两次冒烟通过（冷启 30 s，二次 0.59 s） |
| TP16 混部（39.253 / 34.2） | `39.253:8005/health` 200（容器 `q1-tp16-ziang-ar-mooncake-2tib-node1`，现场既有实例，非本目录脚本所建） |
| 4m 机器（43.66 / 44.142 / 42.240 / 41.88） | 被外部 TP16 部署占用：容器 `iquest-pd-mooncacke-p/d`，`/pd` 指向 `/mnt/sfs_turbo/huawei/zuoqian/...`；本次检查未见 `7100/7101/1999` 对外监听，使用前需确认归属与状态 |

> 状态随时会变。改完部署请顺手更新本表，或至少保留观测时间。

---

## 8. 维护约定

- 新增部署包：目录名用 `<形态>-<量化/精度>-<并行>-<机数>`，例如
  `pd-mooncacke-mxfp8-1p1d-2m`；必须包含 `README.md`、`docker_start.sh`、`run_proxy.sh`（PD 包）、
  `logs/`，并在本文件的 §1 索引和 §2 机器矩阵里登记。
- 换机器：先改**本文件矩阵**，再改包内 `docker_start.sh` 的 IP 常量与角色脚本的
  `local_ip`/`--host`/proxy 后端，最后重建容器验证。
- 不要在已运行的容器上原地改脚本后直接重跑：`p.sh`/`d.sh` 会 `pkill python`，
  且旧容器可能挂着旧路径。
- 历史包（11-10 及更早）在 `../` 下，未删除；判断现行口径以本目录和
  `../../docs/` 的最近报告为准。