# C8-MXFP KV Cache × PD 分离 适配问题记录

日期：2026-11-09/10 ｜ 环境：new2(10.222.42.240, P0) / new8(10.222.41.88, D0)
代码：`vllm-ascend-prof` worktree，镜像 `vllm-ascend:v0.30.0-a5-openeuler`
结论：**C8 未打通**，当前部署回退为 bf16 KV；本文记录已定位的全部问题与剩余工作。

---

## 1. 目标配置

```bash
--kv-cache-dtype mxfp8
--kv-cache-dtype-skip-layers sliding_window
--block-size 512                     # C8 强制 512 的倍数
# 不加 --disable-hybrid-kv-cache-manager（C8 依赖混合 KV 管理器）
```

IQuest-Q1 层结构（config.json）：`[full, sliding, sliding, sliding] × 21 + 4 = 88` 层，
即 22 个 full 层 + 66 个 sliding 层（sliding_window=4096）。C8 只适用于 full 层；
sliding 层必须保持 bf16（C8-MXFP QFA 路径不支持 sliding window）。

---

## 2. 已解决（配置 / 环境侧）

| # | 问题 | 现象 | 处理 |
|---|---|---|---|
| 1 | `MooncakeHybridConnector` 不支持 C8 四元组 | `ValueError: too many values to unpack (expected 2)` @ `mooncake_hybrid_connector.py:1810` | 换 **`MooncakeConnectorV1`**（框架为 C8 指定的连接器，`attention_c8_mxfp.py:905` 明确要求；其注册按 storage 合并重叠视图） |
| 2 | `--block-size 128` 与 C8 冲突 | `ValueError: C8_MXFP requires --block-size to be a multiple of 512` | 改 512 |
| 3 | `CPLUS_INCLUDE_PATH` 指向 CANN 交叉编译头 | torch 扩展 JIT 失败：`redefinition of 'struct std::numeric_limits'`（quant_flash_attn / apply_rotary_pos_emb / block_sparse_attention） | 删除该 export（它只为 AICPU JIT，全局设置会毒化 g++ 编译） |
| 4 | `PYTHONPATH=/workspace` 放在 source CANN 之后 | `ModuleNotFoundError: No module named 'acl'` | 移到 source 之前（toolkit 是前置追加，不是覆盖） |
| 5 | CANN env 脚本引用未定义变量 | `ZSH_VERSION: unbound variable`（set -u 下） | source 前后 `set +u` / `set -u` |
| 6 | decode 侧 KV 配置被误写为 producer | P/D 角色错 | 修正为 `kv_consumer` / 29010 / engine_id 1 |
| 7 | `--disable-hybrid-kv-cache-manager` 与 C8 冲突 | 见 §3.1 | 不传该开关 |

---

## 3. 未解决（框架侧）

### 3.1 规格构建未尊重 per-layer KV dtype（**已定位，补丁已验证可编译**）

`vllm_ascend/worker/v2/attn_utils.py` 原 334 行：

```python
if type(spec) is FullAttentionSpec and is_c8_mxfp_kv_quant(vllm_config):
    spec = mxfp_cache_spec(spec)          # 无条件转 C8
```

`--kv-cache-dtype-skip-layers`（vLLM `attention.py:297-310`）只改该层
`kv_cache_torch_dtype`，不改 spec → 滑窗层被分配 FP8 页，但 impl 走 bf16 scatter
→ `aclnnScatterPaKvCache EZ0021: key=DT_BFLOAT16, keyCacheRef=DT_FLOAT8_E4M3FN`。

补丁：仅当 `attn_module.kv_cache_torch_dtype == torch.float8_e4m3fn` 才转换；
视图构建侧（原 1580 行）同样以 `kv_cache_spec.dtype == float8_e4m3fn` 为条件。
**该补丁编译通过、C8 只落在 full 层（诊断日志证实），但引出 3.2。**

### 3.2 混合池的 padded 页几何不一致（**阻塞点**）

打开混合 KV 管理器后，实测（`[KVDBG]` / `[C8DBG]` 日志）：

| 层 | dtype | 规格 | block | 每页字节 |
|---|---|---|---|---|
| full(0,1,5,9,13,17,21…) | fp8 | FullAttentionSpec | **1024**（512 被对齐放大） | 270,336 |
| sliding(2,3,6,7…) | bf16 | SlidingWindowSpec | 128 | padding 到 270,336 |

滑窗层两种分配形态都存在：
- 合并形态：`k_numel == v_numel == 237,760,512`（K/V 同一 buffer），1759 页
- 独立形态：`k_numel == v_numel == 118,880,256`，1759 块

`_reshape_kv_cache_v2` 的 tuple 分支直接 `raw.view(dtype).view(k_shape)`，
无法表达 padding，依次报：

```
RuntimeError: shape '[1759, 128, 1, 128]' is invalid for input of size 118880256
RuntimeError: setStorage: sizes [14072, 128, 1, 128], strides [16896, ...] 超限
```

根因：`num_blocks_per_kv_block = get_storage_block_size(spec)//kernel_block_size`
对滑窗层算出 1，而物理页实际容纳 2~4 个 kernel 块 + 每块 1,024 B padding；
且各层 padding 排布不统一（有的 K/V 合并、有的独立）。

**已尝试并回退**（均不完整）：
1. 按 `raw.numel()//blocks` 推跨度 → 拿到的是页跨度而非 kernel 块跨度，申请 476 MB，失败
2. 复用 `_reshape_combined_attention_kv_cache`（同基址情形）→ 部分层通过，独立层仍失败
3. 通用 `per_kv = per_block // dense` + `as_strided` → 算术在 118,880,256 上闭合，
   但另一层 `per_kv=2` 时 span 超出分配（14072×16896）

### 3.3 建议的修复方向（交给框架 owner）

- 让 **spec 构建侧**与 `_align_hybrid_attention_page_sizes` 对齐：目前该函数只在
  `mamba_specs` 存在时调用（`attn_utils.py:341`），纯 attention 的混合池未覆盖；
  但 C8(270,336B) 与 bf16(65,536B) 页并不整除，需扩展为「按 token 数对齐 + 记录
  kernel_blocks_per_page」而非仅按字节整除。
- 或在 `KVCacheConfig` 里为混合池显式携带
  `kernel_blocks_per_page` / `page_stride_bytes`，reshape 侧直接用，不再反推。
- 分配侧保证同一 layer 的 K/V 视图布局唯一（合并或独立二选一并写进 spec），
  避免 reshape 需要猜。

### 3.4 连带项（C8 打通后再验）

- **MTP 草稿层**：无 `v_cache_scale`，同样需要排除出 C8（bf16）；当前 decode 用
  `--speculative-config` 关闭状态验证。
- **图模式**：`torch.compile` 路径下 C8 impl 未生效（eager 正常），需复核
  `IQuestQ1AscendAttention.get_impl_cls` 与 quant method 的 `__class__` 交换时序。

---

## 4. 当前生效配置（bf16 基线）

两侧 `launch_static.sh`：bf16 KV、`--block-size 128`、
`--disable-hybrid-kv-cache-manager`、`MooncakeConnectorV1` + `AscendStoreConnector`、
eager、MTP 关闭。P0 端口 **7200**（proxy 需 `--prefiller-ports 7200`）。

---

## 5. 经验

1. 启动参数若用 `\` 续行，**注释行绝不能插在续行之间**——会被拼进上一行并吞掉后续全部参数。
   自检：`awk '/^exec /{i=1} i && /^[[:space:]]*#/{print NR}' script.sh`
2. `/health` 200 不代表可用；必须核对 `/proc/PID/cmdline` 并跑真实请求。
3. 请求 200 也不代表 PD 生效；查两侧吞吐日志与 `no KVConnector found` 告警。

---

## 6. 性能调优记录（bf16 基线 + 参考参数适配，2026-11-10）

压测口径：`vllm bench serve`，random ISL=1024 / OSL=256，经 proxy 1999，`--ignore-eos`。

| 配置 | 并发 | 总吞吐 tok/s | TTFT 中位 | P99 TTFT | TPOT 中位 |
|---|---|---|---|---|---|
| D seqs=16 / P batch=16k / fused_mc2=2(被静默关闭) | 16 | 3953 | 1046 ms | 3197 ms | 11.56 ms |
| 同上 | 32 | **5469** | 3344 ms | 4759 ms | 12.20 ms |
| D seqs=32 | 32 | 5268 | 3436 ms | 5169 ms | 12.36 ms |
| P batch=32768 | 32 | 4751 | 3394 ms | 7075 ms | 12.33 ms |
| P 恢复 fused_mc2=1（去掉 multistream） | 32 | 4684 | 3350 ms | 7152 ms | 12.53 ms |

结论：
- TPOT 稳定 11.5-12.5 ms（MTP+图模式生效，优于历史优化档 20.2 ms）
- **TTFT 在 c32 恒定 ~3.3 s，与 P 侧任何参数无关** → P prefill 已饱和
- D 侧并发 16→32 无收益；P batch 16k→32k 劣化（P99 TTFT 7.1 s）
- 每请求 KV 传输固定开销 ~300 ms（14 token 的请求也是这个数）

### 关键坑：参考参数直接照搬会降级
```
WARNING [ascend_config.py:869] MegaMoe is not supported for this model config;
        additional_config.enable_fused_mc2 will be set to 0
```
参考脚本的 `enable_fused_mc2: 2` 在 IQuest 上被整条关闭；原部署的 **1**
（dispatch_ffn_combine）才有效。且 `multistream_overlap_shared_expert` 与
`enable_fused_mc2` 互斥（P 侧参考脚本本就没有该项）。

### V1 连接器的额外约束
```
RuntimeError: Remote KV metadata does not contain layer
'model.layers.88.mtp_model_layer.self_attn.attn'
```
`MooncakeConnectorV1` 按层名对齐 KV 元数据 → **P 侧也必须带 `--speculative-config`**
（参考脚本里 P 同样带），否则 D 的 MTP 草稿层在 P 侧无对应元数据。

### 下一步候选
1. P 侧 prefill 并行度：参考部署用 **PP2 跨两机**（4 机形态），单机 TP8 已是上限
2. 降低每请求 KV 传输固定开销（300 ms）：批量合并 / 减小握手开销
3. C8 打通后 KV 传输量减半，TTFT 与吞吐都会受益
