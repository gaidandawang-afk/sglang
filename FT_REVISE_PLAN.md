# DP-only FT 整改追踪

基线：`codex/dp-only-ft-squashed` @ `74cafe366`
工作区：`.claude/worktrees/dp-only-ft-revise`（分支 `worktree-dp-only-ft-revise`）

原则（与 OWNER 对齐）：
- 不无脑删代码；删的是**不必要的冗余**和**历史反复修改的残渣**。
- 接受"好的设计且不显著增加代码"的重构；目标合入开源社区，设计与风格并重。
- 日志最多一行、try 占比要低（贴原生 e63 EEP）、入参校验抽函数、状态派生抽函数、列表式写法走紧致风格。
- 所有 FT 相关日志只允许占用一行，不保留仅用于定位的日志信息。
- **重构边界 = 只动 Codex 提交的代码**（FT 自有代码）。他人负责的函数（elastic_ep/mooncake 等）即使看似死代码也原样保留，越界改动退回最小必要。判断存疑先问 OWNER。

状态图例：`[ ]` 未开始 / `[~]` 进行中 / `[x]` 完成待 review / `[OK]` review 通过

---

## 第 0 批 — 越界退让（elastic_ep.py，最优先）

| # | 位置 | 问题 | 改法 | 状态 |
|---|------|------|------|------|
| 0.1 | `elastic_ep.py:141` | 顺手删了他人的 `_get_process_group_backend` | **还原删除**，不碰（即使无调用者，非我方代码保留） | [x] |
| 0.2 | `elastic_ep.py:179` | `_refresh_ep_members` 引入 `MOONCAKE_EP_FORCE_FALLBACK` 分支 + `raise` | 收缩为单行守卫 `if EPBuffer._buffer is not None: EPBuffer._buffer.update_ep_member()`，删掉 env 分支与 raise | [x] |

---

## 第 1 批 — controller.py 写法/风格（已 review，指令明确）

| # | 位置 | 问题 | 改法 | 状态 |
|---|------|------|------|------|
| 1.1 | `controller.py:25` `is_ft_supported_config` | 一串硬编码 if-return，新增 gate 要插进瀑布 | 写成**一组 gate 表**（条件→错误码的可迭代），单函数遍历产出第一个失败；行为/错误码不变 | [x] |
| 1.2 | `controller.py:65` `status_response` | 嵌套三元推导 rank state，难看 | **状态派生抽单独函数** `_rank_state(rank) -> RankState`，body 线性 if | [x] |
| 1.3 | `controller.py:111` `pause_targets` | 多行列表推导 | 改紧致写法，对齐 `resume_targets` 单行风格 | [x] |
| 1.4 | `controller.py:125` `_normalize_mask` | 被质疑冗余 | **判断：不冗余**（mask 长度对齐 dp_size + 布尔化的边界整形），保留；挂起到 3.1 随 `_update_availability` 一并处理 | [~]→3.1 |

---

## 第 2 批 — manager.py 风格/纪律（已 review，横向要求）

| # | 位置 | 问题 | 改法 | 状态 |
|---|------|------|------|------|
| 2.1 | `manager.py:77` | 入参校验（ranks 是否 list/int 化、shutdown/params 冲突）混在 apply 主体 | **抽独立校验函数** `_parse_apply_args`，返回 `(ranks, timeout, error)`；apply 主体只调一次 | [x] |
| 2.2 | `manager.py:105` | apply 日志多行 | **压成一行**，砍掉 `active_mask`（`effective_active_mask()` 派生噪声），对齐原生 e63 EEP 日志密度 | [x] |
| 2.3 | `manager.py:113`（横向） | try/except 占比过高 | 全 FT 扫一遍。**降 2 处**：`_publish_active_ranks` 删 TimeoutError 改名（让原生抛）、`_send_command_collect` 删超时 4 行 warning（留 finally 清 pending）。**保留 5 处**：输入整形、apply 发布补偿、apply resume failstop、`_fatal_task_wrapper` 杀进程——均为分布式事务/安全必需 | [x] |

---

## 第 3 批 — 结构性重构（方案已讨论并贴回 PR，需 OWNER 逐条批）

> 这批会引入新结构，与"少代码"纪律存在张力，故**逐条待批**，批一条做一条。

| # | 位置 | 方案 | 规模 | 状态 |
|---|------|------|------|------|
| 3.1 | `controller.py:128` `_begin_availability_pause` 链 | 抽 `_try_begin_pause()` 公共互斥骨架；`observe_*` 经 `_update_availability(mutate)` 收编快照；`_falling_edge` 抽出；`begin_exception_pause` 退成一行调骨架。**净效应：删两套复制三段式，约不增代码** | 小 | [x] |
| 3.2 | `controller.py:161` `pending/mark/take` | 三接口 → `get_unpublished_effective_active_mask()` + `mark_effective_active_mask_published(mask)`；删 `take`；字段改 `_last_published_effective_active_mask`。仅 controller/manager/test，不动路由/ACK/失败/HTTP。补 State 单测三条契约 | 小 | [x] |
| 3.3 | `controller.py:234` `_ApplyOp` 策略对象 | instruction 四处字符串分发 → `build_apply_op` 查表 + 多态（`_ApplyOp`/`_RetryOp`/`_ScaleDownOp`/`_RecoverOp`）。`validate_apply`/`begin_recover` 签名不变、内部委托 op；manager 两处 instruction 判断消除（`op.needs_resume()`/`op.isolated_ranks()`）。错误码与可观察行为不变 | 中 | [x] |

---

## 第 4 批 — 新提的 review 项

| # | 位置 | 问题 | 结论/改法 | 状态 |
|---|------|------|------|------|
| 4.1 | `manager.py:213` `_drop_process_inactive_pause_targets` | 有什么用？能不能删？ | **不能删**——处理"pause 目标在等 ACK 途中进程死亡"，否则事务卡超时、锁不放、后续 apply 全拒。属分布式正确性。已瘦身：日志 5 行→1 行，逻辑不动 | [x] 保留+瘦身 |
| 4.2 | `manager.py:268` `_fatal_task_wrapper` | 有啥用？ | **保留，不动**——fire-and-forget 后台任务的"预期外异常"保险丝：把 asyncio 默认的"静默吞掉 + GC 警告"升级为记日志 + 杀进程 + 退出（fail-stop，与 `_failstop` 同哲学）。预期内异常（超时/拒 ACK）已在各 `_send_*` 内部处理，不经过它。删了 FT 会带病假死，比 crash 更难查。**触发场景**：`fault_kill_pause_double_scale_down` 用例 | [x] 保留 |
| 4.3 | `manager.py:277` `_handle_exception_pause` | 是不是又重复了？ | **是，已收敛删除**——它跟 availability 两入口是同一模式（拿 targets→`_pause_schedulers`），只是多包一层 async 壳。收进 `handle_rank_fault`：strategy 门 + `begin_exception_pause()` + 空 target 拦截 + `_create_task(_pause_schedulers)`。三条 pause 触发链（process/mooncake/exception）结构统一。**目标范式**：故障场景 × 策略 × 操作尽量统一函数调用，可观察行为零变化 | [x] 已删 |
| 4.4 | `manager.py:292` `_publish_active_ranks` | 在故障链路中有实际作用吗？数据源/发给谁/为什么发 | **保留，不动**——属**恢复/操作链路（apply）**，非故障链路。**数据源**：`get_unpublished_effective_active_mask()`，即三源合成 effective mask（process_alive ∧ mooncake_member 再扣 disabled）自上次发布以来的净变化。**发给谁**：经 `send_to_scheduler`(ZMQ，与推理请求同 socket) → DPC，DPC `update_active_ranks` 更新路由表 `status` 并回 `ActiveRanksUpdateReqOutput` ACK。**为什么发**：apply(scale_down/recover) 改了 disabled → 必须让调度面同步"谁能接新活"，且作为事务需 ACK 确认（失败则 `commit_recover(set())` 回滚+503）。**与故障链路的差异（有意）**：故障观察走 fire-and-forget（observe 返回 `ActiveRanksOutput` 随消息流，不等 ACK，故障要快）；apply 走事务等 ACK（恢复要稳）。删了 apply 失去路由确认 | [x] 保留 |

---

## 第 5 批 — 当前会话交接状态（2026-07-24）

> **下一位 agent 的起点**：先执行 `git status --short` 和 `git log --oneline -5`，确认仍在 `worktree-dp-only-ft-revise` 且 HEAD 为 `b0c1416d5`。若不一致，先检查漂移，不能按本节直接继续改代码。

| # | 提交 | 已完成内容 | 已确认的边界/结论 | 状态 |
|---|---|---|---|---|
| 5.1 | `94ee057ca` | FT 相关日志统一为物理一行；同步收缩 FT control 日志 | 不保留仅用于定位的 FT 日志信息；全仓扫描范围不只 `manager.py` | [x] |
| 5.2 | `806abd21d` | `_fatal_task_wrapper` 与 `_failstop` 收敛 | 后台 FT task 的未预期异常仍 fail-stop；预期的 FT 命令/API失败由各自事务路径处理 | [x] |
| 5.3 | `b0c1416d5` | DPC FT 消息、进程映射和 watchdog 简化 | `SchedulerProcessInfo` 删除，仅保留与 `scheduler_procs` 同序的 `scheduler_process_dp_ranks`；watchdog 仍以 `connection.wait()`/sentinel 为核心 | [x] |
| 5.4 | `b0c1416d5` | watchdog 去除冗余防御层 | 删除 `_on_exit` 异常吞噬、`_stop_event` 和 `stop()` pipe 写入异常吞噬；保留 `stop_join_timeout`、Pipe、`remaining` 与 `wait()` | [x] |
| 5.5 | `b0c1416d5` | DPC dispatch 还原 e63 的四种负载策略 | FT 的全 DP 不可用拦截只在 `dispatching_with_trace()`；显式 `routed_dp_rank` 的 inactive 拦截只在 `maybe_external_dp_rank_routing()`；不引入 `_next_active_rank` | [x] |
| 5.6 | `b0c1416d5` | 本地 scheduler 异常退出上报 | 每个本地 DPC 的后台 watchdog 均监听（含 node0）；node0 主线程运行 `event_loop()`，非 node0 FT 主线程 `watchdog.wait()`；随后统一执行 e63 的 `proc.join()` | [x] |

### 5.7 消息与路由结论（继续 review 时不得反转）

- `FaultToleranceCommandReqInput`：FT manager → Node0 DPC → 指定 DP leader；scheduler 将它作为 control request 在本地 attention TP/CP block 内传播。
- `FaultToleranceCommandReqOutput`、`ActiveRanksUpdateReqOutput`、`FaultToleranceRankFaultOutput`、`ProcessActiveRanksOutput`：均回流到 TokenizerManager 的 dispatcher，不是 DPC 下行请求。
- `ProcessActiveRanksOutput(ranks: List[int], active: bool)` 的 `ranks` 是逻辑 DP rank；watchdog 单 child 异常上报 `[dp_rank]`，rejoin DPC 上报本节点去重后的 DP rank 集合。
- `enable_dp_attention_local_control_broadcast` 仍只决定 DPC 原生 `send_control_message()` 的广播范围；FT command 不经该 fallback。scheduler 内部的 FT local broadcast 是另一层语义，不能混为一个 `local_ctrl` 判断。

### 5.8 当前验证与限制

- 已通过：`python -m py_compile python/sglang/srt/managers/data_parallel_controller.py python/sglang/srt/utils/watchdog.py`，`git diff --check`（提交前）。
- 未能运行 watchdog pytest：Windows 上即使设置 `PYTHONPATH=python`，收集阶段仍因 `python/sglang/srt/utils/common.py` 导入 Unix-only `resource` 失败。Linux/CI 应运行：`PYTHONPATH=python python -m pytest test/registered/unit/utils/test_subprocess_watchdog.py -q`。
- 架构/场景事实以 `D:\Codex\shared\2026-07-17\SGLang DP-only FT 架构设计与验证指南\README.md` 第 3.5、3.7、5.2、5.3、7、8 节为准；该共享文档不属于本仓库提交。

### 5.9 后续 review 入口

1. 继续按 OWNER 的逐段 code review 推进；每个新问题先与 e63 和第 5.7 节消息边界对照，再决定是否修改。
2. 若改动 watchdog：不得删除 `connection.wait()`、sentinel `remaining`、Pipe stop 唤醒、`fail_stop_on_exit=False` 或非 node0 的 `wait()`，除非同步修改共享架构指南并给出覆盖多本地 child / 整节点退出 / rejoin 的验证。
3. 每次提交后在本节追加提交号、结论、验证与未决问题；`README.md` 的第 7、8 节只记录架构/验证进展，不承担逐条 code-review 交接。

---

## 第 6 批 — io_struct FT dataclass 字段审查

| # | 位置 | 问题 | 结论 | 状态 |
|---|------|------|------|------|
| 6.1 | `io_struct.py` 六个 FT dataclass | 是否有不需要的字段 | **无冗余**。逐字段核对消费方后，每个字段都有下游用途，且非"仅为定位"的噪声（符合字段最小化原则）。详见下方字段说明 | [OK] |

### 6.1.1 字段说明（io_struct.py FT dataclass）

| Dataclass | 字段 | 消费方 / 用途 |
|---|---|---|
| `FaultToleranceCommandReqInput` | `request_id` | scheduler 回填到 Output；manager 查 `_pending_commands` |
| | `command` | scheduler 判 pause/resume |
| | `target_ranks` | DPC 投递 + scheduler MLP-sync 等待 |
| `FaultToleranceCommandReqOutput` | `request_id` / `rank` / `success` / `message` | manager 查 pending、`acked.add(rank)`、分 ack/failed；`message` 设进 future exception 透传到 apply 的 503 响应体 |
| `FaultToleranceRankFaultOutput` | `rank` / `message` | manager 单行 warning 日志 |
| `ActiveRanksOutput` | `status` | manager 喂 `observe_mooncake_active_ranks` |
| | `request_id` | observe 路径留 `None`（fire-and-forget）；仅 `_publish_active_ranks` 的 apply 路径填 `uuid` 做路由 ACK 关联。**有意的双路径 Optional 设计** |
| `ProcessActiveRanksOutput` | `ranks` / `active` | manager `observe_process_active_ranks`、`_drop_process_inactive_pause_targets` |
| `ActiveRanksUpdateReqOutput` | `request_id` / `success` / `message` | manager 查 `_pending_active_rank_updates`、分 set_result/set_exception；`message` 进 `RuntimeError` |

### 6.1.2 `rid` vs `request_id`（不可复用）

`rid`（`BaseReq` 基类，`io_struct.py:52`）是**推理请求级** ID：`Optional[Union[str, List[str]]]`，关联用户请求输入↔输出，配 `regenerate_rid()`。

FT 的 `request_id` 是**控制事务级** ID：manager 在 `_publish_active_ranks` 现造 `uuid`，关联"下发的命令/路由发布"↔"回流的 ACK"。

**不应复用 `rid`**：① 类型不合（`rid` 可 None/list，`request_id` 需非空 str）；② 语义分层——5.7 节明确 FT 控制消息与推理数据消息的 ID 命名空间隔离，混用会撞名、模糊 e63 对齐边界。`ActiveRanksOutput.request_id` 用 `Optional[str]=None` 是为了单类型承载 observe/apply 双路径，是合理最小实现。

### 6.2 `_ft_rank()` 的 None 兜底不可达

`scheduler.py:1566` `_ft_rank()` = `dp_rank if dp_rank is not None else 0`。FT gate 已含 `_dp_attention_gate` + `ft_requires_dp_gt1`，FT 启用时 `dp_rank` 恒非 None，`else 0` 分支不可达；且返回 0 会把非 DP 配置误标成 DP0，反误导。**OWNER 已批：删 `_ft_rank()`，三处调用（1555/3823/3873）直接用 `self.dp_rank`。** 已执行：方法删除 + 三处替换，`grep _ft_rank` 无残留，`py_compile` 通过。状态：[x]

### 6.3 FT 只支持 DP attention 的根因（设计前提澄清）

不是"mooncake EP 代码明写只支持 DP attention"，而是 **FT 的三源状态模型必须按 DP 聚合，DP 聚合只在 DP attention 下存在**：

1. rank 布局 `dp_rank = tp_rank // (A*C)`（README 3.2）只在 DP attention 下定义；`model_runner.py:365` `dp_size = ... if enable_dp_attention else 1`。
2. mooncake→DP 投影在 `scheduler.py:3227-3238`：发送 `ActiveRanksOutput`（mooncake DP mask）本身就以 `enable_dp_attention` 为前提，`reshape(dp_size,-1).prod(axis=1)` 依赖连续 `(dp,cp,tp)` 布局；非 DP attention 下 `dp_size=1`，"按 DP 隔离/路由"语义不成立。
3. **代码无明写绑定**：`_handle_elastic_ep`（server_args.py:3303）只断言 `pp_size==1`、设 IB device，不强制 `enable_dp_attention`。`elastic_ep_backend` 与 `enable_dp_attention` 是两个独立开关，无硬绑定。
4. **文档未单独记录**：共享指南 README/GRAVEYARD 无专段解释，只在 3.2/4.8 隐含。"mooncake EP 只支持 DP attention"更像经验性事实。

这条解释了 dp_gt1 / mooncake backend / tp 整除三个 gate 的共同根因。**待办：建议把该设计前提同步进共享指南 README 4.8 节。** 状态：[x] 结论记录，README 同步待办

### 6.4 `_process_next_overlap_result` 替代 e63 `pop_and_process` —— 非单纯等效

e63 把"取队首→process→移除"写两遍：`event_loop_overlap` 闭包 `pop_and_process`（先 popleft 再 process）+ `pause_generation` 内联展开。当前统一为 `_process_next_overlap_result()` 方法（`scheduler.py:1631`），三处复用（1687/1699/3775），消重复。

**关键不只是消重复——顺序本身是 FT 必要加固**：当前是**先 `process_batch_result` 后 `popleft`**。`_ft_discard_inflight_window`（1571-1573）的 fault window 依赖 `result_queue` 里躺着的失败批次副本；若像 e63 先 popleft，`process_batch_result` 抛异常时该批次已被移出队列，且与 `last_batch` 非同一对象（是 `batch.copy()`），discard 会漏掉它——那些请求拿不到 503、KV 可能泄漏。**先 process 后 popleft 保证失败批次仍留在 result_queue 供完整 abort。** 状态：[x] 结论记录

### 6.5 `_ft_discard_inflight_window` 拆解 —— 无实质冗余

函数职责清晰（收集 fault window → 按 rid 去重 → 逐 req 释放 KV + abort → 清空状态）。逐候选点拆解结论（OWNER 已同意保留）：

| 候选冗余点 | 判断 | 理由 |
|---|---|---|
| `getattr(self, "result_queue", None)`（1571） | 保留 | FT 下 `enable_overlap` 可为 False（`disable_overlap_schedule`/mlx），此时 `dispatch_event_loop` 走 `event_loop_normal`，**不初始化 `result_queue`**。`getattr`+None 兜底是区分 overlap/non-overlap 的必要分支，非无脑防御。印证 README 4.6"non-overlap 清当前批；overlap 清完整窗口" |
| `if result_queue is not None` 两次（1572 extend、1624 clear） | 保留 | 同源：non-overlap 下无该属性，extend 与 clear 各需守卫；两次判断服务不同操作，不能合并 |
| `discarded_by_rid.setdefault` 去重 | 保留 | 按 rid 去重保序的标准写法；overlap 下同 req 可同时出现在 cur/last/running/result_queue 多个 batch，必须去重避免重复 abort/重复释放 KV |
| 逐字段清空（1620-1627） | 保留 | running_batch 重置空 batch、chunked_req 条件置 None（含 `_chunked_req_scheduled_last_iter` 联动）、result_queue clear、cur/last 置 None——各自语义不同，不能并 |

**与 6.2 的对照（判断 None 兜底是否冗余的标准）**：`_ft_rank()` 是真冗余，因为 FT gate 锁死了"FT 启用 ⇒ dp_rank 非 None"这一前提；而 `result_queue` 的 None 在 non-overlap FT 下**真实可达**，故保留。**标准：一个 None 兜底是否冗余，看它对应的状态组合在 FT gate 下是否真的不可达。** 状态：[OK] 拆解完成，无改动

---

## 待对齐 / 待你决定

1. ~~3.3 `_ApplyOp` 去留~~ → **已全批，已做。**
2. 后续新文件 review：你提 comment，我按本文档追加条目并标状态，并行推进。
3. **验证限制**：本地 Windows 跑不了完整 pytest（`sglang/__init__` 链到 Unix `resource`）。已用"文件级加载 + 手动 unittest runner"跑通 `test_controller.py` 15/15 + 各改点的行为等价断言。`test_manager.py`（依赖 mock 的 manager 完整链路）建议在 Linux/CI 跑一次确认。

---

## 进度日志

- 2026-07-23：建文档；录入 12 条意见。
- 2026-07-23：**第 0/1/2/3 批全部完成**（0 越界退回；1 controller 写法；2 manager 风格+降 try；3.1 pause 骨架+availability 收编；3.2 pending/mark 两接口+单测；3.3 _ApplyOp 策略对象）。`test_controller.py` 15/15 通过。待 OWNER review + `test_manager.py` Linux 复跑。
- 2026-07-24：补齐第 5 批交接状态：记录 `94ee057ca`、`806abd21d`、`b0c1416d5` 的 DPC/watchdog review 结论、消息边界与 Windows 验证限制。后续 agent 以第 5 批作为本轮 review/revise 的续接入口。
