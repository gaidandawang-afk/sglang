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

## 待对齐 / 待你决定

1. ~~3.3 `_ApplyOp` 去留~~ → **已全批，已做。**
2. 后续新文件 review：你提 comment，我按本文档追加条目并标状态，并行推进。
3. **验证限制**：本地 Windows 跑不了完整 pytest（`sglang/__init__` 链到 Unix `resource`）。已用"文件级加载 + 手动 unittest runner"跑通 `test_controller.py` 15/15 + 各改点的行为等价断言。`test_manager.py`（依赖 mock 的 manager 完整链路）建议在 Linux/CI 跑一次确认。

---

## 进度日志

- 2026-07-23：建文档；录入 12 条意见。
- 2026-07-23：**第 0/1/2/3 批全部完成**（0 越界退回；1 controller 写法；2 manager 风格+降 try；3.1 pause 骨架+availability 收编；3.2 pending/mark 两接口+单测；3.3 _ApplyOp 策略对象）。`test_controller.py` 15/15 通过。待 OWNER review + `test_manager.py` Linux 复跑。
