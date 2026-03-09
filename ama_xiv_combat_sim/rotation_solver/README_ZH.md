# VPR Rotation Solver（新手上手）

这份文档对应 `ama_xiv_combat_sim/rotation_solver/vpr_mvp_solver.py` 的最小版本，用来做：

1. 从你的 VPR 战斗日志（CSV）学习“下一步常用技能”先验；
2. 按这个先验生成一段动作计划；
3. 用模拟器计算该计划的期望总伤害。

---

## 1) 你需要准备什么

### A. 必要输入（最少）

- 一份 rotation CSV（可以由 FFLogs 转换得到）
- CSV 至少应有这些列：
  - `time`
  - `skill_name`
  - `job_class`
  - `skill_conditional`（可以为空）

> 说明：解析使用的是项目内置 `CSVUtils.read_rotation_from_csv`，列名按小写处理。

### B. 用于模拟打分时还需要

- `stats`（你的面板/装备参数，如 wd、main_stat、crit 等）
- `skill_library`（通过 `create_skill_library()` 创建）

---

## 2) 它会输出什么

这个 MVP 会产出三类结果：

1. **训练样本**：`list[VPRActionSample]`
   - 每条样本是 `(state, action)`
   - `state` 包含：
     - 当前战斗时间 `elapsed_time_s`
     - 与上一动作的时间差 `gcd_delta_s`
     - 是否在 120s 爆发窗口 `in_burst_window`
     - 上一个动作 `last_action`

2. **动作计划**：`list[str]`
   - 例如：`["Steel Fangs", "Dreadwinder", ...]`
   - 表示接下来按键序列（技能名）

3. **期望总伤害**：`float`
   - 通过模拟器对该动作计划打分得到

---

## 3) 最小可运行示例（照抄即可）

```python
import ama_xiv_combat_sim

from ama_xiv_combat_sim.rotation_solver import (
    VPRFrequencyPriorModel,
    VPRGenerationConfig,
    build_vpr_samples_from_csv,
    generate_vpr_action_plan,
    evaluate_action_plan_expected_damage,
)
from ama_xiv_combat_sim.simulator.skills import create_skill_library
from ama_xiv_combat_sim.simulator.stats import Stats

# 1) 从 CSV 构建训练样本
samples = build_vpr_samples_from_csv("your_vpr_rotation.csv", player_job="VPR")

# 2) 训练一个最简单的“频次先验”模型
model = VPRFrequencyPriorModel()
model.fit(samples)

# 3) 生成计划（horizon=要生成多少步）
plan = generate_vpr_action_plan(
    model,
    seed_actions=["Steel Fangs"],
    start_time_s=0.0,
    gcd_s=2.5,
    config=VPRGenerationConfig(horizon=24, top_k=3),
)

print("Generated plan:", plan)

# 4) 用模拟器评估计划伤害
SKILL_LIBRARY = create_skill_library()
stats = Stats(
    wd=132,
    weapon_delay=2.64,
    main_stat=3330,
    det_stat=2182,
    crit_stat=2596,
    dh_stat=940,
    speed_stat=400,
    tenacity=400,
    job_class="VPR",
)

expected_damage = evaluate_action_plan_expected_damage(
    actions=plan,
    stats=stats,
    skill_library=SKILL_LIBRARY,
    num_samples=100000,
)

print("Expected damage:", expected_damage)
```

---

## 4) 推荐你的实际工作流（新手版）

1. **先做数据**：把 FFLogs 转成规范 CSV（至少保证 `time/skill_name/job_class`）
2. **先验证可读**：先跑 `build_vpr_samples_from_csv` 看是否有样本
3. **先跑短计划**：`horizon=8~12` 检查生成技能名是否合理
4. **再跑长计划**：`horizon=24~40`，再用模拟器打分
5. **多次对比**：对比不同 seed、不同 fight 时长、不同 stats

---

## 5) 你最常见会遇到的问题

- **问题：样本数是 0**
  - 原因：`job_class` 不是 `VPR`，或 CSV 列名不对
- **问题：计划里有奇怪技能名**
  - 原因：原始日志中混入队友技能/异常行，需清洗 CSV
- **问题：模拟报技能找不到**
  - 原因：CSV 技能名与技能库命名不一致（需做映射）

---

## 6) 这个 MVP 的定位（很重要）

这是“可开工”的最小基线，不是最终最优 solver：

- 它目前是**频次先验模型**（不是深度学习）
- 它主要作用是：给搜索器提供候选动作
- 最终建议你在下一步叠加：
  - 合法动作约束（CD/资源/连段）
  - beam search
  - 多目标评分（期望伤害 + 稳定性）

如果你是代码新手，建议你先跑通上面的最小示例，再逐步加复杂功能。
