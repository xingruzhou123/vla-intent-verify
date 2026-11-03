这版判断是对的。需要再补一句：第 0 道门不是要求理论上证明“只有 VLA 能做”，而是通过公平基线证明，访问动作表示或联合生成连续动作带来了模块化方案没有的实证增益。

下面可以直接贴到新对话：

我想重新寻找一个真正需要 VLA 的机器人安全研究方向。请先做新颖性和必要性核查，不要直接写完整计划或开始实现。

## 前一项目的复盘

原项目研究抓取搬运中的连续安全可行域：夹持力 \(F\) 太小会滑落、太大会损伤，搬运时间尺度 \(s\) 影响动态稳定性。我们在 MuJoCo 中从相同抓取前状态分支，扫描不同的 \((F,s)\)，希望生成反事实数据并微调 \(\pi_0.5\)。

Phase 1 最终覆盖 89 个物体和数万次经过双实现核验的分支仿真。结果显示：

- 力控制、滑落和损伤标签能够可靠实现；
- 速度对安全边界的影响不足以支持原主张；
- 在当前刚体搬运环境、物体范围和观测设定下，校准后的解析物理模型能够恢复大部分可利用的信息价值；
- 主动试探提供额外信息，但固定试探加解析更新已经恢复约 71%–88% 的 oracle gap，具体数值随无偏选择协议变化；
- 剩余误差主要来自校准、probe noise、随机边界和动作网格分辨率，没有发现稳定、可观测、可泛化的 learned residual；
- 没有证据证明在这个环境和期限内引入小型学习模型或 \(\pi_0.5\) 是必要的。

因此，原来的 feasibility-distillation 和 active-probing VLA 路线已经关闭。这个结果不能推广成“抓取搬运不需要学习”或“信息到位后公式总是足够”。有边界的结论是：

> 对主要由低维刚体动力学决定的安全问题，应先检查可观测性、解析可解性和简单模型上限，再考虑增加模型复杂度。

保留下来的资产包括：MuJoCo 环境、同状态反事实分支、双实现标签核验、解析基线、信息条件性能估计、nested evaluation，以及“先过门槛再训练”的研究流程。

## 新方向候选

候选问题是：

> VLA 能否把开放式任务语义转化为明确的安全义务，判断当前场景证据是否足以验证这些义务，并在证据缺失时提出针对性问题，然后生成满足义务的连续动作？

例子包括：

- 热液体激活倾角、加速度和人体距离约束；
- 古董或易碎品激活接触力和碰撞余量约束；
- 无菌容器激活接触区域和污染历史约束；
- 递给人和放到托盘激活不同的速度、朝向和接触要求；
- 无法确认关键属性时，机器人应询问缺失信息，而不是泛化地问“是否继续”。

这个方向延续 CLEAR 和早期 A 项目的“义务 + 证据 + 执行前核验”结构，但必须避免只是把同一个框架换到 VLA 上。

## 开始前必须回答的两个问题

### 1. 为什么需要 VLA，而不是 VLM + 验证器？

一个强模块化基线可能已经足够：

\[
\text{VLM}
\rightarrow
\text{obligation/constraint}
\rightarrow
\text{analytic verifier or motion optimizer}
\]

只有在义务与策略实际生成的连续动作紧密耦合时，VLA 才可能有必要，例如：

- 义务针对的是 policy 即将产生的具体动作和预期效果，而不是静态场景属性；
- 满足义务需要在多个连续动作维度之间权衡，如速度、倾角、接近路径和接触方式；
- 义务不能完整编译成预先定义的符号规则；
- 联合学习能够在未见对象、未见任务组合或闭环变化下，生成满足约束的动作；
- 访问或训练 action representation 能提供 instruction、scene 和外部 planned chunk 无法提供的增量信息。

请设计公平的第 0 道门，至少比较：

1. VLM obligation extraction + rule/physics verifier；
2. VLM obligation extraction + strong motion optimizer；
3. VLA without explicit obligations；
4. joint obligation-conditioned VLA；
5. oracle obligation + optimizer。

如果模块化方案已经达到同等安全性、任务成功率和泛化性能，就不应继续把它写成 VLA 论文。

### 2. 与 CLEAR 和已有工作的边界是什么？

CLEAR 在 LLM 规划层面对显式计划进行基于场景证据的验证。新工作只有在以下方面具有实质差异时才成立：

- 验证对象从显式文本计划转为连续 action generation 或 policy intent；
- obligation 不只是外挂 monitor 的检查项，而是直接参与动作生成或训练；
- 研究的是 semantic requirement 如何改变低层连续控制；
- 评估包含 action-level obligation satisfaction、闭环适应和未见语义—动作组合；
- 方法不仅能 BLOCK，还能生成满足同一语义义务的替代动作。

请重点核查以下邻近方向：

- DeliGrasp 等 semantic description → physical parameter/action；
- ASIMOV、HazardArena 等 semantic robot-safety benchmark；
- semantic constraint extraction 和 language-conditioned motion planning；
- KnowNo、CoFineLLM 及后续 targeted clarification/help-seeking；
- VLA safety monitor、latent guard 和 action verifier；
- CLEAR 本身最接近的贡献边界。

## 这轮任务

请先完成：

1. 最近邻工作的结构化 novelty matrix；
2. 明确哪些 claim 已经被覆盖；
3. 提出 2–3 个真正存在 action–semantic coupling 的具体任务；
4. 为每个任务设计“VLM + verifier/optimizer”强基线；
5. 设计一个低成本第 0 道门，判断 VLA 是否提供不可替代的实证增益；
6. 如果找不到能通过第 0 道门的任务，请明确建议关闭这个方向，不要为了使用 VLA 反向制造问题。

暂时不要写四个月计划，也不要开始收集数据或微调模型。先证明问题新、VLA 有必要、并且与 CLEAR 有实质区别。
