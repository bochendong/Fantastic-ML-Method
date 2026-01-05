# GRPO (Group Relative Policy Optimization)

GRPO（Group Relative Policy Optimization，组相对策略优化）是一种强化学习算法，主要用于大模型训练场景。GRPO通过引入"组相对"机制，将样本划分为多个组，在组内计算相对优势，从而显著提高样本利用率和策略更新的稳定性。

## 核心思想

GRPO的核心创新在于其独特的**组相对策略更新机制**。传统的PPO算法在更新策略时，仅考虑当前样本与历史策略的相对优势，而GRPO则将样本划分为多个组（如按任务类型、难度等级、序列长度等维度），在组内计算相对优势值。

### 主要特点

1. **样本利用更高效**：组内样本具有更高的相似性，相对优势计算更精准
2. **策略更新更稳定**：组间差异作为正则化项，防止策略过度偏向特定样本
3. **适应复杂任务**：特别适合多任务、长序列的大模型训练场景
4. **自动组间平衡**：通过组间正则化自动平衡不同组的影响

## 数学原理

### 目标函数

GRPO的目标函数结合了PPO的裁剪机制和组间正则化：

$$L^{GRPO}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \cdot A_t^{group}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t^{group}\right)\right] - \beta \cdot D_{KL}(\pi_\theta \parallel \pi_{old})$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率
- $A_t^{group}$ 是**组内相对优势**，与传统优势函数不同
- $\epsilon$ 是裁剪参数（通常为0.1或0.2）
- $\beta$ 是组间差异正则化系数
- $D_{KL}(\pi_\theta \parallel \pi_{old})$ 是KL散度正则化项

### 组内相对优势

GRPO的关键创新在于**组内相对优势**的计算：

$$A_t^{group} = R_t - \bar{R}_{group(t)}$$

其中：
- $R_t$ 是样本 $t$ 的回报（return）
- $\bar{R}_{group(t)}$ 是样本 $t$ 所在组的平均回报
- $group(t)$ 表示样本 $t$ 所属的组

**与传统优势函数的区别**：
- **传统优势**：$A_t = R_t - V(s_t)$，相对于状态价值函数
- **组内相对优势**：$A_t^{group} = R_t - \bar{R}_{group}$，相对于组内平均回报

### 组划分策略

GRPO需要将样本划分为多个组，常见的划分方式包括：

1. **按任务类型分组**：不同任务类型的样本分到不同组
2. **按难度等级分组**：根据任务难度（如序列长度、复杂度）分组
3. **按奖励范围分组**：根据奖励值的大小范围分组
4. **按时间步分组**：在长序列任务中，按时间步位置分组

## 算法流程

### 1. 收集经验数据

```python
def collect_trajectories(env, policy, num_trajectories):
    """
    收集轨迹数据，同时记录组信息
    """
    trajectories = []
    for _ in range(num_trajectories):
        states, actions, rewards, log_probs, group_id = [], [], [], [], []
        state = env.reset()
        done = False
        
        # 确定当前轨迹的组ID（例如根据任务类型或难度）
        current_group = determine_group(env)  # 需要根据具体任务实现
        
        while not done:
            action, log_prob = policy.sample(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            group_id.append(current_group)
            
            state = next_state
        
        trajectories.append((states, actions, rewards, log_probs, group_id))
    
    return trajectories
```

### 2. 计算组内相对优势

```python
def compute_group_advantages(rewards, group_indices, gamma=0.99):
    """
    计算组内相对优势
    
    注意：与PPO不同，GRPO不需要value network！
    - PPO需要：values = value_network(states) 来计算 V(s)
    - GRPO不需要：直接使用组内平均回报作为基线
    
    参数：
    - rewards: List[float] - 奖励序列
    - group_indices: List[int] - 每个样本所属的组ID
    - gamma: float - 折扣因子
    
    返回：
    - group_advantages: List[float] - 组内相对优势序列
    - returns: List[float] - 回报序列
    """
    # 首先计算每个样本的回报（return）
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    
    # 按组计算平均回报
    unique_groups = list(set(group_indices))
    group_means = {}
    for group_id in unique_groups:
        group_returns = [returns[i] for i in range(len(returns)) if group_indices[i] == group_id]
        group_means[group_id] = np.mean(group_returns)
    
    # 计算组内相对优势
    group_advantages = []
    for i in range(len(returns)):
        group_id = group_indices[i]
        group_mean = group_means[group_id]
        advantage = returns[i] - group_mean  # 组内相对优势
        # 注意：这里使用的是组内平均回报，而不是值函数V(s)
        # PPO中：A_t = R_t - V(s_t)，需要value network估计V(s)
        # GRPO中：A_t^{group} = R_t - R̄_group，直接计算，无需value network
        group_advantages.append(advantage)
    
    return group_advantages, returns
```

### 3. GRPO更新

```python
def grpo_update(policy, states, actions, old_log_probs, group_advantages,
                returns, epsilon=0.2, beta=0.01, epochs=10, batch_size=64):
    """
    GRPO更新函数
    """
    dataset = list(zip(states, actions, old_log_probs, group_advantages, returns))
    
    for epoch in range(epochs):
        random.shuffle(dataset)
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_states, batch_actions, batch_old_log_probs, \
                batch_advantages, batch_returns = zip(*batch)
            
            # 计算当前策略的log概率
            new_log_probs = policy.get_log_prob(batch_states, batch_actions)
            
            # 计算重要性采样比率
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # PPO裁剪目标
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL散度正则化项（组间正则化）
            old_probs = torch.exp(batch_old_log_probs)
            new_probs = torch.exp(new_log_probs)
            kl_div = torch.sum(new_probs * (torch.log(new_probs) - torch.log(old_probs)), dim=-1).mean()
            
            # 总损失
            loss = policy_loss + beta * kl_div
            
            # 更新策略
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
```

### 4. 完整训练循环

```python
def train_grpo(env, policy, num_iterations=1000, 
               num_trajectories=20, epsilon=0.2, beta=0.01, epochs=10):
    """
    GRPO训练主循环
    """
    for iteration in range(num_iterations):
        # 1. 收集经验
        trajectories = collect_trajectories(env, policy, num_trajectories)
        
        # 2. 处理轨迹数据
        states, actions, rewards, old_log_probs, group_indices = [], [], [], [], []
        for traj in trajectories:
            states.extend(traj[0])
            actions.extend(traj[1])
            rewards.extend(traj[2])
            old_log_probs.extend(traj[3])
            group_indices.extend(traj[4])
        
        # 3. 计算组内相对优势
        group_advantages, returns = compute_group_advantages(
            rewards, group_indices, gamma=0.99
        )
        
        # 4. 归一化优势（可选）
        group_advantages = (group_advantages - np.mean(group_advantages)) / (np.std(group_advantages) + 1e-8)
        
        # 5. GRPO更新
        grpo_update(policy, states, actions, old_log_probs, 
                   group_advantages, returns, epsilon, beta, epochs)
        
        # 6. 评估性能
        if iteration % 10 == 0:
            avg_reward = evaluate_policy(env, policy)
            print(f"Iteration {iteration}, Average Reward: {avg_reward}")
```

## 关键实现细节

### 1. 组划分策略

组划分是GRPO的关键，需要根据具体任务设计：

```python
def determine_group(env, state=None):
    """
    根据环境或状态确定组ID
    示例：根据任务类型、难度、序列长度等
    """
    # 示例1：根据任务类型
    if hasattr(env, 'task_type'):
        return env.task_type
    
    # 示例2：根据序列长度
    if state is not None:
        seq_length = len(state)
        if seq_length < 100:
            return 0  # 短序列组
        elif seq_length < 500:
            return 1  # 中等序列组
        else:
            return 2  # 长序列组
    
    # 默认分组
    return 0
```

### 2. 组内相对优势 vs 传统优势

这是GRPO与PPO/TRPO的**核心区别**：

**传统优势函数**（PPO/TRPO）：
- 使用值函数 $V(s)$ 作为基线：$A_t = R_t - V(s_t)$
- **需要训练额外的值函数网络（Value Network / Critic）**
- 值函数网络输入状态 $s$，输出 $V(s)$ 的估计值
- 使用GAE（Generalized Advantage Estimation）计算优势：
  ```python
  # PPO中需要先估计V(s)
  values = value_network(states)  # 需要value network
  advantages = compute_gae(rewards, values)  # A_t = R_t - V(s_t)
  ```

**组内相对优势**（GRPO）：
- 使用组内平均回报作为基线：$A_t^{group} = R_t - \bar{R}_{group}$
- **不需要额外的值函数网络**
- 直接从组内样本计算平均回报作为基线
- 计算方式：
  ```python
  # GRPO中直接计算组内平均
  group_means = compute_group_means(returns, group_indices)  # 不需要value network
  group_advantages = returns - group_means  # A_t^{group} = R_t - R̄_group
  ```

**关键区别总结**：

| 特性 | PPO/TRPO | GRPO |
|------|----------|------|
| **基线方式** | 值函数 $V(s)$ | 组内平均回报 $\bar{R}_{group}$ |
| **需要Value Network** | ✅ 是 | ❌ 否 |
| **计算复杂度** | 需要训练额外网络 | 只需统计计算 |
| **适用场景** | 通用场景 | 多任务、异构数据 |
| **优势** | 理论更完善 | 实现更简单，无需额外网络 |

#### 简单数值例子

假设我们收集了6个样本，分为2个组（组0和组1），每个样本的回报（return）如下：

**样本数据**：
```
样本1: return = 10, group = 0
样本2: return = 12, group = 0
样本3: return = 8,  group = 0
样本4: return = 20, group = 1
样本5: return = 18, group = 1
样本6: return = 22, group = 1
```

**PPO的计算方式**（需要Value Network）：
```python
# 1. 使用value network估计每个状态的V(s)
states = [s1, s2, s3, s4, s5, s6]
values = value_network(states)  # 假设输出: [9, 11, 7, 19, 17, 21]

# 2. 计算优势：A_t = R_t - V(s_t)
advantages_ppo = [
    10 - 9 = 1,   # 样本1
    12 - 11 = 1,  # 样本2
    8 - 7 = 1,    # 样本3
    20 - 19 = 1,  # 样本4
    18 - 17 = 1,  # 样本5
    22 - 21 = 1   # 样本6
]
# 注意：需要训练value_network来估计V(s)
```

**GRPO的计算方式**（不需要Value Network）：
```python
# 1. 计算组内平均回报
group_0_returns = [10, 12, 8]
group_1_returns = [20, 18, 22]
group_0_mean = (10 + 12 + 8) / 3 = 10.0
group_1_mean = (20 + 18 + 22) / 3 = 20.0

# 2. 计算组内相对优势：A_t^{group} = R_t - R̄_group
advantages_grpo = [
    10 - 10.0 = 0.0,   # 样本1（组0，等于组内平均）
    12 - 10.0 = 2.0,   # 样本2（组0，高于组内平均）
    8 - 10.0 = -2.0,   # 样本3（组0，低于组内平均）
    20 - 20.0 = 0.0,   # 样本4（组1，等于组内平均）
    18 - 20.0 = -2.0,  # 样本5（组1，低于组内平均）
    22 - 20.0 = 2.0    # 样本6（组1，高于组内平均）
]
# 注意：不需要value network，直接统计计算即可
```

**关键观察**：
1. **PPO**：所有样本的优势都是1（假设value network估计准确），因为都是相对于各自状态的V(s)
2. **GRPO**：优势是相对于组内平均的，组0中样本2最好（+2.0），组1中样本6最好（+2.0）
3. **GRPO的优势**：
   - 不需要训练value network
   - 组内相对比较更直观（相对于同组其他样本）
   - 特别适合多任务场景（不同组可能代表不同任务）

### 3. 组间正则化

KL散度正则化项 $\beta \cdot D_{KL}(\pi_\theta \parallel \pi_{old})$ 的作用：
- 防止策略过度偏向某个组
- 保持组间的平衡
- $\beta$ 控制正则化强度（通常为0.01-0.1）

## 超参数设置

- **裁剪参数 ($\epsilon$)**: 0.1 或 0.2
- **组间正则化系数 ($\beta$)**: 0.01-0.1
- **折扣因子 ($\gamma$)**: 0.99
- **每次迭代轨迹数**: 20-50
- **每次迭代更新轮数 (epochs)**: 3-10
- **批次大小 (batch_size)**: 64-256
- **学习率**: 3e-4（Adam优化器）

## 优缺点

### 优点

1. **样本效率高**：组内样本共享信息，相对优势计算更精准
2. **策略稳定**：组间正则化防止策略过度偏向特定组
3. **适应复杂任务**：特别适合多任务、长序列的大模型训练
4. **无需值函数网络**：组内相对优势不需要额外的值函数估计

### 缺点

1. **需要组划分**：需要设计合适的组划分策略
2. **计算开销增加**：需要计算组内统计信息
3. **组划分敏感**：组划分策略对性能影响较大
4. **实现复杂度**：比PPO实现更复杂

## 与其他算法对比

| 特性 | PPO | TRPO | GRPO |
|------|-----|------|------|
| **基线方式** | 值函数 $V(s)$ | 值函数 $V(s)$ | 组内平均回报 |
| **需要值函数网络** | 是 | 是 | 否 |
| **样本效率** | 中等 | 高 | 很高（组内共享） |
| **多任务适应性** | 中等 | 中等 | 优秀 |
| **实现复杂度** | 简单 | 复杂 | 中等 |
| **适用场景** | 通用 | 通用 | 大模型、多任务 |

## 应用场景

GRPO特别适合以下场景：

1. **大语言模型训练**：处理不同长度、不同难度的文本生成任务
2. **多任务学习**：同时学习多个相关任务
3. **长序列任务**：处理长序列的强化学习问题
4. **异构数据**：数据分布差异较大的场景

## 参考资料

- [DeepSeek GRPO算法解析](https://cloud.baidu.com/article/3588245)
- [Group Relative Policy Optimization for Large Language Models](相关论文)

