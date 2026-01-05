# TRPO (Trust Region Policy Optimization)

TRPO（Trust Region Policy Optimization，信任域策略优化）是一种策略梯度算法，通过限制策略更新的幅度来保证策略改进的单调性，从而稳定训练过程。

## 核心思想

TRPO的核心思想是在策略更新时，确保新策略与旧策略之间的KL散度不超过一个阈值（信任域），从而避免策略更新过大导致的性能崩溃。

### 主要特点

1. **单调改进保证**：通过约束策略更新的幅度，确保每次更新都能提升策略性能
2. **自然策略梯度**：使用自然梯度方法，考虑策略空间的几何结构
3. **自适应步长**：通过线搜索自动调整更新步长，确保满足KL散度约束

## 数学原理

### 目标函数

TRPO的目标是最大化以下目标函数：

$$L(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot A(s,a)\right]$$

其中：
- $\pi_\theta(a|s)$ 表示**策略概率**：在状态 $s$ 下，策略 $\pi_\theta$ 选择动作 $a$ 的概率
  - 这是条件概率的表示方法，读作"在状态 $s$ 的条件下，策略选择动作 $a$ 的概率"
  - 例如：如果 $\pi_\theta(左转|路口) = 0.7$，表示在路口这个状态下，策略有70%的概率选择左转
  - $\theta$ 是策略的参数（如神经网络的权重）
- $\pi_{\theta_{old}}(a|s)$ 是旧策略在状态 $s$ 下选择动作 $a$ 的概率
- $A(s,a)$ 是优势函数，表示动作 $a$ 在状态 $s$ 下相对于平均水平的优势

#### 目标函数的作用机制

这个目标函数通过**重要性采样比率**（importance sampling ratio）$\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 来评估新策略的改进：

1. **重要性采样比率** $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$：
   - 例如：在某个状态下，旧策略 $\pi_{\theta_{old}}(左转|路口) = 0.3$，新策略 $\pi_\theta(左转|路口) = 0.6$
   - 重要性采样比率 = $\frac{0.6}{0.3} = 2$，表示新策略选择左转的概率是旧策略的2倍

2. **优势函数** $A(s,a)$：
   - $A(s,a) = Q(s,a) - V(s)$，表示选择动作 $a$ 比平均策略好多少
   - $A(s,a) > 0$：这个动作比平均水平好，应该增加选择概率
   - $A(s,a) < 0$：这个动作比平均水平差，应该减少选择概率

3. **目标函数的含义**：
   - 当 $A(s,a) > 0$（好动作）时，我们希望 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 增大，即新策略更倾向于选择这个好动作
   - 当 $A(s,a) < 0$（坏动作）时，我们希望 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 减小，即新策略减少选择这个坏动作
   - 通过最大化这个目标函数，我们可以在不重新收集数据的情况下，利用旧策略的经验数据来改进新策略

4. **为什么需要这个形式**：
   - 在策略梯度方法中，我们通常需要根据当前策略收集数据
   - 但TRPO使用旧策略收集的数据，通过重要性采样比率来"校正"到新策略的期望
   - 这样可以在不重新采样的情况下评估新策略的性能，提高样本效率

### 约束条件

策略更新必须满足KL散度约束：

$$\mathbb{E}\left[\text{KL}\left(\pi_{\theta_{old}}(\cdot|s) \parallel \pi_\theta(\cdot|s)\right)\right] \leq \delta$$

其中 $\delta$ 是信任域半径（通常为0.01或0.05）。

### 优化问题

TRPO将策略优化问题转化为带约束的优化问题：

$$\begin{aligned}
\max_{\theta} \quad & L(\theta) \\
\text{s.t.} \quad & \mathbb{E}\left[\text{KL}\left(\pi_{\theta_{old}}(\cdot|s) \parallel \pi_\theta(\cdot|s)\right)\right] \leq \delta
\end{aligned}$$

#### 优势函数 $A(s,a)$ 详解

优势函数是TRPO算法中的核心概念，用于评估动作的相对好坏：

**定义**：
$$A(s,a) = Q(s,a) - V(s)$$

其中：
- $Q(s,a)$ 是**动作价值函数**（Q函数）：在状态 $s$ 下选择动作 $a$ 后，能获得的期望累积奖励
  - $Q(s,a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0=s, a_0=a\right]$
- $V(s)$ 是**状态价值函数**：在状态 $s$ 下，遵循当前策略能获得的期望累积奖励
  - $V(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0=s\right] = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q(s,a)]$

**为什么使用优势函数而不是Q函数**：
1. **减少方差**：优势函数是Q函数减去基线（baseline）$V(s)$，可以减少策略梯度估计的方差，使训练更稳定
2. **相对评估**：是这个动作**相对于其他动作的价值**，而不是它本身的价值

## 算法流程

### 1. 收集经验数据

```python
def collect_trajectories(env, policy, num_trajectories):
    trajectories = []
    for _ in range(num_trajectories):
        states, actions, rewards = [], [], []
        state = env.reset()  # state: numpy.ndarray, shape取决于环境
        done = False
        
        while not done:
            action = policy.sample(state)  # action: int（离散）或 numpy.ndarray（连续）
            next_state, reward, done, _ = env.step(action)
            # next_state: numpy.ndarray, 与state相同shape
            # reward: float
            # done: bool
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        trajectories.append((states, actions, rewards))
    
    return trajectories  # List[Tuple[List, List, List]]
```

**具体类型示例**：

```python
# 环境示例（使用OpenAI Gym）
import gym
env = gym.make('CartPole-v0')

class Policy:
    def __init__(self, network):
        self.network = network  # network 是神经网络（如 torch.nn.Module）
        # network 的输入：state (torch.Tensor)
        # network 的输出：动作概率分布 (torch.Tensor, shape=(action_dim,))
    
    def sample(self, state):
        # state: numpy.ndarray -> torch.Tensor
        state_tensor = torch.FloatTensor(state)
        probs = self.network(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def get_probs(self, states, actions):
        probs = self.network(states)
        return probs.gather(1, actions.unsqueeze(1)).squeeze(1)

network = torch.nn.Sequential(
    torch.nn.Linear(state_dim, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, action_dim),
    torch.nn.Softmax(dim=-1)
)

policy = Policy(network)

trajectories = collect_trajectories(env, policy, num_trajectories=20)
```


### 2. 计算优势函数

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    使用GAE (Generalized Advantage Estimation) 计算优势函数
    
    参数说明：
    - rewards: List[float] - 轨迹中的奖励序列
    - values: List[float] - 状态价值函数 V(s) 的估计值序列
        * values 的来源：通过值函数网络（Value Network / Critic Network）估计得到
        * 值函数网络输入状态 s，输出该状态的期望累积奖励 V(s)
        * 例如：values = [V(s_0), V(s_1), V(s_2), ..., V(s_T)]
    - gamma: float - 折扣因子
    - lam: float - GAE 参数（λ）
    
    返回：
    - advantages: List[float] - 优势函数序列 A(s_t, a_t)
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        # 需要 values[t+1]，如果是最后一个时间步，values[t+1] = 0（终止状态）
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t+1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return advantages
```

**values 的来源**：

在TRPO中，`values` 是通过**值函数网络（Value Network / Critic）**估计得到的：

```python
# 值函数网络（Critic Network）
class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)  # 输出单个值 V(s)
        )
    
    def forward(self, state):
        return self.network(state).squeeze(-1)  # 返回 V(s)

# 在训练循环中使用
value_network = ValueNetwork(state_dim)

# 收集轨迹后，估计每个状态的价值
states = [s for traj in trajectories for s in traj[0]]  # 展平所有状态
states_tensor = torch.FloatTensor(states)

with torch.no_grad():
    values = value_network(states_tensor).tolist()  # 估计 V(s) 值

# 然后计算优势函数
advantages = compute_advantages(rewards, values, gamma, lam)
```

**关键点**：
- `values` 是状态价值函数 $V(s)$ 的估计，表示在状态 $s$ 下的期望累积奖励
- 通过值函数网络（Critic）学习得到，与策略网络（Actor）一起训练
- 在计算优势函数时，使用 $A(s,a) = Q(s,a) - V(s)$，其中 $Q(s,a) \approx r + \gamma V(s')$

### 3. 计算策略梯度

```python
def compute_policy_gradient(states, actions, advantages, old_probs, policy):
    """
    计算策略梯度
    """
    new_probs = policy.get_probs(states, actions)
    ratio = new_probs / old_probs
    
    # 策略梯度
    policy_gradient = ratio * advantages
    
    return policy_gradient, ratio
```

### 4. 自然策略梯度

```python
def natural_policy_gradient(policy_gradient, fisher_information_matrix):
    """
    计算自然策略梯度: $F^{-1} \cdot \nabla L$
    """
    # 使用共轭梯度法求解 $F^{-1} \cdot \nabla L$
    natural_grad = conjugate_gradient_solve(
        fisher_information_matrix, 
        policy_gradient
    )
    return natural_grad
```

### 5. 线搜索更新策略

```python
def line_search_update(policy, natural_grad, old_policy, states, 
                       advantages, old_probs, max_kl=0.01):
    """
    使用线搜索找到满足KL约束的最大步长
    """
    step_size = 1.0
    for _ in range(10):  # 最多尝试10次
        # 更新策略参数
        new_params = policy.params + step_size * natural_grad
        policy.set_params(new_params)
        
        # 计算KL散度
        new_probs = policy.get_probs(states)
        kl = compute_kl_divergence(old_policy, policy, states)
        
        # 计算目标函数改进
        ratio = new_probs / old_probs
        improvement = (ratio * advantages).mean()
        
        if kl <= max_kl and improvement > 0:
            return True  # 更新成功
        
        step_size *= 0.5  # 减小步长
    
    # 如果线搜索失败，不更新策略
    policy.set_params(old_policy.params)
    return False
```

### 6. 完整训练循环

```python
def train_trpo(env, policy, value_network, num_iterations=1000, 
               num_trajectories=20, max_kl=0.01):
    """
    TRPO训练主循环
    
    参数：
    - value_network: ValueNetwork - 值函数网络，用于估计 V(s)
    """
    for iteration in range(num_iterations):
        # 1. 收集经验
        trajectories = collect_trajectories(env, policy, num_trajectories)
        
        # 2. 处理轨迹数据
        states, actions, rewards = [], [], []
        for traj in trajectories:
            states.extend(traj[0])  # 展平所有状态
            actions.extend(traj[1])  # 展平所有动作
            rewards.extend(traj[2])  # 展平所有奖励
        
        # 3. 使用值函数网络估计 V(s)
        states_tensor = torch.FloatTensor(states)
        with torch.no_grad():
            values = value_network(states_tensor).tolist()  # 估计每个状态的价值
        
        # 4. 计算优势函数
        advantages = compute_advantages(rewards, values, gamma=0.99, lam=0.95)
        old_probs = policy.get_probs(states, actions)
        
        # 5. 计算策略梯度
        policy_grad, ratio = compute_policy_gradient(
            states, actions, advantages, old_probs, policy
        )
        
        # 6. 计算Fisher信息矩阵和自然梯度
        fisher_matrix = compute_fisher_matrix(policy, states)
        natural_grad = natural_policy_gradient(policy_grad, fisher_matrix)
        
        # 7. 线搜索更新策略
        old_policy = copy.deepcopy(policy)
        line_search_update(
            policy, natural_grad, old_policy, 
            states, advantages, old_probs, max_kl
        )
        
        # 8. 评估性能
        if iteration % 10 == 0:
            avg_reward = evaluate_policy(env, policy)
            print(f"Iteration {iteration}, Average Reward: {avg_reward}")
```

## 关键实现细节

### Fisher信息矩阵

Fisher信息矩阵用于计算自然梯度：

```python
def compute_fisher_matrix(policy, states):
    """
    计算Fisher信息矩阵: F = E[∇log π(a|s) ∇log π(a|s)^T]
    数学公式: $F = \mathbb{E}\left[\nabla\log\pi(a|s) \cdot \nabla\log\pi(a|s)^T\right]$
    """
    # 使用自动微分计算梯度
    # 实际实现中通常使用共轭梯度法，避免显式计算F矩阵
    pass
```

### 共轭梯度法

由于Fisher信息矩阵很大，通常使用共轭梯度法求解：

```python
def conjugate_gradient_solve(Ax_func, b, max_iter=10):
    """
    使用共轭梯度法求解 Ax = b
    Ax_func: 函数，计算 A*x
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    
    for _ in range(max_iter):
        Ap = Ax_func(p)
        alpha = (r.dot(r)) / (p.dot(Ap))
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        if r_new.norm() < 1e-10:
            break
        
        beta = (r_new.dot(r_new)) / (r.dot(r))
        p = r_new + beta * p
        r = r_new
    
    return x
```

## 超参数设置

- **信任域半径 ($\delta$)**: 0.01 或 0.05
- **折扣因子 ($\gamma$)**: 0.99
- **GAE参数 ($\lambda$)**: 0.95
- **每次迭代轨迹数**: 20-50
- **线搜索最大步数**: 10
- **共轭梯度迭代次数**: 10-20

## 优缺点

### 优点

1. **训练稳定**：通过KL约束保证单调改进
2. **样本效率高**：相比其他策略梯度方法，样本利用率更高
3. **理论保证**：有单调改进的理论保证

### 缺点

1. **计算复杂**：需要计算Fisher信息矩阵和共轭梯度，计算量大
2. **实现复杂**：实现难度较高
3. **超参数敏感**：信任域半径等超参数需要仔细调优

## 参考资料

- [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

