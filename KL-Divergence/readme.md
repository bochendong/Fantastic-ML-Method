
# KL Divergence

KL散度(Kullback-Leibler Divergence)是用来度量两个概率分布相似度的指标.

KL 散度衡量的是：如果你“以为”数据来自分布 Q，但真实其实来自分布 P，你会多浪费多少信息。

对两个离散概率分布 P 和 Q

$$KL(P|Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)}$$

**Example**

假设真实分布：
$$P(A) = 0.5, P(B) = 0.5$$

但是你认为的
$$P(A) = 0.5, P(B) = 0.5$$

那么

$$KL(P|Q) = 0.5 * \log \frac{0.5}{0.5} + 0.5 * \log \frac{0.5}{0.5} = 0$$


**Example**

假设真实分布：
$$P(A) = 0.5, P(B) = 0.5$$

但是你认为的
$$P(A) = 0.9, P(B) = 0.1$$

那么

$$KL(P|Q) = 0.5 * \log \frac{0.5}{0.1} + 0.5 * \log \frac{0.5}{0.9} = 0.22185$$

**Question: 在TRPO中使用到的**

$$KL(\pi_{new}|\pi_{old})$$

是什么意思

**答案：**

在TRPO（Trust Region Policy Optimization）中，$KL(\pi_{new}||\pi_{old})$ 用于衡量**新策略** $\pi_{new}$ 与**旧策略** $\pi_{old}$ 之间的差异。


1. **在TRPO中的作用**：
   - **约束策略更新幅度**：TRPO要求 $KL(\pi_{new}||\pi_{old}) \leq \delta$（$\delta$ 是信任域半径，通常为0.01或0.05）
   - **防止策略崩溃**：限制新策略与旧策略的差异，避免策略更新过大导致性能急剧下降
   - **保证单调改进**：通过限制KL散度，确保策略改进的稳定性




## 和Softmax, cross-entropy的区别


In the Softmax classifier, the function mapping f(x<sub>i</sub>;W)=Wx<sub>i</sub> stays unchanged, but we now interpret these scores as the unnormalized log probabilities for each class and replace the hinge loss with a cross-entropy loss that has the form:

$$L_i = -log *\frac(e^s_{y_i}{\sum_j e^{s_j}})$$

比如我们预测一张猫的图片，模型得出的分数是（分数越大代表模型认为越像）

$$score(cat) = 3.2, score(car) = 5.1, score(frog) = -1.7$$

那么，想做soft max loss，我们就要先 $exp$

$$e^{score(cat)} = 24.5, e^{score(car)} = 164.0, e^{score(frog)} = 0.18$$

然后再归一化

$$prob(cat) = 0.13, prob(car) = 0.87, prob(frog) = 0.0$$

最后计算

$$loss = -log (0.13)$$

**Quesiton: 最小和最大的softmax loss是什么**

最小是0，最大是正无穷

**Quesiton: 怎么能让我的loss最小**

这其实需要我们的score等于无限才可以做到，或者cat的score远大于其他的，所以在实践中，我们拿不到等于0的loss，所以0只是理论上的最小的loss。


**Quesiton: 和L1，L2的区别**
而L1或者L2的loss，只是判断你是否猜到了true label，猜准了这个network就开摆了，所以softmax更push模型的



**Quesiton: softmax loss和KL divergence的区别**

如果是one hot encoding，那么其实没有区别

**对于soft label的情况：**

当真实标签是soft label（如 [0.7, 0.2, 0.1]）而不是one-hot（如 [1, 0, 0]）时，两者就有区别了：

1. **Softmax Loss (Cross-Entropy Loss)**：
   - 通常只关注**最大概率的类别**（需要先对真实标签做argmax）
   - 公式：$L = -\log(\text{predicted\_prob}_{argmax(\text{true\_label})})$
   - 例如：真实标签 [0.7, 0.2, 0.1]，预测 [0.8, 0.15, 0.05]
     - 先argmax得到类别0，然后计算 $-\log(0.8) = 0.223$
   - **忽略了其他类别的概率信息**

2. **KL Divergence**：
   - 考虑**整个概率分布**的差异
   - 公式：$KL(P||Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$
   - 例如：真实标签 [0.7, 0.2, 0.1]，预测 [0.8, 0.15, 0.05]
     - $KL = 0.7 \log \frac{0.7}{0.8} + 0.2 \log \frac{0.2}{0.15} + 0.1 \log \frac{0.1}{0.05}$
     - $= 0.7 \times (-0.133) + 0.2 \times 0.288 + 0.1 \times 0.693 = 0.062$
   - **保留了所有类别的概率信息**

**关键区别总结：**
- Softmax loss在soft label下通常只优化最大概率类别，**丢失了分布信息**
- KL divergence会考虑所有类别的概率，**保留了完整的分布信息**，更适合处理soft label（如知识蒸馏、标签平滑等场景）




## Reverse KL

$$ReverseKL(Q|P) = \sum_x Q(x)\log \frac{Q(x)}{P(x)}$$

### 与正向 KL 的关键区别

**Forward KL (P||Q) - Mode-Covering（模式覆盖）**：
- 当 $P(x) > 0$ 但 $Q(x) = 0$ 时，惩罚会非常大（趋向无穷）
- 要求 Q **必须覆盖** P 的所有非零概率区域
- 结果：Q 会变得**更宽泛**，覆盖所有可能

**详细解释**：

从公式 $KL(P||Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)}$ 可以看出：
- 如果某个 $x$ 满足 $P(x) > 0$ 但 $Q(x) = 0$，那么 $\log \frac{P(x)}{0} = \log(\infty) = \infty$
- 这意味着只要 P 在某个区域有概率，Q 就**绝对不能**在那个区域为零，否则损失会爆炸

**直观例子**：
假设真实分布 P 是双峰分布（有两个峰值）：
- $P(A) = 0.3$（第一个峰）
- $P(B) = 0.1$（两个峰之间的低谷）
- $P(C) = 0.6$（第二个峰）

如果我们用单峰分布 $Q_1(A) = 0.4, Q_1(B) = 0.0, Q_1(C) = 0.6$ 来近似：
$$KL(P||Q_1) = 0.3 * \log \frac{0.3}{0.4} + 0.1 * \log \frac{0.1}{0.0} + 0.6 * \log \frac{0.6}{0.6} = \infty$$
- 因为 $P(B) = 0.1 > 0$ 但 $Q_1(B) = 0$，导致第二项为无穷大

为了最小化 Forward KL，Q 必须"安全起见"，在**所有** P 有概率的地方都分配非零概率，即使那些地方概率很小。这导致 Q 会变得**更宽泛**，像一个"保险网"一样覆盖所有可能。

**Reverse KL (Q||P) - Mode-Seeking（模式寻找）**：
- 当 $Q(x) > 0$ 但 $P(x) = 0$ 时，惩罚会非常大（趋向无穷）
- 要求 Q **不能有** P 为零的概率区域
- 结果：Q 会变得**更紧凑**，只关注 P 的主要模式
