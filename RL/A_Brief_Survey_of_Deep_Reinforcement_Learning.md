## 《A Brief Survey of Deep Reinforcement Learning》阅读报告

### RL中的挑战

1. 最优策略必须从与环境的交互试错中推断出来，agent唯一的学习信号只有回报（2017年）
2. agent的观测值具有很强的时序相关性
3. agent必须处理长期的时间依赖性（long-range time dependencies），通常经过很多轮的与环境的交互，一系列的动作才变得有意义，这就是credit assignment 问题
