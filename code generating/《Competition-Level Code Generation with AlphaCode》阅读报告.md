## 《Competition-Level Code Generation with AlphaCode》阅读报告

### 研究现状

现有的大型Language Model已经被证明可以用于生成代码，但是只能生成一些短小的代码片段，或者在解决复杂、不可见的、需要编程技巧的问题上，这些模型的性能还很弱。面临的挑战有：

1. 需要搜索大量的代码：
    ```
    Generating code that solves a specific task requires searching in a huge structured space of programs with a very sparse reward signal.
    ```
    1.1 仅仅改变单个字符就有可能改变整个程序的逻辑，即使这没有引起崩溃，所以代码的相关性不能仅仅依靠文本字面上的相关性；
    1.2 一题多解：同一个问题可能有多种编码逻辑可供选择
    1.3 在很多（编程）领域，尤其是编程竞赛，对于每个问题常常只有有限个样本和解题方案可供训练

2. 衡量代码是否有效的测试用例通常是不可见的，需要提供一个有效的测评代码的基准。

现有的方案在生成大型程序代码上还不可靠，加上有效测试用例的缺乏，使得存在较高的假阳性率（false positive rate）。

本文的任务：

1. 提出一个生成代码的新新方案：AlphaCode，使用大型transformer模型，用GitHub上的代码pre-train，然后在精选过的编程竞赛问题上fine-tuning，总体流程如下：
![../images/9/1644312267351.jpg](../images/9/1644312267351.jpg)

```
1.1 pre-train
1.2 fine tune
1.3 Large scale sampling: Generate a very large number of samples fomr our models for each problem.
1.4 Filter the samples to obtain a small set of candidate submissions (at most 10), to be evaluated on the hidden test cases, by using the example tests and clustering to pick samples based on program behaviour. 
```

2. 发布新的关于编程竞赛数据集CodeContests，用于模型训练：


3. 证明本文生成代码的方法并非直接复制训练集的代码片段，而是根据自然语言描述生成的。

本文的创新点在于：便捷、高效的sampling和filtering解题目代码上。

### 研究方法

* 数据集构建:CodeContests
1. 数据结构包含：
1.1 题目难度等级
1.2 解题方法归类标签，如"greedy"、"dp"
1.3 编程者的正确、错误的提交，编程语言有C++、Python、Java
1.4 测试用例，含题目自带的样例测试用例（example test）和评分用的测试用例（hidden test cases）

数据样本如下：
![../images/9/ak6y8-oxno8.jpeg](../images/9/ak6y8-oxno8.jpeg)

2. 为了防止数据“泄漏”（将训练集用于模型测试），本文对整个数据作了如下划分：所有训练集都在GitHub提交的日期2021/07/14或其之前；验证集的提交在2021/07/15至2021/09/20期间；测试集的提交2021/09/21之后

* 模型架构

1. 基于seq2seq架构，对条件概率建模$p(Y | X)$，其中X为编程问题的描述（encoder的输入），Y为自回归的输出一个个代码token(decoder的输出)。

    ![../images/9/1644499579099.jpg](../images/9/1644499579099.jpg)

    本文还发现：使用浅层的encoder(层数少)和深层的decoder（层数多）的搭配可以极大改善模型的性能。
各种模型配置如下：

    ![../images/9/1644497004599.jpg](../images/9/1644497004599.jpg)

2. 使用JAX、Haiku工具建立模型

3. 使用multi-query attention：每个attention block使用全量的query heads，而key和value heads只使用一部分（共享key和value heads），这样可以减少内存和cache的使用，提高sampling的效率。

4. tokenize：使用SentencePiece tokenizer方法，使用GitHub和自身CodeContest数据集一共8000个token，encoder和decoder都使用相同的tokenizer

* Pre-training

使用GitHub的代码进行预训练。

encoder使用masked language modeling
decoder使用标准的交叉熵损失预测下一个token

将GitHub代码文件均匀切分为两部分，前半部分作为encoder的输入，后半部分用于decoder

* Fine-tuning

使用自身CodeContests数据进行模型微调。

同样，encoder使用masked language modeling
decoder使用标准的交叉熵损失预测下一个token。

encoder输入为问题的自然语言描述，而解题的代码用于decoder。

另外，本文还使用了以下技术作为改进：

1. Tempering:
```
Tempering, introduced by Dabre and Fujita (2020)([ Softmax tempering for training neural machine translation models.](https://arxiv.org/pdf/2009.09372.pdf)), is a regularization technique that
makes the token probability distribution artificially smoother or sharper at training time by dividing
the output logits of a model by a scalar temperature 푇 before the softmax layer
```

2. Value conditioning & prediction (强化学习)

数据集包含一道问题正确和错误的提交，本文使用Value conditioning & prediction区分这两类的提交。

在Value conditioning阶段，在问题描述中插入这个提交是否正确的信息，如下：

![../images/9/1644500206736.jpg](../images/9/1644500206736.jpg)

那么，在采样阶段，模型就总会采样到正确的sample。

而在Value prediction阶段，会加入一个辅助的预测任务（训练中才有），以便在一个小型的transformer中也可以使用最后一层token表示来分类这个提交正确与否：
```
we added an auxiliary value prediction
11
Competition-Level Code Generation with AlphaCode
task during training such that the last layer token representations before projecting to logits are also
used in a small Transformer to classify whether the submission is correct.
```


3. GOLD
(to be continue...)

### 研究结论


### 启示

1. multi-query attention对transformer计算性能的影响
2.  


### 附：

* 数据集：

[https://github.com/deepmind/code_contests](https://github.com/deepmind/code_contests)

[https://codeforces.com/](https://codeforces.com/)

* 应用：

[https://alphacode.deepmind.com/](https://alphacode.deepmind.com/)