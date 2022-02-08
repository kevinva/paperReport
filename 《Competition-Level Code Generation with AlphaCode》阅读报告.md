## 《Competition-Level Code Generation with AlphaCode》阅读报告

### 研究现状

现有的大型Language Model已经被证明可以用于生成代码，但是只能生成一些短小的代码片段，或者在解决复杂、不可见的、需要编程技巧的问题上，这些模型的性能还很弱。面临的挑战有：

1. 需要搜索大量的代码：
    1.1 仅仅改变单个字符就有可能改变整个程序的逻辑，即使这没有引起崩溃，所以代码的相关性不能仅仅依靠文本字面上的相关性；
    1.2 一题多解：同一个问题可能有多种编码逻辑可供选择
2. 衡量代码是否有效的测试用例通常是不可见的，需要提供一个有效的测评代码的基准。

现有的方案在生成大型程序代码上还不可靠，加上有效测试用例的缺乏，使得存在较高的假阳性率（false positive rate）。

本文的任务：

1. 提出一个生成代码的新新方案：AlphaCode，使用大型transformer模型，用GitHub上的代码pre-train，然后在精选过的编程竞赛问题上fine-tuning，总体流程如下：
![./images/9/1644312267351.jpg](./images/9/1644312267351.jpg)

2. 发布新的关于编程竞赛数据集CodeContests，用于模型训练：
![./images/9/ak6y8-oxno8.jpeg](./images/9/ak6y8-oxno8.jpeg)

3. 证明本文生成代码的方法并非直接复制训练集的代码片段，而是根据自然语言描述生成的。

### 研究方法


### 研究结论


### 附：

* 数据集：

[https://github.com/deepmind/code_contests](https://github.com/deepmind/code_contests)

[https://codeforces.com/](https://codeforces.com/)

* 应用：

[https://alphacode.deepmind.com/](https://alphacode.deepmind.com/)