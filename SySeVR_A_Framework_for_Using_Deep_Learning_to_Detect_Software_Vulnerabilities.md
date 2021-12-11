## 《SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities》阅读报告


### 研究现状

* 基于模式匹配的方法检测代码漏洞需要大量的人力劳动
* 现有静态分析系统普遍存在高假阴性率

而且，承接作者之前的论文《VulDeePecker: A Deep Learning-Based System for Vulnerability Detection》提到的不足：

* 只考虑与函数/API调用相关的漏洞
* 代码语义信息只利用数据间的依赖
* 只能用于BiLSTM训练学习
* 不能解析假阳性和假阴性的原因

基于以上，作者提出新的代码漏洞检测模型：SySeVC——Syntax-based, Semantics-based, and Vector Representation

### 研究目标

核心目标：根据语法和语义对代码进行特征提取，以用于漏洞检测（code2vec）


### 研究方法

受由图像识别中候选区域概念(region proposal)的启发，流程概览：

![./images/2/1639206985335.jpg](./images/2/1639206985335.jpg)

具体到漏洞检测：

![./images/2/1639207025301.jpg](./images/2/1639207025301.jpg)

整个系统是基于一个个函数/API实现进行的。

1. 提取SyVC

用静态分析工具（如Checkmatrx）生成一些漏洞语法特征（vulnerability syntax characteristics），如本文提到的4类：

* 函数/API调用（简称FC）
* 数组使用（简称AU）
* 指针使用（简称PU）
* 算术表达式（简称AE）

将程序段的每个语句转化为抽象语法树(AST)，遍历树中的每个节点，跟上述提到的各类漏洞语法特征进行“匹配”，如“匹配”成功，则认为这个节点对应的代码元素为SyVC。

譬如：

对于程序段：
```
void func()
{
    ...
    char source[100];
    source[99] = '\0';
    ...
}
```

可对应如下AST:
![./images/2/1639215168867.jpg](./images/2/1639215168867.jpg)

发现根据AU的匹配规则：
(i). 结点是一个IdentifierDeclStatement结点下的identifier
(ii). IdentifierDeclStatement结点含有“[”和“]”符号

则认为”source“是一个SyVC


========================
回顾AST:
![./images/2/hoho2.jpg](./images/2/hoho.jpg)
*(from: [https://code2vec.org/](https://code2vec.org/))*

========================

2. 将SyVC转为SeVC

先定义一些概念：
* 控制流图（CFG）: 它的节点是函数语句，边表示相邻语句间的运行先后关系。
* 数据依赖（data dependency）：如果CFG中有一条A->B的路径，且在A语句中计算得到的值会在B语句中使用，则称B数据依赖A。
* 控制依赖（control dependency）：如果CFG中有一条A->B的路径，且B是否执行需要看A执行的结果, 则称B控制依赖A。
* 程序依赖图（PDG）: 它的节点与CFG中节点表示意义意义，边为表示相邻语句间的数据依赖或控制依赖
* 前向切片（forward slice）：PDG中从SyVC节点出发所有可达节点的语句集合
* 过程间前向切片（interprocedural forward slice）：包含前向切片的所有语句，以及PDG中SyVC节点通过函数调用可以到达SyVC节点的语句（很绕啊。。。。顶！）
* 后向切片（backward slice）：PDG中所有可达SyVC节点的且以该节点为终点的语句集合
* 过程间后向切片（interprocedural backward slice）：包含后向切片的所有语句，以及PDG中通过函数调用可到达SyVC节点的语句（很绕啊。。。。顶！）
* 程序切片（program slice）：过程间前向切片和过程间后向切片的语句删除其中重复的部分的组合

具体流程如下图：

![./images/2/1639216946265.jpg](./images/2/1639216946265.jpg)

3. 将SeVC编码为向量

4. 模型训练与预测

本文作者主要使用GiGRU：

![./images/2/1639210145244.jpg](./images/2/1639210145244.jpg)

### 研究结论


### 优点

### 不足

### 个人启发




* *附：论文链接 [https://arxiv.org/pdf/1807.06756v3.pdf](https://arxiv.org/pdf/1807.06756v3.pdf)*