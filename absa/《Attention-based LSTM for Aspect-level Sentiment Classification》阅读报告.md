## 《Attention-based LSTM for Aspect-level Sentiment Classification》阅读报告


### 研究现状

1. 大部分方法只是检测整个句子的情感极性
2. 人工设计的特征耗费人力，而预约结果的好坏十分取决于特征的设计

提测关于注意力的方法，能关注句子某个方面的情感学习。


### 研究方法

* AE-LSTM(LSTM with Aspect Embedding)

对方面词进行嵌入

* AT-LSTM(Attention-based with LSTM)

![./images/7/1643640996581.jpg](./images/7/1643640996581.jpg)

以下列出技术注意力向量的方法：

$$
M = tanh(\begin{bmatrix}
W_hH \\ W_v v_a\bigotimes e_N
\end{bmatrix})
$$

$$
\alpha = softmax(w^TM) 
$$

$$
r = H\alpha ^T
$$

其中，
H为隐藏层向量矩阵H = [h_1, h_2, h_3, h_4, ..., h_N]

$e_N$为N维的1数值向量

那么，$v_a\bigotimes e_n = [v;v;v;...;v]$，即重复连接$v_a$N次

最后模型输出为：

$$h_* = tanh(W_pr + W_xh_N)$$
$$y = softmax(W_sh^* + b_s)$$


* ATAE-LSTM(Attention-based LSTM with Aspect Embedding)

![./iamges/7/1643643408987.jpg](./images/7/1643643408987.jpg)

类似于AT-LSTM，但输入向量为方面词嵌入和词陷入的拼接。


* 损失函数，使用交叉熵：

$$loss = -\sum_{i}^{} \sum_{j}^{} y_i^jlogy_i^j + \lambda \left\|\theta  \right\|^2$$

其中，i是句子的索引，j是分类的索引

* 输入的词向量使用Glove方法


### 研究结论

* 准确度对比：

![./images/7/WechatIMG70.jpeg](./images/7/WechatIMG70.jpeg)

* 注意力可视化：句子的特定方面对句子中各个单词的关注程度：

![./images/7/WechatIMG71.jpeg](./images/7/WechatIMG71.jpeg)

### 附

* 论文链接： [https://aclanthology.org/D16-1058/](https://aclanthology.org/D16-1058/)

* 数据集：[https://alt.qcri.org/semeval2014/task4/](https://alt.qcri.org/semeval2014/task4/)