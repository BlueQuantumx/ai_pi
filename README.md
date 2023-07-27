这是我申请加入 HUST 校内 AIπ 团队的第一轮测试题目，题目包含了机器学习的基础知识，以及一些基础的机器学习算法。

以下为进行简化与必要的隐私处理后的题目：

---

## 问题背景

### Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines

### Overview

**Can you predict whether people got H1N1 and seasonal flu vaccines using information they shared about their backgrounds, opinions, and health behaviors?**

In this challenge, we will take a look at vaccination, a key public health measure used to fight infectious diseases. Vaccines provide immunization for individuals, and enough immunization in a community can further reduce the spread of diseases through "herd immunity".

Vaccines for the COVID-19 virus are still under development and not yet available. Beginning in spring 2009, a pandemic caused by the H1N1 influenza virus, colloquially named "swine flu", swept across the world. Researchers estimate that in the first year, it was responsible for between [151,000 to 575,000 deaths globally](https://www.cdc.gov/flu/pandemic-resources/2009-h1n1-pandemic.html).

A vaccine for the H1N1 flu virus became publicly available in October 2009. In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission. A better understanding of how these characteristics are associated with personal vaccination patterns can provide guidance for future public health efforts.

**This is a practice designed to be accessible to participants at all levels. That makes it a great place to dive into the world of data science competitions. Come on in from the waiting room and try your (hopefully steady) hand at predicting vaccinations.**

### Problem description

Your goal is to predict how likely individuals are to receive their H1N1 and seasonal flu vaccines. Specifically, you'll be predicting two probabilities: one for `h1n1_vaccine` and one for `seasonal_vaccine`.

Each row in the dataset represents one person who responded to the National 2009 H1N1 Flu Survey.

### Labels

For this competition, there are two target variables:

- `h1n1_vaccine` - Whether respondent received H1N1 flu vaccine.
- `seasonal_vaccine` - Whether respondent received seasonal flu vaccine.

Both are binary variables: `0` = No; `1` = Yes. Some respondents didn't get either vaccine, others got only one, and some got both. This is formulated as a multilabel (and *not* multiclass) problem.

### The features in this dataset

You are provided a dataset with 36 columns. The first column `respondent_id` is a unique and random identifier. The remaining 35 features are described below.

For all binary variables: `0` = No; `1` = Yes.

- `h1n1_concern`\- Level of concern about the H1N1 flu.
  - `0` = Not at all concerned; `1` = Not very concerned; `2` = Somewhat concerned; `3` = Very concerned.
- `h1n1_knowledge`\- Level of knowledge about H1N1 flu.
  - `0` = No knowledge; `1` = A little knowledge; `2` = A lot of knowledge.
- `behavioral_antiviral_meds` - Has taken antiviral medications. (binary)
- `behavioral_avoidance` - Has avoided close contact with others with flu-like symptoms. (binary)
- `behavioral_face_mask` - Has bought a face mask. (binary)
- `behavioral_wash_hands` - Has frequently washed hands or used hand sanitizer. (binary)
- `behavioral_large_gatherings` - Has reduced time at large gatherings. (binary)
- `behavioral_outside_home` - Has reduced contact with people outside of own household. (binary)
- `behavioral_touch_face` - Has avoided touching eyes, nose, or mouth. (binary)
- `doctor_recc_h1n1` - H1N1 flu vaccine was recommended by doctor. (binary)
- `doctor_recc_seasonal` - Seasonal flu vaccine was recommended by doctor. (binary)
- `chronic_med_condition` - Has any of the following chronic medical conditions: asthma or an other lung condition, diabetes, a heart condition, a kidney condition, sickle cell anemia or other anemia, a neurological or neuromuscular condition, a liver condition, or a weakened immune system caused by a chronic illness or by medicines taken for a chronic illness. (binary)
- `child_under_6_months` - Has regular close contact with a child under the age of six months. (binary)
- `health_worker` - Is a healthcare worker. (binary)
- `health_insurance` - Has health insurance. (binary)
- `opinion_h1n1_vacc_effective`\- Respondent's opinion about H1N1 vaccine effectiveness.
  - `1` = Not at all effective; `2` = Not very effective; `3` = Don't know; `4` = Somewhat effective; `5` = Very effective.
- `opinion_h1n1_risk`\- Respondent's opinion about risk of getting sick with H1N1 flu without vaccine.
  - `1` = Very Low; `2` = Somewhat low; `3` = Don't know; `4` = Somewhat high; `5` = Very high.
- `opinion_h1n1_sick_from_vacc`\- Respondent's worry of getting sick from taking H1N1 vaccine.
  - `1` = Not at all worried; `2` = Not very worried; `3` = Don't know; `4` = Somewhat worried; `5` = Very worried.
- `opinion_seas_vacc_effective`\- Respondent's opinion about seasonal flu vaccine effectiveness.
  - `1` = Not at all effective; `2` = Not very effective; `3` = Don't know; `4` = Somewhat effective; `5` = Very effective.
-  `opinion_seas_risk`\- Respondent's opinion about risk of getting sick with seasonal flu without vaccine.
  - `1` = Very Low; `2` = Somewhat low; `3` = Don't know; `4` = Somewhat high; `5` = Very high.
-  `opinion_seas_sick_from_vacc`\- Respondent's worry of getting sick from taking seasonal flu vaccine.
  - `1` = Not at all worried; `2` = Not very worried; `3` = Don't know; `4` = Somewhat worried; `5` = Very worried.
- `age_group` - Age group of respondent.
- `education` - Self-reported education level.
- `race` - Race of respondent.
- `sex` - Sex of respondent.
- `income_poverty` - Household annual income of respondent with respect to 2008 Census poverty thresholds.
- `marital_status` - Marital status of respondent.
- `rent_or_own` - Housing situation of respondent.
- `employment_status` - Employment status of respondent.
- `hhs_geo_region` - Respondent's residence using a 10-region geographic classification defined by the U.S. Dept. of Health and Human Services. Values are represented as short random character strings.
- `census_msa` - Respondent's residence within metropolitan statistical areas (MSA) as defined by the U.S. Census.
- `household_adults` - Number of *other* adults in household, top-coded to 3.
- `household_children` - Number of children in household, top-coded to 3.
- `employment_industry` - Type of industry respondent is employed in. Values are represented as short random character strings.
- `employment_occupation` - Type of occupation of respondent. Values are represented as short random character strings.

### Performance metric

Performance will be evaluated according to the area under the receiver operating characteristic curve (ROC AUC) for each of the two target variables. The mean of these two scores will be the overall score. A higher value indicates stronger performance.

### Submission format

Please refer to `submission_format.csv` for the correct format. 
The predictions for the two target variables should be **float probabilities** that range between `0.0` and `1.0`. Because the competition uses ROC AUC as its evaluation metric, the values you submit must be the probabilities that a person received each vaccine, *not* binary labels.

### Good luck!

## 问题描述

### 1 数据处理

​	真实世界的数据往往包含缺失值、异常值和一些无法直接用于机器学习模型的特征，数据预处理通常是必不可少的环节。这一题我们希望你利用pandas,numpy,matplotlib,完成以下要求：

#### 具体要求

- 将训练集的feature和label两个csv文件按respondent_id合并以利于后续处理；
- 对缺失值进行合适的处理；
- 将分类属性进行OneHot编码；
- 对数值属性进行必要的操作，如归一化处理等；
- 可视化分析数据，展示数据分布，发现规律；
- 除此之外，你还可以进行必要的数据探索，如计算相关性等等。

#### 提示

参考资料

- （美）麦金尼（McKinney W.）著；唐学韬译. 利用Python进行数据分析 [M]. 北京：机械工业出版社, 2016.01.

- [NumPy官方网站](https://numpy.org)
- 机器学习实战：基于Scikit-Learn和TensorFlow

### 2 数学原理之LR

​	在后续的问题中，你可能需要使用逻辑回归解决问题。由于逻辑回归的原理十分经典，这部分内容将考察你对其中数学原理的理解程度，你需要使用简单的数学公式回答以下问题：

#### 具体要求

1. 给出利用**梯度下降法**求解**逻辑回归**的**前反向公式推导**，在这一部分中，你只需要考虑**最朴素的二分类逻辑回归模型**。
2. 假设标签集不是 {0,1} 而是 **{1,-1}**，将会有什么变化，请给出推导。
3. 在问题2.1的基础上，即标签集为 {0,1} 的情况，分别增加**L1正则化**和**L2正则化**，公式和模型效果分别会有什么变化，请给出推导。
4. 给出**核逻辑回归的对偶形式**。
5. （选做）如果你愿意，给出利用**二阶优化算法**如牛顿法求解逻辑回归的公式推导，只需要考虑最朴素的逻辑回归模型。

#### 提示

- 除了问题2.4以外，请给出关键推导过程，仅仅给出结果是无效的。
- 如果你愿意，问题2.4可以给出关键性的公式推导或解释。
- 建议使用 LaTeX 语法书写数学公式，例如你可以很容易查询到，梯度公式如下（假设数据下标从1到m）:

$$
g(\boldsymbol w)=\frac{\partial J(\boldsymbol w)}{\partial \boldsymbol w}=\sum_{i=1}^m(\sigma(\boldsymbol w^\top\boldsymbol x_i)-y_i)\boldsymbol x_i
$$

- 参考资料：李航.统计学习方法[M].北京:清华大学出版社,2019.5.1

### 3 数学原理之SVM

​	在后续的问题中，你可能需要使用支持向量机解决问题。SVM的部分原理如下。

- 支持向量机原问题：

$$
\min_{\boldsymbol{w},b}f(\boldsymbol{w})=\frac{1}{2}\|\boldsymbol{w}\|_2^2,\quad\mathrm{ s.t. }\ y_i(\boldsymbol{w}^\top\boldsymbol{x}_i+b)\ge 1,\forall 1\le i\le m
$$

- 支持向量机对偶问题：

$$
\max_{\boldsymbol{\alpha}\ge 0}g(\boldsymbol{}\alpha) =-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_j y_iy_j\boldsymbol x_i^\top\boldsymbol{x}_j+\sum_{i=1}^m \alpha_i,\quad \mathrm{s.t.}\ \boldsymbol{y}^\top\boldsymbol\alpha=0
$$

你需要使用简单的数学公式回答以下问题：

#### 具体要求

1. 给出支持向量机最大间隔准则原问题的推导，解释分类超平面和支持超平面的含义。
2. 给出对偶问题的推导，并回答问题：强对偶性在支持向量机中始终成立吗？
3. 根据前面的结果推导**SVM的KKT条件**。
4. 如果数据非线性可分怎么办？给出修改后的原问题和对偶问题。
5. 在问题3.4的基础上，若**允许少量样本破坏约束**，应增加怎样的损失函数，请给出修改后的原问题和对偶问题。

#### 提示

- 可以直接使用点到超平面的距离公式。可以认为你已经掌握拉格朗日乘子法，无需对此再进行证明推导。
- 问题3.2中回答问题只需要回答是否，如果你愿意，也可以给出简单的解释。
- 除了问题3.4与问题3.5以外，请给出推导过程，仅仅给出结果是无效的。
- 如果你愿意，问题3.4与问题3.5可以给出关键性的公式推导或解释。
- 建议使用 LaTeX 语法书写数学公式。
- 参考资料：李航.统计学习方法[M].北京:清华大学出版社,2019.5.1

### 4 逻辑回归

​	逻辑回归是一种非常适合入门的二分类机器学习方法。请在问题1的基础上，使用pandas,numpy,matplotlib库对以下问题进行解答：

#### 具体要求

- 实现最基本的逻辑回归，使用批量梯度下降，调整学习率和迭代次数，分析梯度的收敛情况。在训练集上进行十折交叉验证，分析不同测试集划分方式对结果的影响，考虑是否有过拟合现象。

- 给逻辑回归添加正则项，观察过拟合现象是否减轻。

- 实现随机梯度下降，对比与批量梯度下降的优劣，并简要说明改进方法。

- （选做）可视化训练过程。

- （选做）尝试实现Adam算法。


#### 提示

- 由于缺少测试集的label，所以对模型的评估在训练集上使用十折交叉验证(仅在此允许调用sklearn的train_test_split或者KFold实现)

- 提交上述一系列操作后你认为表现效果最好的测试集上的结果（以Submission format为准，注意不是0-1二分类）

参考资料：

https://www.coursera.org/specializations/machine-learning-introduction

[《动手学深度学习》 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/)


### 5 决策树

​		决策树（Decision Tree）是一类常见的机器学习方法。请你在问题1的基础上，通过**numpy**和**pandas**库使用决策树算法完成对问题背景中的问题的求解。

#### 具体要求

- 对问题1处理后的数据集，根据**信息增益**准则选择最优特征。
- 使用**CART算法**生成决策树。
- 使用**CART剪枝**算法进行剪枝。
- 实现对树的**可视化**。

#### 注意

- 最终结果的格式，以**问题背景**中**Submission format**为准，但我们以**源代码**为主要评分标准。

#### 提示

- 李航.统计学习方法[M].北京:清华大学出版社,2019.5.1

### 6 支持向量机

​	支持向量机(Support Vector Machines，SVM)是一种二类分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机;支持向量机还包括核技巧，这使它成为实质上的非线性分类器。请你在问题1的基础上，通过**numpy**和**pandas**库使用支持向量机完成对问题背景中的问题的求解。

#### 具体要求

- 对问题1处理后的数据集，使用**合页损失函数**表示的**线性支持向量机**，使用**梯度下降法**进行求解。
- 选择**常用的核函数**，使用**SMO算法**，使用**非线性支持向量机**进行求解。

#### 注意

- 最终结果的格式，以**问题背景**中**Submission format**为准，但我们以**源代码**为主要评分标准。

#### 提示

- 寻找一个函数，将**几何间隔**映射为**概率输出**。
- 李航.统计学习方法[M].北京:清华大学出版社,2019.5.1

### 7 基于NumPy的MLP实现

​		神经网络是一种受到人类大脑神经元结构启发而设计的计算模型，其强大的特征提取和模式识别能力使得神经网络成为机器学习和人工智能领域中的重要技术之一，在各个领域具有广泛的应用。在这里，我们希望带你快速入门实现神经网络的全过程，虽然以后可能不会再从头手写神经网络，但是理解其中的算法会让你在使用神经网络时更有底气。

- 利用NumPy库实现全连接神经网络，层数不限，结点数不限（可以采用多种激活函数和配套的前向后向传播算法）按照实现的情况评分，同时评判不同激活函数的优劣，并说明理由。

​		**具体要求（实现流程）：**

​			a)    初始化神经网络的权重和偏置项；

​			b)   实现前向传播函数，接受输入数据并计算网络的输出；

​			c)    实现反向传播函数，利用训练数据和输出误差更新权重和偏置项；

​			d)   使用梯度下降算法训练网络，迭代进行权重和偏置项的更新，直到达到收敛条件或预设的训练次数；

​			e)    测试网络性能，使用测试数据计算准确率或其他评估指标。

​		**加分项：** 可视化训练过程；使用多折交叉验证；模型优化。

-  利用实现好的全连接神经网络对**问题背景**中的数据集进行处理和预测，根据测试分数进行评分。

​		**具体要求：**

​			a)    在处理好的数据集上可以跑，不报错；

​			b)   测试达到一定的分数。

- 参考资料： [Machine Learning Glossary — ML Glossary documentation (ml-cheatsheet.readthedocs.io)](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)

### 8 基于Pytorch的深度学习

​		PyTorch 是一个开源的深度学习框架，由 Facebook 的人工智能研究团队开发并维护。它结合了灵活的动态计算图和丰富的神经网络模块，使得开发者能够轻松地构建和训练各种深度学习模型。目前，Pytorch已经成为主流科研和工业使用的深度学习编程框架。在这里，我们希望带你快速入门Pytorch的使用，Pytorch对于深度学习的算法实现来说只是一个工具，所以不必畏难。

- 利用Pytorch实现一个全连接神经网络，按照实现的情况评分。

​		**具体要求（实现流程）：**

​			a)    定义全连接神经网络模型；

​			b)   设定超参数，创建模型实例；

​			c)    定义损失函数和优化器；

​			d)   准备输入数据和输出数据；

​			e)    训练模型；

​			f)    使用训练好的模型进行预测。

​		**加分项：** 可视化训练过程；利用GPU进行训练；模型优化。

- 利用实现好的全连接神经网络对**问题背景**中的数据集进行处理和预测，根据测试分数进行评分。

  **具体要求：**

  ​	a)    在处理好的数据集上可以跑，不报错；

  ​	b)   测试达到一定的分数。

参考资料： [PyTorch documentation — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/index.html)
