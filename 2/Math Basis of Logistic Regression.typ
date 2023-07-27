// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", authors: (), body) = {
  // Set the document's basic properties.
  set document(author: authors, title: title)
  set page(numbering: "1", number-align: center)
  set text(font: "Linux Libertine", lang: "zh")

  // Title row.
  align(center)[
    #block(text(weight: 700, 1.75em, title))
  ]

  // Author information.
  pad(
    top: 0.5em,
    bottom: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center, strong(author))),
    ),
  )

  // Main body.
  set par(justify: true)

  body
}

#show: project.with(
  title: "Math Basis of Logistic Regression",
  authors: (
    "周陆延",
  ),
)

#let yhat = $accent(y, hat)$

+ 给出利用梯度下降法求解逻辑回归的前反向公式推导，在这一部分中，你只需要考虑最朴素的二分类逻辑回归模型

  - 前向传播
  
    线性回归 $ f(x) = omega^top x + b $
    
    令 $x' = (1, x)$ 则 $f(x)$ 可统一为 $ f(x) = omega'^top x $
    
    为处理二分类问题将线性回归 $(-infinity, +infinity)$ 的值域转换为 $(0, 1)$ 对概率进行预测，嵌套 $sigma(x) = 1/(1+e^(-x))$ 函数
    
    即逻辑回归的前向传播公式 $ h_omega (x) = 1/(1 + e^(-omega^top x)) $
  
  - 反向传播
    
    根据逻辑回归的极大似然估计设置交叉熵损失函数 $ J(omega) = -1/m sum_(i=1)^m y log accent(y, hat) + (1-y)log(1-accent(y, hat)) $
  
    求梯度得 $  (delta J(omega)) / (delta omega) &= -1/m sum_(i=1)^m y 1/accent(y, hat) yhat' + (1-y) 1/(1-yhat)yhat' \
    &= -1/m sum_(i=1)^m (y/yhat + (1-y)/(1-yhat))yhat' \
    &= -1/m sum_(i=1)^m (sigma(omega^top x^((i))) - y^((i)))x^((i)) $
  
    使用梯度下降法更新 $omega := omega - alpha (delta J(omega)) / (delta omega)$ 其中 $alpha$ 为学习率

+ 假设标签集不是 ${0,1}$ 而是 ${1,-1}$，将会有什么变化，请给出推导

  似然函数变为
  $ L(omega)=sqrt(Pr(x)^(1+y)(1-Pr(x))^(1-y)) $

  取负对数得到损失函数
  $ J(omega) = -1/2 ((1+y)log yhat + (1-y) log (1-yhat)) $

  对 $y=1,y=-1$ 分类讨论得
  $ J(omega) = log (1+e^(y*omega^top x)) $

+ 在问题2.1的基础上，即标签集为 {0,1} 的情况，分别增加L1正则化和L2正则化，公式和模型效果分别会有什么变化，请给出推导

  最基本的正则化方法是在原目标（代价）函数中添加惩罚项，对复杂度高的模型进行“惩罚”，即 $ accent(J, tilde)(x) = J(x) + lambda Omega(omega) $

  正则化可理解为对原损失函数最优化过程添加约束 $ min_omega J(omega) \ s.t. Omega(omega) <= C $ 利用拉格朗日算子法，我们可将上述带约束条件的最优化问题转换为不带约束项的优化问题，构造拉格朗日函数 $ L(omega, lambda)=J(omega) + lambda (Omega(omega)-C) $ 设 $lambda$ 最优解为 $lambda^*$ 则对拉格朗日函数最小化等价于 $ min_omega J(omega) + lambda^* Omega(omega) $

  - L2 正则化

    即使用L2范数作为惩罚 $ J(omega) = -1/m sum_(i=1)^m y log accent(y, hat) + (1-y)log(1-accent(y, hat)) + lambda/(2m)||omega||_2^2 $

    梯度则变为 $ (delta J(omega)) / (delta omega) &= -1/m (sum_(i=1)^m (sigma(omega^top x^((i))) - y^((i)))x^((i))) + lambda/m omega $

    考虑对模型的影响，令 $omega^*$ 为未正则化的目标函数的最优解，对 $J(omega)$ 作二阶泰勒展开近似（$omega^*$ 为最优，无一阶导项；略去样本数量 $m$） $ accent(J, hat)(omega) = J(omega^*) + 1/2(omega-omega^*)^top H (omega-omega^*) $
    
    当 $accent(J, hat)(omega)$ 最小时，其梯度为 $ (delta accent(J, hat)(omega)) / (delta omega) = H(omega-omega^*) = 0 $

    加入惩罚项，记此时的最优解为 $omega'$ 得 $ H(omega'-omega^*) + lambda omega' = 0 \
    omega' = (H + lambda I)^(-1)H omega^*
    $

    由 $H$ 实对称，将其合同到对角矩阵 $H=Q Lambda Q^top$ 带入上式得 $ omega' &= (Q Lambda Q^top+lambda I)^(-1) Q Lambda Q^top omega^* \
    &= (Q Lambda Q^top+Q (lambda I) Q^top)^(-1) Q Lambda Q^top omega^* \
    &= (Lambda + lambda I)^(-1) Lambda omega^* $

    发现 $omega'$ 相比 $omega^*$ 是依据 $H$ 的特征值在对应分量上做了缩放，在 $H$ 特征值较大的方向影响较小，在特征值较小的方向影响较大，使对减少目标函数作用显著的参数被保留，作用微弱的参数被衰减

  - L1 正则化

    即使用L1范数作为惩罚 $ J(omega) = -1/m (sum_(i=1)^m y^((i)) log yhat^((i)) + (1-y^((i)))log(1-yhat^((i)))) + lambda/m||omega||_1 $
  
    梯度则变为 $ (delta J(omega)) / (delta omega) &= -1/m (sum_(i=1)^m (sigma(omega^top x^((i))) - y^((i)))x^((i))) + lambda/m"sgn"(w) $

    与L2正则化类比，但是由于 $"sgn"(x)$ 的特殊性，进一步假设 $H$ 为对角阵（对数据进行主成分分析后成立），则
    
    $ accent(accent(J, tilde), hat)(omega) &= J(omega^*) + 1/2(omega-omega^*)^top H (omega-omega^*) + lambda||omega||_1 \
     &= J(omega^*) + sum_i (1/2 H_(i, i) (omega_i-omega_i^*)^2 + lambda |omega_i|) $

    令
    $ accent(accent(J, tilde), hat)'(omega) = sum_i H_(i,i)(omega_i - omega_i^*) + lambda"sgn"(omega_i) = 0 $

    得
    $ omega_i = "sgn"(omega^*_i) max{|omega^*|-lambda/H_(i,i), 0} $

    可见当 $|omega^*|<=lambda/H_(i,i)$ 时会使得 $omega_i$ 变成 $0$，使得参数稀疏化

+ 给出核逻辑回归的对偶形式

  $
  min_alpha & 1/(2lambda) sum_(i=1)^m sum_(j=1)^m alpha_i alpha_j y_i y_j x_i^top x_j + sum_(i=1)^m [alpha_i log alpha_i + (1-alpha_i)log (1- alpha_i)] \
  "s.t." &0 <= alpha_i < 1  (i in [m])
  $