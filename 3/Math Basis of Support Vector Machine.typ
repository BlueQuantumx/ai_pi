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
  title: "Math Basis of Support Vector Machine",
  authors: (
    "周陆延",
  ),
)

#set math.equation(numbering: "(1)", supplement: [式])
#let x = $bold(x)$
#let omega = $bold(omega)$
#let b = $bold(b)$
#let alphab = $bold(alpha)$

+ 给出支持向量机最大间隔准则原问题的推导，解释分类超平面和支持超平面的含义
  
  分类超平面为任意将样本划分为两个部分的超平面
  
  支持超平面则是距离两侧支持向量距离之和最大的分类超平面
  
  记样本点 $#x$ 到超平面 $(omega, b)$ 的距离为 $ r=|omega^top #x + b| / ||omega|| $
  
  假设超平面能将样本分类则 $exists space bold((omega, b)) "s.t." forall i$
  
  $
  cases(
    omega^top #x _i + b >= +1 "if" y_i = +1,
    omega^top #x _i + b <= -1 "if" y_i = -1,
  )
  $
  
  则称刚好使等号成立的样本点为支持向量，两个异类支持向量到超平面的距离和 $gamma = 2/(||omega||_2^2)$ 为间隔，要使 $gamma$ 最大，即
  
  $
  &max 2/(||omega||_2^2) = min 1/2 ||omega||_2^2 \
  &"s.t." y_i (omega^top #x _i + b) >= 1
  $

+ 给出对偶问题的推导，并回答问题：强对偶性在支持向量机中始终成立吗？

  对原问题应用拉格朗日乘数法

  $
  &L(omega, b, alphab) = 1/2 ||omega||_2^2 + sum_(i=1)^m alpha_i [1 - y_i (omega^top #x _i + b)] \
  &"s.t." alphab >= bold(0)
  $

  首先证明原问题 $arrow.l.r.double$
  
  $
  &min_(omega, b) max_alphab L(omega, b, alphab) \
  &"s.t." alpha >= bold(0)
  $

  证：
  
  $
  //because
  cases(
    "if" (omega, b) "在可行域内" space max_alphab L(omega, b, alphab) = 1/2 ||omega||_2^2 + 0 + 0,
    "if" (omega, b) "不在可行域内" space max_alphab L(omega, b, alphab) = 1/2 ||omega||_2^2 + infinity + infinity
  )
  $
  
  然后记对偶函数
  
  $
  g(alphab) = min_(omega, b) L(omega, b, alphab)
  $

  对偶问题即为

  $
  &max_alphab min_(omega, b) L(omega, b, alphab) \
  &"s.t." alphab >= bold(0)
  $

  求偏导得

  $
  (delta L) / (delta omega) = 0 arrow.r.double omega = sum_(i=1)^m alpha_i y_i #x _i \
  (delta L) / (delta b) = 0 arrow.r.double sum_(i=1)^m alpha_i y_i = 0
  $

  带回原式即为

  $
  max g(alphab) &= 1/2 sum_(i=1)^m sum_(j=1)^m alpha_i y_i alpha_j y_j #x _i^top #x _j + sum_(i=1)^m alpha_i [1 - y_i ((sum_(i=1)^m alpha_i y_i #x _i^top) #x _i + b)] \
  &= 1/2 sum_(i=1)^m sum_(j=1)^m alpha_i y_i alpha_j y_j #x _i^top #x _j + sum_(i=1)^m (alpha_i - sum_(i=1)^m alpha_i alpha_j y_i y_j #x _j^top #x _i - alpha_i y_i b) \
  &= -1/2 sum_(i=1)^m sum_(j=1)^m alpha_i alpha_j y_i y_j #x _i^top #x _j + sum_(i=1)^m alpha_i \
  "s.t." sum_(i=1)^m alpha_i &y_i = 0
  $

  由支持向量机本身为凸优化问题且满足 Slater 条件（一定 $exists (omega, b) "s.t."$ 某些点在 margin 内部），得强对偶性在支持向量机中始终成立

+ 根据前面的结果推导SVM的KKT条件

  $
  cases(
    alpha_i >= 0,
    y_i (omega^top #x _i + b) >= 1,
    alpha_i (y_i (omega^top #x _i + b) - 1) = 0,
  )
  $

+ 如果数据非线性可分怎么办？给出修改后的原问题和对偶问题

  使用核技巧将数据映射到高维空间再进行划分

  修改后的原问题：

  $
  &min_(omega, b) 1/2 ||omega||_2^2 \
  &"s.t." y_i (omega^top phi.alt(#x _i) + b) >= 1
  $

  修改后的对偶问题：

  $
  &max_alphab sum_i^m alpha_i - 1/2 sum_(i=1)^m sum_(j=1)^m alpha_i alpha_j y_i y_j kappa(#x _i, #x _j) \
  &"s.t." sum_(i=1)^m alpha_i y_i = 0
  $

  其中 $phi.alt(#x)$ 是映射后的特征向量， $kappa(#x _i, #x _j)$ 是两个向量在高维空间里的内积

+ 在问题3.4的基础上，若允许少量样本破坏约束，应增加怎样的损失函数，请给出修改后的原问题和对偶问题

  应增加形如

  $
  C sum_i^m cal(l)_(0\/1)(y_i (omega^top #x _i + b))
  $

  的项，其中 $cal(l)_(0\/1)$ 是 0/1 损失函数

  $
  cal(l)_(0\/1)(z) = cases(
    1 "if" z < 1,
    0 "otherwise",
  )
  $

  但考虑到 $cal(l)_(0\/1)(z)$ 较劣的数学性质，实际应用中通常寻找一些代替函数，如：

  $
  cal(l)_"hinge" (z) = max (0, 1 - z) \
  cal(l)_exp (z) = exp (-z) \
  cal(l)_log (z) = log (1 + exp (-z)) \
  $

  下面对选用 $cal(l)_"hinge" (z)$ 的原问题与对偶问题进行推导

  原问题

  $
  &min 1/2 ||omega||_2^2 + C sum_i^m max (0, 1 - y_i (omega^top #x _i + b)) \
  &"s.t." y_i (omega^top #x _i + b) >= 1
  $ <origin>

  引入松弛变量 $xi_i >= 0$ 重写 @origin 得到修改后的原问题

  $
  &min 1/2 ||omega||_2^2 + C sum_i^m xi_i \
  &"s.t." cases(
    y_i (omega^top #x _i + b) >= 1 - xi_i,
    xi_i >= 0
  )
  $

  对偶问题为

  $
  &max_alphab sum_i^m alpha_i - 1/2 sum_(i=1)^m sum_(j=1)^m alpha_i alpha_j y_i y_j #x _i^top #x _j \
  &"s.t." cases(
    sum_(i=1)^m alpha_i y_i = 0,
    0 <= alpha_i <= C
  )
  $

  其中和 Hard-Margin SVM 的主要区别在于

  $
  triangle.b_bold(xi) L(omega, b, alphab, bold(xi), bold(mu)) = C - alphab - bold(mu) = bold(0) \
  "又" cases(
    alpha_i >= 0, mu_i >= 0
  )\
  arrow.double 0 <= alpha_i <= C
  $
