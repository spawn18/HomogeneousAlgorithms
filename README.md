# Homogeneous algorithms testing framework

## Black-box function optimization

In optimization we often encounter functions whose properties are highly unknown.
We don't know anything about the graph, local minima, range of values, etc. 
What we do have is the rule that allows us to calculate function values.
Such functions seem to be susceptible to brute force only.
Because optimizations highly depend on the information given, we still need assumptions to make progress.
One of the common assumptions for black-box functions is their **Lipschitz continuity**.

## Homogeneous algorithms

An optimization algorithm is called **homogeneous** if it generates the same test points for functions in the set  

$`\{ f(x) + c \mid c \in \mathbb{R} \}`$

Homogeneous algorithms are independent of the range of values.

There are several approaches, the most common is the following.  
Let $`D=[a,b]`$ be the domain and $`M = \{x_0, \dots, x_n\}`$ the set of starting points. Then optimization goes like this:

1. $`\forall x_i \in M`$ calculate $`f(x_i)`$, set $`k := n`$
2. Build $`m_k(x), s_k(x)`$
3. Build $`P_k(x) = m_k(x) - s_k(x)`$
4. Calculate $`x_{k+1} = \arg\min\limits_{x\in D} P_k(x)`$
5. Calculate $`f(x_{k+1})`$
6. If some condition $`\phi(x \mid x_0, \dots, x_k)`$ is met then stop, else set $`k := k+1`$ and go to step 2.  

Take  

```math
x^* = \arg\min\limits_{i=\overline{1,k}} f(x_i), \quad f^* = f(x^*)
```

as the solution.

### Modeling process

$`m(x)`$ interpolates the target function:

1. $`m(x_i) = f(x_i), \ i = \overline{1,k}`$
2. $`m(x) \in \text{Lip}(L_m)`$
3. 

```math
m(x_i \mid x_1, f(x_1)+c, \dots, x_k, f(x_k)+c) = 
m(x_i \mid x_1, f(x_1), \dots, x_k, f(x_k)) + c
```

$`s(x)`$ is the measure of uncertainty. Intuitively it dictates how much we do not know about the target function at a given point:

1. $`s(x_i) = 0`$
2. $`s(x) \geq 0`$
3. $`s(x) \in \text{Lip}(L_s)`$
4. 

```math
s(x \mid x_1, f(x_1)+c, \dots, x_k, f(x_k)+c) = 
s(x \mid x_1, f(x_1), \dots, x_k, f(x_k))
```

$`P(x)`$ is the minimum criteria. Its global minimum dictates the point of maximum uncertainty in the model, thus giving a potential candidate point to evaluate.

## Lipschitz constant estimation

To perform optimization the Lipschitz constant must be known, since the function graph can easily be bounded.
If it is known then optimization becomes significantly faster. The speed depends on whether it is given globally or locally, because the latter is highly preferable.
If the constant is *not* known (which is the most common case), then it can be estimated at each iteration.

After points $`\{ (x_i,y_i) \}_{i=0}^{k}`$ have been evaluated, for each interval $`[x_{i-1}, x_i], \ i=\overline{1,k},`$ calculate

```math
\lambda_i = \max \left\{ \frac{|y_j - y_{j-1}|}{x_j - x_{j-1}} \ \bigg| \ j=i-1,i,i+1 \right\}
```

```math
\lambda^{\max} = \max\limits_{1\leq i\leq k}\frac{|y_i-y_{i-1}|}{x_i-x_{i-1}}
```

```math
X^{\max} = \max\limits_{1\leq i\leq k} \{ x_i - x_{i-1}\}
```

```math
\gamma_i = \lambda^{\max} \frac{(x_i - x_{i-1})}{X^{\max}}
```

```math
H_i = \max \{\gamma_i, \lambda_i, \xi\}
```

For $`i=1`$ or $`i=k`$ in $`\lambda_i`$, only $`j=i,i+1`$ and $`j=i-1,i`$ are used respectively.  

The Lipschitz constant estimate on each interval is  

```math
\mu_i = rH_i, \quad i=\overline{1,k}
```

where $`r>1`$ is a "reserve" parameter. It is recommended to choose $`1.1 \leq r \leq 1.4`$.

## Algorithms

### NL[^1]
Classic Piyavskiy[^2] algorithm but without Lipschitz constant known *a priori*.  
It is estimated during optimization.

### CubicSplineGrad
$`m(x)`$ is a cubic spline interpolator (clamped edge constraints).

1. $`x_0 = f(a), \ x_1 = f(b), \ k := 2`$
2. Estimate Lipschitz constants $`\mu_i`$ on each interval
3. Interpolate points $`\{ (x_i,y_i) \}_{i=0}^{k}`$ with $`m(x)`$
4. For every interval $`[x_{i-1}, x_i], \ i=\overline{1,k}`$ calculate  

```math
m_{\text{left}}^i = m_k'(x_{i-1}), \quad
m_{\text{right}}^i = m_k'(x_i)
```

5. Smooth derivative values with $`\theta(x)`$ for every interval  

```math
\theta_{\text{left}}^i = \theta(m_{\text{left}}^i), \quad
\theta_{\text{right}}^i = \theta(m_{\text{right}}^i)
```

6. For every interval calculate  

```math
Q=\frac{y_i-y_{i-1}}{x_i-x_{i-1}}
```

- If $`Q \geq 0`$:  

```math
\mu_{\text{left}} = \max \{ \theta_{\text{left}}^i \mu_i, \xi\}, \quad
\mu_{\text{right}} = \max \{ \theta_{\text{right}}^i \mu_i, Q+\xi\}
```

- Else:  

```math
\mu_{\text{left}} = \max \{ \theta_{\text{left}}^i \mu_i, -Q+\xi\}, \quad
\mu_{\text{right}} = \max \{ \theta_{\text{right}}^i \mu_i, \xi\}
```

Obtain  

```math
s_k(x) = \min\limits_{i=\overline{1,k}} \{ \mu_{\text{left}}(x-x_{i-1}), \ \mu_{\text{right}}(x_i-x)\}
```

7. $`x_{k+1} = \arg\min\limits_{x\in D} P_k(m_k(x), s_k(x)), \quad k:=k+1`$
8. If $`\min\limits_{i=\overline{1,k}} |x_{k+1}-x_i| < \epsilon`$ (tolerance), then stop, else go to step 2.

### CubicSpline
Same as **CubicSplineGrad** except there's no step 5, i.e. $`\theta_{\text{left}}^i`$ and $`\theta_{\text{right}}^i`$ are always 1.

### GradNL
Same as **CubicSplineGrad** except that $`m_k(x)`$ is used only for derivative values.
Interpolant is not used directly in search criteria $`P(x)`$, i.e.  

```math
s_k(x) = m_k(x) - L(x)
```

where $`L(x)`$ is Lipschitz’s minorant:

```math
L(x) = \max\limits_{x\in D}\{ y_{i-1} - L_{\text{left}}(x-x_{i-1}), \ y_i - L_{\text{right}}(x_i - x)\}, \quad i=\overline{1,k}
```

### QradNL
Same as **GradNL** except that minorants are quadratic:

```math
L(x) = \max\limits_{x\in D}\left\{ y_{i-1} - L_{\text{left}}\frac{(x-x_{i-1})^2}{x_i-x_{i-1}}, \ y_i - L_{\text{right}}\frac{(x_i - x)^2}{x_i-x_{i-1}} \right\}, \quad i=\overline{1,k}
```

## $`\theta`$ function

$`\theta(x,\alpha,\beta)`$ is a monotonic function that maps derivative values from range $`(-\infty, +\infty)`$ to $`[0,\alpha]`$.  
It adheres to properties below:

1. $`\theta(0) = 1`$
2. $`\lim\limits_{x\to -\infty}\theta(x) = \alpha`$
3. $`\lim\limits_{x\to +\infty}\theta(x) = 0`$
4. $`\theta(x)`$ is monotonically increasing

To be more precise it is a family of functions of two parameters $`\alpha,\beta`$.  
$`\alpha`$ defines the ceiling, while $`\beta`$ defines the rate of change.  
A convenient but not simple definition, used in the algorithms, is:

```math
\theta(x,\alpha,\beta) =
\begin{cases} 
    \dfrac{2}{\pi}\arctan\left(\dfrac{(\alpha-1)x}{\beta}\right) + 1, & x \geq 0 \\[1.2em]
    \dfrac{2(\alpha-1)}{\pi}\arctan\left(\dfrac{x}{\beta}\right) + 1, & x \leq 0
\end{cases}
```

This definition gives a smooth $`\theta`$ function satisfying properties above.

## Results (evaluation count comparison)

| Function | NL   | CubicSpline | CubicSplineGrad | GradNL | QradNL |
|----------|------|-------------|-----------------|--------|--------|
| 1        | 23   | 24          | 13              | 13     | 13     |
| 2        | 25   | 28          | 28              | 23     | 18     |
| 3        | 89   | 86          | 46              | 60     | 59     |
| 4        | 29   | 32          | 32              | 30     | 24     |
| 5        | 33   | 37          | 33              | 26     | 29     |
| 6        | 42   | 43          | 45              | 41     | 39     |
| 7        | 26   | 27          | 28              | 23     | 26     |
| 8        | 75   | 86          | 52              | 54     | 34     |
| 9        | 24   | 31          | 30              | 26     | 25     |
| 10       | 27   | 31          | 29              | 28     | 25     |
| 11       | 45   | 50          | 50              | 41     | 40     |
| 12       | 36   | 46          | 48              | 34     | 34     |
| 13       | 32   | 36          | 37              | 49     | 28     |
| 14       | 32   | 33          | 34              | 28     | 35     |
| 15       | 46   | 48          | 36              | 43     | 26     |
| 16       | 52   | 40          | 29              | 54     | 26     |
| 17       | 62   | 63          | 31              | 74     | 28     |
| 18       | 27   | 30          | 29              | 26     | 24     |
| 19       | 8    | 11          | 11              | 6      | 8      |
| 20       | 27   | 30          | 30              | 24     | 26     |
| **Average** | 38.0 | 40.6        | 33.55           | 35.15  | 28.35  |

## Graphs

<img src="pictures/CubicSpline.png" width="350">
<img src="pictures/CubicSplineGrad.png" width="350">
<img src="pictures/GradNL.png" width="350">
<img src="pictures/NL.png" width="350">
<img src="pictures/QradNL.png" width="350">

### Experimental functions
<img src="pictures/Spec1.png" width="350">
<img src="pictures/Spec2.png" width="350">
<img src="pictures/Spec3.png" width="350">
<img src="pictures/Spec4.png" width="350">
<img src="pictures/Spec5.png" width="350">

[^1]: Сергеев, Ярослав Дмитриевич. *Диагональные методы глобальной оптимизации* / Я. Д. Сергеев, Д. Е. Квасов.  
[^2]: https://www.mathnet.ru/links/0b9a247877b1fbf0e0b8f41cc75e2ebd/zvmmf6654.pdf
