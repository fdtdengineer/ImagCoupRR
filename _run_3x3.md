
## CME 3x3

### 注入同期レーザーの運動方程式

$$
\begin{align*}
  \frac{\mathrm{d}a_m(t)}{\mathrm{d}t} &= 
  \left( g - g_s {|a_m(t)|}^2 \right) a_m(t) - \sum_{n\neq m} \kappa_{mn} (a_m(t) + a_n(t)) \nonumber\\
  &= \left( g - \sum_{n\neq m} \kappa_{mn} - g_s {|a_m(t)|}^2 \right) a_m(t) - \sum_{n\neq m} \kappa_{mn} a_n(t) \nonumber\\
\end{align*}  
$$


以下、サイト数 $N=3$ の場合を考える。

$\kappa_{mn}=1+\varepsilon \quad \mathrm{for} \quad (m,n)\in\{ (1,3),(3,1)\}$ および $\kappa_{mn}=1 \ (\mathrm{other})$  とする。
$\varepsilon=0.01$ は縮退を解消するためのパラメータ。（$\varepsilon=0$ とすると2つの状態が縮退して最終状態が不定となるため）



$$
\begin{align*}
  \frac{\mathrm{d}\bm{a}(t)}{\mathrm{d}t}
  &= (gI_3-iH)\bm{a}(t) + 
  g_s \bm{\widetilde{a}}(t)
\end{align*}
$$


$$
\begin{align*}
H = 
\begin{bmatrix}
-(2+\varepsilon) & -1 & -(1+\varepsilon) \\
-1 & -2 & -1 \\
-(1+\varepsilon) & -1 & -(2+\varepsilon) \\
\end{bmatrix}, \quad
\bm{\widetilde{a}}(t) =
  \begin{bmatrix}
    \left| a_1(t) \right|^2 a_1(t) \\
    \left| a_2(t) \right|^2 a_2(t) \\
    \left| a_3(t) \right|^2 a_3(t) 
  \end{bmatrix}
\end{align*}
$$


$a_k(t)=r_k(t)e^{i\phi_k(t)}$

