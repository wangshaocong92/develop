# CuTe Layout Algebra {#cute-layout-algebra .title .is-size-3 style="font-family: 'PT Sans Narrow', sans-serif"}
## [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Introduction "Introduction"){.headerlink}Introduction {#Introduction style="scroll-margin: 1em;"}

[CuTe layout algebra](https://github.com/NVIDIA/cutlass/blob/v3.5.1/media/docs/cute/02_layout_algebra.md){target="_blank" rel="noopener"} is extremely important for understanding and applying [CUTLASS](https://github.com/NVIDIA/cutlass/){target="_blank" rel="noopener"} for accelerated computing. Despite the fact that CuTe has a [documentation](https://github.com/NVIDIA/cutlass/blob/v3.5.1/media/docs/cute/02_layout_algebra.md){target="_blank" rel="noopener"} for its layout algebra, it cannot be understood completely without first understanding its mathematical foundations. I tried to create some proofs for the CuTe layout algebra on my own and realized that it was a huge amount of work. Gratefully, [Jay Shah](https://research.colfax-intl.com/author/jay-shah/){target="_blank" rel="noopener"} has created a paper ["A Note on the Algebra of CuTe Layouts"](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf) that completes the CuTe layout algebra mathematical foundations that I wanted to create.

As my proofreading, I found Jay Shah's paper mostly error-free, except for a few very minor oversights and typos. However, it does skip some details without which the paper is a little bit hard to understand. In this article, based on Jay Shah's paper, I would like to provide more proofs and explanations of the CuTe layout algebra, some of which are not present in Jay Shah's paper. Most of the definitions and annotations will follow Jay Shah's paper.

This article can be read as a complement to Jay Shah's paper, but it's also completely standalone for understanding the CuTe layout algebra.

## [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Layout-Algebra-Preliminaries "Layout Algebra Preliminaries"){.headerlink}Layout Algebra Preliminaries {#Layout-Algebra-Preliminaries style="scroll-margin: 1em;"}

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-1-Layout "Definition 2.1: Layout"){.headerlink}Definition 2.1: Layout {#Definition-2-1-Layout style="scroll-margin: 1em;"}

A *layout* $L$ is a pair of positive integer tuples $\mathbf{S}$ and $\mathbf{D}$ of matching dimensions. We call $\mathbf{S}$ the *shape* and $\mathbf{D}$ the *stride*. We write $L = \mathbf{S}:\mathbf{D}$.

A flattened layout means that there is no internal parentheses in the shape and stride. For example, $L = (5,2,2):(16,80,4)$ is a flattened layout, whereas $L = (5,(2,2)):(16,(80,4))$ is not. Flattening a layout will not change the semantics and operations of the layout.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-2-Layout-Size-Length-and-Mode "Definition 2.2: Layout Size, Length, and Mode"){.headerlink}Definition 2.2: Layout Size, Length, and Mode {#Definition-2-2-Layout-Size-Length-and-Mode style="scroll-margin: 1em;"}

Let $\alpha \geq 0$ be an integer and $L = \mathbf{S}:\mathbf{D} = (M_{0},M_{1},\ldots,M_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$ be a layout. Then:

-   The *size* of $L$ is the product $M = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha}$.
-   The *length* of $L$ is the integer $\alpha + 1$.
-   A *mode* of $L$ is one of the entries $(M_{k}):(d_{k})$ for $0 \leq k \leq \alpha$. We may regard this as a length 1 layout.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Concatenation "Concatenation"){.headerlink}Concatenation {#Concatenation style="scroll-margin: 1em;"}

Given two layouts $L = \mathbf{S}:\mathbf{D}$ and $L^{\prime} = \mathbf{S}^{\prime}:\mathbf{D}^{\prime}$, let $\mathbf{S}^{\prime\prime}$ and $\mathbf{D}^{\prime\prime}$ be the shape and stride tuples given by (the flattening of) $(\mathbf{S},\mathbf{S}^{\prime})$ and $(\mathbf{D},\mathbf{D}^{\prime})$ respectively. Then the *concatenation* of $L$ and $L^{\prime}$ is given by the layout

$$(L,L^{\prime}) = \mathbf{S}^{\prime\prime}:\mathbf{D}^{\prime\prime}$$

and we say that $(L,L^{\prime})$ is decomposed by $L$ and $L^{\prime}$.

Inductively, given layouts $L_{0},L_{1},\ldots,L_{N}$, we can then form the concatenation $(L_{0},L_{1},\ldots,L_{N})$. Conversely, given $L$ a layout, $L$ is maximally decomposed by its modes.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Isomorphism "Isomorphism"){.headerlink}Isomorphism {#Isomorphism style="scroll-margin: 1em;"}

Let $\mathbf{S} = (M_{0},M_{1},\ldots,M_{\alpha})$ and $\mathbf{D} = (d_{0},d_{1},\ldots,d_{\alpha})$ be the respective shape and stride tuples of $L = \mathbf{S}:\mathbf{D}$. Let $M = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha}$ be the size of $L$ and let $\lbrack 0,M) \subset \mathbb{N}$ be the subset of the natural numbers given by $0,1,2,\ldots,M - 1$. Then we have an [isomorphism](https://en.wikipedia.org/wiki/Isomorphism){target="_blank" rel="noopener"}

$$\begin{array}{r}
{\iota:\lbrack 0,M) \cong \lbrack 0,M_{0}) \times \lbrack 0,M_{1}) \times \ldots \times \lbrack 0,M_{\alpha})}
\end{array}$$

Given any $x \in \lbrack 0,M)$, the isomorphism $\iota$ maps $x$ to the tuple

$$\begin{array}{r}
{x\mapsto\left( x\quad{mod}\,\, M_{0},\left\lfloor \frac{x}{M_{0}} \right\rfloor\quad{mod}\,\, M_{1},\ldots,\left\lfloor \frac{x}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor\quad{mod}\,\, M_{\alpha} \right)}
\end{array}$$

The isomorphism mapping is bijective. In our case, given any tuple $(x_{0},x_{1},\ldots,x_{\alpha}) \in \lbrack 0,M_{0}) \times \lbrack 0,M_{1}) \times \ldots \times \lbrack 0,M_{\alpha})$, the isomorphism inverse maps the tuple to the integer

$$\begin{array}{r}
{\left( x_{0},x_{1},\ldots,x_{\alpha} \right)\mapsto x_{0} + x_{1} \cdot M_{0} + x_{2} \cdot M_{0} \cdot M_{1} + \ldots + x_{\alpha} \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}}
\end{array}$$

It's straightforward to verify the above isomorphism mapping is valid and proof the above isomorphism mapping is bijective (by contradiction).

One could imagine the isomorphism as a mapping between a one-dimensional coordinate and a multi-dimensional coordinate.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-3-Layout-Function "Definition 2.3: Layout Function"){.headerlink}Definition 2.3: Layout Function {#Definition-2-3-Layout-Function style="scroll-margin: 1em;"}

Given a layout $L$, its *layout function* is the function $f_{L}:\lbrack 0,M) \rightarrow \mathbb{N}$ is defined to be the composite

$$\begin{array}{r}
{\lbrack 0,M) \cong \lbrack 0,M_{0}) \times \lbrack 0,M_{1}) \times \ldots \times \lbrack 0,M_{\alpha}) \subset \mathbb{N}^{\times (\alpha + 1)}\overset{\cdot d_{0}, \cdot d_{1},\ldots, \cdot d_{\alpha}}{\rightarrow}\mathbb{N}^{\times (\alpha + 1)}\overset{+}{\rightarrow}\mathbb{N}}
\end{array}$$

In other words, $f_{L}$ is the composition of the multilinear function

$$\begin{array}{r}
{\lbrack 0,M_{0}) \times \lbrack 0,M_{1}) \times \ldots \times \lbrack 0,M_{\alpha}) \rightarrow \mathbb{N}} \\
{(x_{0},x_{1},\ldots,x_{\alpha})\mapsto x_{0} \cdot d_{0} + x_{1} \cdot d_{1} + \ldots + x_{\alpha} \cdot d_{\alpha}}
\end{array}$$

determined by the stride, with the isomorphism $\iota$, determined by the shape.

Computing the value of a layout function $f_{L}$ at a point $x \in \lbrack 0,M)$ can be decomposed into computing the sum of the values of the layout function at multiple points. This is sometimes useful for computing the value of the layout function at a point handily.

Given a layout $L = (M_{0},M_{1},\ldots,M_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$ and $x \in \lbrack 0,M)$,

$$\begin{array}{r}
{x\mapsto\left( x_{0},x_{1},\ldots,x_{\alpha} \right)\mapsto x_{0} \cdot d_{0} + x_{1} \cdot d_{1} + \ldots + x_{\alpha} \cdot d_{\alpha}}
\end{array}$$

We also have

$$\begin{matrix}
x_{0}^{\prime} & {\mapsto\left( x_{0},0,0,\ldots,0 \right)\mapsto x_{0} \cdot d_{0}} \\
x_{1}^{\prime} & {\mapsto\left( 0,x_{1},0,\ldots,0 \right)\mapsto x_{1} \cdot d_{1}} \\
 & {\vdots} \\
x_{\alpha}^{\prime} & {\mapsto\left( 0,0,0,\ldots,x_{\alpha} \right)\mapsto x_{\alpha} \cdot d_{\alpha}}
\end{matrix}$$

Therefore, we have

$$\begin{array}{r}
{f_{L}(x) = f_{L}(x_{0}^{\prime}) + f_{L}(x_{1}^{\prime}) + \ldots + f_{L}(x_{\alpha}^{\prime})}
\end{array}$$

where

$$\begin{matrix}
x_{0}^{\prime} & {= x\quad{mod}\,\, M_{0}} \\
x_{1}^{\prime} & {= \left\lfloor \frac{x}{M_{0}} \right\rfloor\quad{mod}\,\, M_{1} \cdot M_{0}} \\
 & {\vdots} \\
x_{\alpha}^{\prime} & {= \left\lfloor \frac{x}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor\quad{mod}\,\, M_{\alpha} \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}}
\end{matrix}$$

For example, given a layout $L = (3,2):(2,3)$ and $x = 5$, we have

$$\begin{matrix}
{f_{L}(5)} & {= f_{L}(5\quad{mod}\,\, 3) + f_{L}\left( \left\lfloor \frac{5}{3} \right\rfloor\quad{mod}\,\, 2 \cdot 3 \right)} \\
 & {= f_{L}(2) + f_{L}(3)} \\
 & {= 2 \cdot 2 + \left\lfloor \frac{3}{3} \right\rfloor \cdot 3} \\
 & {= 4 + 3} \\
 & {= 7}
\end{matrix}$$

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Extension-of-Layout-Function "Extension of Layout Function"){.headerlink}Extension of Layout Function {#Extension-of-Layout-Function style="scroll-margin: 1em;"}

Based on the [definition of layout function](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-3-Layout-Function), the extension of the layout function $f_{L}$ is the function, ${\hat{f}}_{L}:\mathbb{N} \rightarrow \mathbb{N}$, defined by replacing $M_{\alpha}$ with $\infty$ in the definition of $f_{L}$, i.e., the composite

$$\begin{array}{r}
{\mathbb{N} \cong \lbrack 0,M_{0}) \times \lbrack 0,M_{1}) \times \ldots \times \lbrack 0,M_{\alpha - 1}) \times \mathbb{N} \subset \mathbb{N}^{\times (\alpha + 1)}\overset{\cdot d_{0}, \cdot d_{1},\ldots, \cdot d_{\alpha}}{\rightarrow}\mathbb{N}^{\times (\alpha + 1)}\overset{+}{\rightarrow}\mathbb{N}}
\end{array}$$

where the extension of the isomorphism $\iota$, $\hat{\iota}$, is given by

$$\begin{array}{r}
{x\mapsto\left( x\quad{mod}\,\, M_{0},\left\lfloor \frac{x}{M_{0}} \right\rfloor\quad{mod}\,\, M_{1},\ldots,\left\lfloor \frac{x}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 2}} \right\rfloor\quad{mod}\,\, M_{\alpha - 1},\left\lfloor \frac{x}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor \right)}
\end{array}$$

The extension of the isomorphism mapping is also bijective. The inverse mapping of the extension of the isomorphism is also given by

$$\begin{array}{r}
{\left( x_{0},x_{1},\ldots,x_{\alpha - 1},x_{\alpha} \right)\mapsto x_{0} + x_{1} \cdot M_{0} + x_{2} \cdot M_{0} \cdot M_{1} + \ldots + x_{\alpha} \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}}
\end{array}$$

One could imagine the extension of the isomorphism defines the last dimension of the shape to be a "batch" dimension and the batch size can be infinite.

## [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Coalescence "Coalescence"){.headerlink}Coalescence {#Coalescence style="scroll-margin: 1em;"}

Coalescence simplifies the layout and does not change the layout function.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Coalescence-Rules "Coalescence Rules"){.headerlink}Coalescence Rules {#Coalescence-Rules style="scroll-margin: 1em;"}

Considering a layout with just two integral modes, $A = (N_{0},N_{1}):(d_{0},d_{1})$, we have four cases to consider:

1.  $N_{1} = 1$.
2.  $N_{0} = 1$.
3.  $d_{1} = N_{0}d_{0}$.
4.  Anything else.

In the first case, obviously, $A = (N_{0},1):(d_{0},d_{1}) = (N_{0}):(d_{0})$. This can be further flattened to $A = N_{0}:d_{0}$.

In the second case, also obviously, $A = (1,N_{1}):(d_{0},d_{1}) = (N_{1}):(d_{1})$. This can be further flattened to $A = N_{1}:d_{1}$.

In the third case, we have $A = (N_{0},N_{1}):(d_{0},N_{0}d_{0}) = (N_{0}N_{1}):(d_{0})$. This can be further flattened to $A = N_{0}N_{1}:d_{0}$.

In the fourth case, we could do nothing and $A$ remains the same.

There is one case that can often be misunderstood, that is $d_{0} = N_{1}d_{1}$. In this case, we have $A = (N_{0},N_{1}):(N_{1}d_{1},d_{1})$. At first glance, it seems that we could coalesce $A$ to $(N_{0}N_{1}):(d_{1})$. However, this is not correct, because it changes the layout function.

Considering a layout with more than two integral modes, we could apply the above rules recursively, each time we could try to coalesce two adjacent integral modes, until no more coalescence is possible. This guarantees that the laytout function remains the same.

*Proof*

Let $L = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$ be a layout of the concatenation of layouts $L_{0}$, $L_{1}$, $\ldots$, $L_{\alpha}$, where each $L_{k} = (N_{k}):(d_{k})$ for $k \in \lbrack 0,\alpha\rbrack$. Given a coordinate $x\mapsto(x_{0},x_{1},\ldots,x_{\alpha})$, we have the layout function for layout $L$ as follows:

$$\begin{matrix}
{f_{L}(x)} & {= x_{0}d_{0} + x_{1}d_{1} + x_{2}d_{2} + \ldots + x_{\alpha}d_{\alpha}}
\end{matrix}$$

The layout function for layout $L$ is just the sum of the layout functions of each $L_{k}$, i.e.,

$$\begin{matrix}
{f_{L}(x)} & {= f_{L_{0}}(x_{0}) + f_{L_{1}}(x_{1}) + \ldots + f_{L_{\alpha}}(x_{\alpha})} \\
 & {= x_{0}d_{0} + x_{1}d_{1} + x_{2}d_{2} + \ldots + x_{\alpha}d_{\alpha}}
\end{matrix}$$

Suppose two adjacent integral modes, say $A = (L_{i},L_{i + 1}) = (N_{i},N_{i + 1}):(d_{i},d_{i + 1})$, can be coalesced to a new layout $A_{i}^{\prime}$ whose shape is $(N_{i}N_{i + 1})$, which satisfies the first, second, and third coalescence rules. By the definition of coalescence, we must have $f_{A^{\prime}}(x) = f_{A}(x)$.

For any $x_{i} \in \lbrack 0,N_{i})$ and $x_{i + 1} \in \lbrack 0,N_{i + 1})$, we have $x^{\prime} = x_{i} + x_{i + 1} \cdot N_{i}$, where $x^{\prime} \in \lbrack 0,N_{i}N_{i + 1})$.

$$\begin{matrix}
{f_{A^{\prime}}(x^{\prime})} & {= f_{A}(x^{\prime})} \\
 & {= x_{i}d_{i} + x_{i + 1}d_{i + 1}} \\
 & {= f_{L_{i}}(x_{i}) + f_{L_{i + 1}}(x_{i + 1})}
\end{matrix}$$

Given any coordinate $x\mapsto(x_{0},x_{1},\ldots,x_{i - 1},x_{i},x_{i + 1},x_{i + 2},\ldots,x_{\alpha})$ for the layout $L$, after coalescing $A$ to $A^{\prime}$, the coordiante $x\mapsto(x_{0},x_{1},\ldots,x_{i - 1},x^{\prime},x_{i + 2},\ldots,x_{\alpha})$ for the layout $L^{\prime}$, where $L^{\prime}$ is the layout after coalescing $A$ to $A^{\prime}$.

This is easy to verify because of the isomorphism of coordinates.

$$\begin{matrix}
x & {= x_{0} + x_{1} \cdot N_{0} + x_{2} \cdot N_{0}N_{1} + \ldots + x_{i - 1} \cdot N_{0}N_{1}\ldots N_{i - 2} + x_{i} \cdot N_{0}N_{1}\ldots N_{i - 1} + x_{i + 1} \cdot N_{0}N_{1}\ldots N_{i - 1}N_{i} + x_{i + 2} \cdot N_{0}N_{1}\ldots N_{i - 1}N_{i}N_{i + 1} + \ldots + x_{\alpha} \cdot N_{0}N_{1}\ldots N_{\alpha - 1}} \\
 & {= x_{0} + x_{1} \cdot N_{0} + x_{2} \cdot N_{0}N_{1} + \ldots + x_{i - 1} \cdot N_{0}N_{1}\ldots N_{i - 2} + (x_{i} + x_{i + 1} \cdot N_{i}) \cdot N_{0}N_{1}\ldots N_{i - 1} + x_{i + 2} \cdot N_{0}N_{1}\ldots N_{i - 1}N_{i}N_{i + 1} + \ldots + x_{\alpha} \cdot N_{0}N_{1}\ldots N_{\alpha - 1}} \\
 & {= x_{0} + x_{1} \cdot N_{0} + x_{2} \cdot N_{0}N_{1} + \ldots + x_{i - 1} \cdot N_{0}N_{1}\ldots N_{i - 2} + x^{\prime} \cdot N_{0}N_{1}\ldots N_{i - 1} + x_{i + 2} \cdot N_{0}N_{1}\ldots N_{i - 1}N_{i}N_{i + 1} + \ldots + x_{\alpha} \cdot N_{0}N_{1}\ldots N_{\alpha - 1}}
\end{matrix}$$

Then we have

$$\begin{matrix}
{f_{L^{\prime}}(x)} & {= f_{L_{0}}(x_{0}) + f_{L_{1}}(x_{1}) + \ldots + f_{L_{i - 1}}(x_{i - 1}) + f_{A^{\prime}}(x^{\prime}) + f_{L_{i + 2}}(x_{i + 2}) + \ldots + f_{L_{\alpha}}(x_{\alpha})} \\
 & {= f_{L_{0}}(x_{0}) + f_{L_{1}}(x_{1}) + \ldots + f_{L_{i - 1}}(x_{i - 1}) + f_{L_{i}}(x_{i}) + f_{L_{i + 1}}(x_{i + 1}) + f_{L_{i + 2}}(x_{i + 2}) + \ldots + f_{L_{\alpha}}(x_{\alpha})} \\
 & {= f_{L}(x)}
\end{matrix}$$

Therefore, the layout function remains the same after coalescing $A$ to $A^{\prime}$.

This concludes the proof.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#By-Mode-Coalescence "By-Mode Coalescence"){.headerlink}By-Mode Coalescence {#By-Mode-Coalescence style="scroll-margin: 1em;"}

In some cases, when the modes are not completely integral in the layout and we would like to keep the number of modes unchanged, we could perform *by-mode coalescence*. This can be achieved by disabling the coalescence of adjacent integral modes from two different modes in the coalescence rules for any layout with more than two integral modes.

For example, if we have a layout of two modes $A = ((N_{0},N_{1},\ldots,N_{\alpha}),(N_{\alpha + 1},N_{\alpha + 2},\ldots,N_{\beta}):((d_{0},d_{1},\ldots,d_{\alpha}),(d_{\alpha + 1},d_{\alpha + 2},\ldots,d_{\beta})))$, to perform by-mode coalescence, we could coalesce the integral modes $(N_{0},N_{1},\ldots,N_{\alpha})$ and $(N_{\alpha + 1},N_{\alpha + 2},\ldots,N_{\beta})$ separately until no more coalescence is possible for each one, and no more coalescence will be performed further between the two consequent coalesced modes even if they can be coalesced. This will result in a layout with the same number of modes as before.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Implication-of-Coalescence "Implication of Coalescence"){.headerlink}Implication of Coalescence {#Implication-of-Coalescence style="scroll-margin: 1em;"}

Coalescence simplifies the layout and does not change the layout function. It is a useful operation to reduce the complexity of the layout and the related computation while preserving its functionality. If non-by-mode coalescence is performed on a layout, the layout can be simplified such that each mode is an integral mode, i.e., $A = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$ where $N_{k}$ is an integer, not a tuple of integers, and $N_{k} > 1$ for $k \in \lbrack 0,\alpha\rbrack$.

The property of coalescence is very important and most of the proofs we will present in the article assumes that the layout is coalesced. This assumption is fine mathematically because the coalescence does not change the layout function.

## [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Complementation "Complementation"){.headerlink}Complementation {#Complementation style="scroll-margin: 1em;"}

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-4-Sorted-Layout "Definition 2.4: Sorted Layout"){.headerlink}Definition 2.4: Sorted Layout {#Definition-2-4-Sorted-Layout style="scroll-margin: 1em;"}

Let $A = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$ be a layout. We say that $A$ is *sorted* if $d_{0} \leq d_{1} \leq \ldots \leq d_{\alpha}$ and for every $i < j$, if $d_{i} = d_{j}$, then $N_{i} \leq N_{j}$.

Note that sorting a layout, or more generally, changing the order of modes of a layout, will change the semantics and operations of the layout.

For example, suppose we have a layout $A = (2,4):(4,1)$ and a layout $B = (4,2):(1,4)$. We could see that $B$ is the sorted version of $A$. We could compute the layout function of $A$ and $B$ as follows using lookup tables:

$$\begin{matrix}
{f_{A}(0)} & {= f_{A}(0,0) = 0 \cdot 4 + 0 \cdot 1 = 0} \\
{f_{A}(1)} & {= f_{A}(1,0) = 1 \cdot 4 + 0 \cdot 1 = 4} \\
{f_{A}(2)} & {= f_{A}(0,1) = 0 \cdot 4 + 1 \cdot 1 = 1} \\
{f_{A}(3)} & {= f_{A}(1,1) = 1 \cdot 4 + 1 \cdot 1 = 5} \\
{f_{A}(4)} & {= f_{A}(0,2) = 0 \cdot 4 + 2 \cdot 1 = 2} \\
{f_{A}(5)} & {= f_{A}(1,2) = 1 \cdot 4 + 2 \cdot 1 = 6} \\
{f_{A}(6)} & {= f_{A}(0,3) = 0 \cdot 4 + 3 \cdot 1 = 3} \\
{f_{A}(7)} & {= f_{A}(1,3) = 1 \cdot 4 + 3 \cdot 1 = 7}
\end{matrix}$$

$$\begin{matrix}
{f_{B}(0)} & {= f_{B}(0,0) = 0 \cdot 1 + 0 \cdot 4 = 0} \\
{f_{B}(1)} & {= f_{B}(1,0) = 1 \cdot 1 + 0 \cdot 4 = 1} \\
{f_{B}(2)} & {= f_{B}(2,0) = 2 \cdot 1 + 0 \cdot 4 = 2} \\
{f_{B}(3)} & {= f_{B}(3,0) = 3 \cdot 1 + 0 \cdot 4 = 3} \\
{f_{B}(4)} & {= f_{B}(0,1) = 0 \cdot 1 + 1 \cdot 4 = 4} \\
{f_{B}(5)} & {= f_{B}(1,1) = 1 \cdot 1 + 1 \cdot 4 = 5} \\
{f_{B}(6)} & {= f_{B}(2,1) = 2 \cdot 1 + 1 \cdot 4 = 6} \\
{f_{B}(7)} & {= f_{B}(3,1) = 3 \cdot 1 + 1 \cdot 4 = 7}
\end{matrix}$$

We could see that the layout $B$ is typically referred as the column-major layout, and the layout $A$ is typically referred as the row-major layout. They are completely different layouts.

More generally, the sorted layout is a just like the "generalization" of the column-major layout.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-5-Admission-for-Complementation "Definition 2.5: Admission for Complementation"){.headerlink}Definition 2.5: Admission for Complementation {#Definition-2-5-Admission-for-Complementation style="scroll-margin: 1em;"}

Let $A = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$ be a layout and $M$ be a positive integer. If $A$ is not sorted then replace $A$ with its sorted version. We say that the pair $\{ A,M\}$ is *admissible for complementation* (or simply admissible) if:

-   For all $1 \leq i \leq \alpha$, $N_{i - 1} \cdot d_{i - 1}$ divides $d_{i}$.
-   $N_{\alpha} \cdot d_{\alpha}$ divides $M$.

That $\{ A,M\}$ is admissible for complementation also implies:

-   For all $1 \leq i \leq \alpha$, $N_{i - 1} \cdot d_{i - 1} \leq d_{i}$ and $d_{i - 1} \leq d_{i}$.
-   $N_{\alpha} \cdot d_{\alpha} \leq M$ and $d_{\alpha} \leq M$.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-6-Complementation "Definition 2.6: Complementation"){.headerlink}Definition 2.6: Complementation {#Definition-2-6-Complementation style="scroll-margin: 1em;"}

Let $A = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$ be a layout and $M$ be a positive integer. If $\{ A,M\}$ is admissible for complementation, then if $A$ is not sorted, replace $A$ with its sorted version. The complement of $\{ A,M\}$ is defined to be the layout

$$\begin{array}{r}
{\text{complement}(A,M) = \left( d_{0},\frac{d_{1}}{N_{0}d_{0}},\frac{d_{2}}{N_{1}d_{1}},\ldots,\frac{d_{\alpha}}{N_{\alpha - 1}d_{\alpha - 1}},\frac{M}{N_{\alpha}d_{\alpha}} \right):\left( 1,N_{0}d_{0},N_{1}d_{1},\ldots,N_{\alpha}d_{\alpha} \right)}
\end{array}$$

Note that the size of the complement of $\{ A,M\}$, $\text{size}(\text{complement}(A,M))$, is $\frac{M}{\text{size}(A)} = \frac{M}{N_{0} \cdot N_{1} \cdot \ldots \cdot N_{\alpha}}$.

By definition, the complement of $\{ A,M\}$ is insensitive to the order of the modes of $A$, since it will always be sorted before complementation.

The complement of $\{ A,M\}$ is strictly increasing. This might not be very obvious, so we will show a proof.

*Proof*

Suppose $B = \text{complement}(A,M)$, to show that the layout function $f_{B}$, whose domain is a set of natural numbers, is strictly increasing, we need to show that for every two adjacent natural numbers $x$ and $x + 1$, $0 \leq x < x + 1 < \text{size}(B)$, we have $f_{B}(x) < f_{B}(x + 1)$.

Because of the isomorphism, suppose the mapping of $x$ is as follows:

$$\begin{matrix}
x & {\mapsto\left( x_{0},x_{1},\ldots,x_{\alpha},x_{\alpha + 1} \right)}
\end{matrix}$$

By definition of the layout function $f_{B}$, we have

$$\begin{matrix}
{f_{B}(x)} & {= x_{0} + x_{1} \cdot N_{0}d_{0} + x_{2} \cdot N_{1}d_{1} + \ldots + x_{\alpha} \cdot N_{\alpha - 1}d_{\alpha - 1} + x_{\alpha + 1} \cdot N_{\alpha}d_{\alpha}}
\end{matrix}$$

The mapping of $x + 1$ can have many different cases.

In the simplest case,

$$\begin{matrix}
{x + 1} & {\mapsto\left( x_{0} + 1,x_{1},\ldots,x_{\alpha},x_{\alpha + 1} \right)}
\end{matrix}$$

Then we have

$$\begin{matrix}
{f_{B}(x + 1)} & {= x_{0} + 1 + x_{1} \cdot N_{0}d_{0} + x_{2} \cdot N_{1}d_{1} + \ldots + x_{\alpha} \cdot N_{\alpha - 1}d_{\alpha - 1} + x_{\alpha + 1} \cdot N_{\alpha}d_{\alpha}} \\
 & {= f_{B}(x) + 1} \\
 & {> f_{B}(x)}
\end{matrix}$$

In a more complicated case, where $x_{0} = d_{0} - 1$ and $x_{1} < \frac{d_{1}}{N_{0}d_{0}} - 1$, we have

$$\begin{matrix}
{x + 1} & {\mapsto\left( 0,x_{1} + 1,\ldots,x_{\alpha},x_{\alpha + 1} \right)}
\end{matrix}$$

Then we have

$$\begin{matrix}
{f_{B}(x + 1)} & {= 0 + (x_{1} + 1) \cdot N_{0}d_{0} + x_{2} \cdot N_{1}d_{1} + \ldots + x_{\alpha} \cdot N_{\alpha - 1}d_{\alpha - 1} + x_{\alpha + 1} \cdot N_{\alpha}d_{\alpha}} \\
 & {= f_{B}(x) - x_{0} + N_{0}d_{0}} \\
 & {= f_{B}(x) - (d_{0} - 1) + N_{0}d_{0}} \\
 & {= f_{B}(x) + 1 + (N_{0} - 1)d_{0}} \\
 & {> f_{B}(x)}
\end{matrix}$$

Because $N_{0} \geq 1$, we have $(N_{0} - 1)d_{0} \geq 0$, so we have

$$\begin{matrix}
{f_{B}(x + 1)} & {> f_{B}(x)}
\end{matrix}$$

In general, when $x_{0} = d_{0} - 1$, for some $k \in \lbrack 1,\alpha - 1\rbrack$, $x_{i} = \frac{d_{i}}{N_{i - 1}d_{i - 1}} - 1$ for every $i \in \lbrack 1,k\rbrack$, $x_{k + 1} < \frac{d_{k + 1}}{N_{k}d_{k}} - 1$, we have

$$\begin{matrix}
{x + 1} & {\mapsto\left( 0,0,\ldots,0,x_{k + 1} + 1,\ldots,x_{\alpha},x_{\alpha + 1} \right)}
\end{matrix}$$

Then we have

$$\begin{matrix}
{f_{B}(x + 1)} & {= 0 + 0 \cdot N_{0}d_{0} + \ldots + 0 \cdot N_{k - 1}d_{k - 1} + (x_{k + 1} + 1) \cdot N_{k}d_{k} + \ldots + x_{\alpha} \cdot N_{\alpha - 1}d_{\alpha - 1} + x_{\alpha + 1} \cdot N_{\alpha}d_{\alpha}} \\
 & {= f_{B}(x) - x_{0} - \left( \sum\limits_{i = 1}^{k}x_{i} \cdot N_{i - 1}d_{i - 1} \right) + N_{k}d_{k}} \\
 & {= f_{B}(x) - (d_{0} - 1) - \left( \sum\limits_{i = 1}^{k}\left( \frac{d_{i}}{N_{i - 1}d_{i - 1}} - 1 \right) \cdot N_{i - 1}d_{i - 1} \right) + N_{k}d_{k}} \\
 & {= f_{B}(x) - (d_{0} - 1) - \left( \sum\limits_{i = 1}^{k}\left( d_{i} - N_{i - 1}d_{i - 1} \right) \right) + N_{k}d_{k}} \\
 & {= f_{B}(x) - (d_{0} - 1) + \sum\limits_{i = 1}^{k}N_{i - 1}d_{i - 1} - \sum\limits_{i = 1}^{k}d_{i} + N_{k}d_{k}} \\
 & {= f_{B}(x) + \sum\limits_{i = 0}^{k}\left( N_{i} - 1 \right)d_{i} + 1}
\end{matrix}$$

Because $N_{i} \geq 1$ for every $i$, we have $\left( N_{i} - 1 \right)d_{i} \geq 0$ for every $i$, so we have

$$\begin{matrix}
{f_{B}(x + 1)} & {> f_{B}(x)}
\end{matrix}$$

This concludes the proof.

Similarly, we could also prove that the extension of the complement of $\{ A,M\}$ is strictly increasing.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Proposition-2-7 "Proposition 2.7"){.headerlink}Proposition 2.7 {#Proposition-2-7 style="scroll-margin: 1em;"}

Let $\{ A = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha}),M\}$ be admissible for complementation and $B = \text{complement}(A,M)$. Let $C = (A,B)$ be the concatenated layout. Then the size of $C$ is $M$ and $f_{C}:\lbrack 0,M) \rightarrow \mathbb{N}$ restricts to a bijection $\lbrack 0,M) \cong \lbrack 0,M)$.

*Proof*

Because $\text{size}(A) = \prod\limits_{i = 0}^{\alpha}N_{i}$ and $\text{size}(B) = \frac{M}{\prod\limits_{i = 0}^{\alpha}N_{i}}$, we have $\text{size}(C) = \text{size}(A) \cdot \text{size}(B) = M$. Thus the domain of $f_{C}$ is $\lbrack 0,M)$.

Note that the image of $f_{C}$ is the same as that of $f_{C^{\prime}}$ for any permutation $C^{\prime}$ of $C$.

To see this, suppose we have the following layout $C$ and its permutation $C^{\prime}$ in which only one pair of the modes is permuted.

$$\begin{matrix}
C & {= \left( N_{0},N_{1},\ldots,N_{i},\ldots,N_{j},\ldots,N_{\alpha} \right):\left( d_{0},d_{1},\ldots,d_{i},\ldots,d_{j},\ldots,d_{\alpha} \right)} \\
C^{\prime} & {= \left( N_{0},N_{1},\ldots,N_{j},\ldots,N_{i},\ldots,N_{\alpha} \right):\left( d_{0},d_{1},\ldots,d_{j},\ldots,d_{i},\ldots,d_{\alpha} \right)}
\end{matrix}$$

The domains of $f_{C}$ and $f_{C}^{\prime}$ are both $\lbrack 0,M)$. For any $x_{C} \in \lbrack 0,M)$, we have

$$\begin{matrix}
x_{C} & {\mapsto\left( x_{0},x_{1},\ldots,x_{i},\ldots,x_{j},\ldots,x_{\alpha} \right)} \\
x_{C^{\prime}} & {\mapsto\left( x_{0},x_{1},\ldots,x_{j},\ldots,x_{i},\ldots,x_{\alpha} \right)}
\end{matrix}$$

and $x_{C}$ and $x_{C^{\prime}}$ are bijective.

Because by definition, $f_{C}(x_{C}) = f_{C^{\prime}}(x_{C^{\prime}})$, the image of $f_{C}$ is the same as that of $f_{C^{\prime}}$.

For any permutation $C^{\prime}$ of $C$, it can be obtained by permuting one pair of the modes of $C$ at a time and each time the image of $f_{C}$ is the same as that of $f_{C^{\prime}}$. Therefore, the image of $f_{C}$ is the same as that of $f_{C^{\prime}}$ for any permutation $C^{\prime}$ of $C$.

When computing the image of $f_{C}$ we may sort $C$. Without loss of generality, suppose $A = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$ is already sorted. After sorting $C$, the sorted $C^{\prime}$ could only be as follows:

$$\begin{matrix}
C^{\prime} & {= \left( d_{0},N_{0},\frac{d_{1}}{N_{0}d_{0}},N_{1},\frac{d_{2}}{N_{1}d_{1}},N_{2},\ldots,\frac{d_{\alpha}}{N_{\alpha - 1}d_{\alpha - 1}},N_{\alpha},\frac{M}{N_{\alpha}d_{\alpha}} \right):\left( 1,d_{0},N_{0}d_{0},d_{1},N_{1}d_{1},d_{2},\ldots,N_{\alpha - 1}d_{\alpha - 1},d_{\alpha},N_{\alpha}d_{\alpha} \right)}
\end{matrix}$$

Because $d_{i} \leq N_{i}d_{i}$ and $N_{i}d_{i} \leq d_{i + 1}$ for every $i$, when $N_{i} = 1$, $N_{i} \leq \frac{d_{i + 1}}{N_{i}d_{i}}$, when $N_{i}d_{i} = d_{i + 1}$, $\frac{d_{i + 1}}{N_{i}d_{i}} \leq N_{i + 1}$, thus $C^{\prime}$ is sorted and any permutation of $C^{\prime}$ will make it not sorted.

Then we may rewrite

$$\begin{matrix}
C^{\prime} & {= \left( r_{0},r_{1},r_{2},\ldots,r_{\beta} \right):\left( 1,r_{0},r_{0}r_{1},\ldots,r_{0}r_{1}\ldots r_{\beta - 1} \right)}
\end{matrix}$$

where $\beta = 2\alpha + 1$ and the maximum value that $f_{C^{\prime}}$ attains is computed as follows:

$$\begin{matrix}
{f_{C^{\prime}}(M - 1)} & {= f_{C^{\prime}}(r_{0} - 1,r_{1} - 1,r_{2} - 1,\ldots,r_{\beta - 1} - 1,r_{\beta} - 1)} \\
 & {= (r_{0} - 1) + (r_{1} - 1) \cdot r_{0} + (r_{2} - 1) \cdot r_{0}r_{1} + \ldots + (r_{\beta - 1} - 1) \cdot r_{0}r_{1}\ldots r_{\beta - 2} + (r_{\beta} - 1) \cdot r_{0}r_{1}\ldots r_{\beta - 1}} \\
 & {= r_{0} - 1 + r_{0}r_{1} - r_{0} + r_{0}r_{1}r_{2} - r_{0}r_{1} + \ldots + r_{0}r_{1}\ldots r_{\beta - 1} - r_{0}r_{1}\ldots r_{\beta - 2} + r_{0}r_{1}\ldots r_{\beta} - r_{0}r_{1}\ldots r_{\beta - 1}} \\
 & {= r_{0}r_{1}\ldots r_{\beta} - 1} \\
 & {= M - 1}
\end{matrix}$$

Then in this case, to establish the bijectivity assertion, it's sufficient to just show $f_{C^{\prime}}(x)$ is injective, i.e., for any $x,y \in \lbrack 0,M)$, if $f_{C^{\prime}}(x) = f_{C^{\prime}}(y)$, then $x = y$.

Suppose the isomorphism mapping of $x$ and $y$ are as follows:

$$\begin{matrix}
x & {\mapsto\left( x_{0},x_{1},\ldots,x_{\beta} \right)} \\
y & {\mapsto\left( y_{0},y_{1},\ldots,y_{\beta} \right)}
\end{matrix}$$

Because $f_{C^{\prime}}(x) = f_{C^{\prime}}(y)$, we have

$$\begin{array}{r}
{x_{0} + x_{1} \cdot r_{0} + x_{2} \cdot r_{0}r_{1} + \ldots + x_{\beta} \cdot r_{0}r_{1}\ldots r_{\beta - 1} = y_{0} + y_{1} \cdot r_{0} + y_{2} \cdot r_{0}r_{1} + \ldots + y_{\beta} \cdot r_{0}r_{1}\ldots r_{\beta - 1}}
\end{array}$$

We will use strong induction to show that $x_{i} = y_{i}$ for every $i \in \lbrack 0,\beta\rbrack$.

Because $f_{C^{\prime}}(x)\mspace{12mu}{mod}\,\, r_{0} = f_{C^{\prime}}(y)\mspace{12mu}{mod}\,\, r_{0}$, we have $x_{0} = y_{0}$.

Now suppose by the strong induction that given $i \in (0,\beta\rbrack$, for all $j < i$, we have $x_{j} = y_{j}$. we have

$$\begin{array}{r}
{x_{i} \cdot r_{0}r_{1}\ldots r_{i - 1} + x_{i + 1} \cdot r_{0}r_{1}\ldots r_{i} + \ldots + x_{\beta} \cdot r_{0}r_{1}\ldots r_{\beta - 1} = y_{i} \cdot r_{0}r_{1}\ldots r_{i - 1} + y_{i + 1} \cdot r_{0}r_{1}\ldots r_{i} + \ldots + y_{\beta} \cdot r_{0}r_{1}\ldots r_{\beta - 1}}
\end{array}$$

Because $x_{i} \in \lbrack 0,r_{i})$ and $y_{i} \in \lbrack 0,r_{i})$, taking this equation modulo $r_{0}r_{1}\ldots r_{i}$ and dividing by $r_{0}r_{1}\ldots r_{i - 1}$, we have $x_{i} = y_{i}$.

Because $(x_{0},x_{1},\ldots,x_{\beta}) = (y_{0},y_{1},\ldots,y_{\beta})$, and the isomorphism mapping is bijective, we have $x = y$.

Therefore $f_{C^{\prime}}:\lbrack 0,M) \rightarrow \mathbb{N}$ restricts to a bijection $\lbrack 0,M) \cong \lbrack 0,M)$. So does $f_{C}$.

This concludes the proof.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Corollary-2-8-Complementation-Disjointness "Corollary 2.8 Complementation Disjointness"){.headerlink}Corollary 2.8 Complementation Disjointness {#Corollary-2-8-Complementation-Disjointness style="scroll-margin: 1em;"}

The Corollary 2.8 explains what it means of taking a complement of a layout.

In the setting of [Proposition 2.7](https://leimao.github.io/article/CuTe-Layout-Algebra/#Proposition-2-7), let $I = \lbrack 0,\text{size}(A)) = \lbrack 0,N_{0}N_{1}\ldots N_{\alpha})$ be the domain of $f_{A}$. Then

$$\begin{array}{r}
{f_{A}(I) \cap {\hat{f}}_{B}(I) = \{ 0\}}
\end{array}$$

In other words, ${\hat{f}}_{A}$ and ${\hat{f}}_{B}$ have disjoint image when restricted to the domain of $f_{A}$, apart from 0.

Note that in the corollary, $f_{A}$ and ${\hat{f}}_{A}$ are actually interchangeable, because the function domain is restricted to the domain of $f_{A}$.

*Proof*

Let $J = \lbrack 0,\text{size}(B)) = \lbrack 0,\frac{M}{N_{0}N_{1}\ldots N_{\alpha}})$ be the domain of $f_{B}$. Then by [Proposition 2.7](https://leimao.github.io/article/CuTe-Layout-Algebra/#Proposition-2-7), we have

$$\begin{array}{r}
{f_{A}(I) \cap f_{B}(J) = \{ 0\}}
\end{array}$$

To understand this, for any $x_{A} \in I$ and any $x_{B} \in J$, because of the isomorphism, we have

$$\begin{matrix}
x_{A} & {\mapsto\left( x_{A,0},x_{A,1},\ldots,x_{A,\alpha} \right)} \\
x_{B} & {\mapsto\left( x_{B,0},x_{B,1},\ldots,x_{B,\alpha},x_{B,\alpha + 1} \right)}
\end{matrix}$$

Then we have

$$\begin{matrix}
{f_{A}(x_{A})} & {= x_{A,0} + x_{A,1} \cdot N_{0} + x_{A,2} \cdot N_{0}N_{1} + \ldots + x_{A,\alpha} \cdot N_{0}N_{1}\ldots N_{\alpha - 1}} \\
{f_{B}(x_{B})} & {= x_{B,0} + x_{B,1} \cdot N_{0}d_{0} + x_{B,2} \cdot N_{1}d_{1} + \ldots + x_{B,\alpha} \cdot N_{\alpha - 1}d_{\alpha - 1} + x_{B,\alpha + 1} \cdot N_{\alpha}d_{\alpha}}
\end{matrix}$$

We orchestrate new coordinates for layout $C$ as follows:

$$\begin{matrix}
x_{A}^{\prime} & {\mapsto\left( 0,x_{A,0},0,x_{A,1},0,x_{A,2},\ldots,0,x_{A,\alpha},0 \right)} \\
x_{B}^{\prime} & {\mapsto\left( x_{B,0},0,x_{B,1},0,x_{B,2},\ldots,x_{B,\alpha},0,x_{B,\alpha + 1} \right)}
\end{matrix}$$

Then we have

$$\begin{matrix}
{f_{C}(x_{A}^{\prime})} & {= x_{A,0} + x_{A,1} \cdot N_{0} + x_{A,2} \cdot N_{0}N_{1} + \ldots + x_{A,\alpha} \cdot N_{0}N_{1}\ldots N_{\alpha - 1}} \\
 & {= f_{A}(x_{A})} \\
{f_{C}(x_{B}^{\prime})} & {= x_{B,0} + x_{B,1} \cdot N_{0}d_{0} + x_{B,2} \cdot N_{1}d_{1} + \ldots + x_{B,\alpha} \cdot N_{\alpha - 1}d_{\alpha - 1} + x_{B,\alpha + 1} \cdot N_{\alpha}d_{\alpha}} \\
 & {= f_{B}(x_{B})}
\end{matrix}$$

By the Proposition 2.7, we have $f_{C}:\lbrack 0,M) \rightarrow \mathbb{N}$ restricts to a bijection $\lbrack 0,M) \cong \lbrack 0,M)$. If $x_{A}^{\prime} \neq x_{B}^{\prime}$, then $f_{C}(x_{A}^{\prime}) \neq f_{C}(x_{B}^{\prime})$, and $f_{A}(x_{A}) \neq f_{B}(x_{B})$.

Obviously, other than $(0,0,\ldots,0)$, for any values of $x_{A,0},x_{A,1},\ldots,x_{A,\alpha}$ and $x_{B,0},x_{B,1},\ldots,x_{B,\alpha},x_{B,\alpha + 1}$, $\left( 0,x_{A,0},0,x_{A,1},0,x_{A,2},\ldots,0,x_{A,\alpha},0 \right) \neq \left( x_{B,0},0,x_{B,1},0,x_{B,2},\ldots,x_{B,\alpha},0,x_{B,\alpha + 1} \right)$, $x_{A}^{\prime} \neq x_{B}^{\prime}$, $f_{C}(x_{A}^{\prime}) \neq f_{C}(x_{B}^{\prime})$, and $f_{A}(x_{A}) \neq f_{B}(x_{B})$.

This means, for any $x \in I$ that $x \neq 0$, there is no $y \in J$ such that $f_{A}(x) = f_{B}(y)$.

When $x = 0$, we have $f_{A}(x) = f_{B}(x) = 0$. Thus we could claim that

$$\begin{array}{r}
{f_{A}(I) \cap f_{B}(J) = \{ 0\}}
\end{array}$$

In the [Definition 2.6: Complementation](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-6-Complementation), we have shown that the complement of $\{ A,M\}$, $f_{B}$, as well as its extension ${\hat{f}}_{B}$, are strictly increasing.

In addition, by the extension of the isomorphism, we have

$$\begin{array}{r}
{\text{size}(B)\mapsto\left( 0,0,\ldots,0,\frac{M}{N_{\alpha}d_{\alpha}} \right)}
\end{array}$$

Then we have

$$\begin{matrix}
{{\hat{f}}_{B}(\text{size}(B))} & {= 0 + 0 \cdot 1 + 0 \cdot N_{0}d_{0} + \ldots + 0 \cdot N_{\alpha - 1}d_{\alpha - 1} + \frac{M}{N_{\alpha}d_{\alpha}} \cdot N_{\alpha}d_{\alpha}} \\
 & {= M}
\end{matrix}$$

The largest value attained by $f_{A}$ is at $N_{0}N_{1}\ldots N_{\alpha} - 1$, and $f_{A}(N_{0}N_{1}\ldots N_{\alpha} - 1) = (N_{0} - 1)d_{0} + (N_{1} - 1)d_{1} + \ldots + (N_{\alpha} - 1)d_{\alpha}$.

Because $(N_{0} - 1)d_{0} < N_{0}d_{0}$ and $N_{i}d_{i} \leq d_{i + 1}$ for every $i \in \lbrack 0,\alpha - 1\rbrack$, $N_{\alpha}d_{\alpha} \leq M$, we have

$$\begin{matrix}
{f_{A}(N_{0}N_{1}\ldots N_{\alpha} - 1)} & {= (N_{0} - 1)d_{0} + (N_{1} - 1)d_{1} + \ldots + (N_{\alpha} - 1)d_{\alpha}} \\
 & {< N_{0}d_{0} + N_{1}d_{1} - d_{1} + N_{2}d_{2} - d_{2} + \ldots + N_{\alpha}d_{\alpha} - d_{\alpha}} \\
 & {\leq d_{1} + N_{1}d_{1} - d_{1} + N_{2}d_{2} - d_{2} + \ldots + N_{\alpha}d_{\alpha} - d_{\alpha}} \\
 & {= N_{1}d_{1} + N_{2}d_{2} - d_{2} + \ldots + N_{\alpha}d_{\alpha} - d_{\alpha}} \\
 & {\leq d_{2} + N_{2}d_{2} - d_{2} + \ldots + N_{\alpha}d_{\alpha} - d_{\alpha}} \\
 & {\vdots} \\
 & {\leq d_{\alpha} + N_{\alpha}d_{\alpha} - d_{\alpha}} \\
 & {= N_{\alpha}d_{\alpha}} \\
 & {\leq M}
\end{matrix}$$

Thus $f_{A}(N_{0}N_{1}\ldots N_{\alpha} - 1) < {\hat{f}}_{B}(\text{size}(B))$.

In the case of $I \cap J = I$, i.e., $\text{size}(A) \leq \text{size}(B)$. Then we have

$$\begin{array}{r}
{f_{A}(I) \cap f_{B}(I) = \{ 0\}}
\end{array}$$

Because in this case, $f_{B}(I) = {\hat{f}}_{B}(I)$, we have

$$\begin{array}{r}
{f_{A}(I) \cap {\hat{f}}_{B}(I) = \{ 0\}}
\end{array}$$

In the other case of $I \cap J = J$, i.e., $\text{size}(A) \geq \text{size}(B)$. Because the largest value attained by $f_{A}$ is $f_{A}(N_{0}N_{1}\ldots N_{\alpha} - 1)$, and $f_{A}(N_{0}N_{1}\ldots N_{\alpha} - 1) < {\hat{f}}_{B}(\text{size}(B))$, for any $x \in I/J$, we have $f_{A}(x) < {\hat{f}}_{B}(\text{size}(B))$.

Thus,

$$\begin{array}{r}
{f_{A}(I) \cap {\hat{f}}_{B}(I/J) = \varnothing}
\end{array}$$

Therefore,

$$\begin{matrix}
{f_{A}(I) \cap {\hat{f}}_{B}(I)} & {= f_{A}(I) \cap \left( {\hat{f}}_{B}(I) \cup {\hat{f}}_{B}(I/J) \right)} \\
 & {= f_{A}(I) \cap \left( f_{B}(I) \cup {\hat{f}}_{B}(I/J) \right)} \\
 & {= \left( f_{A}(I) \cap f_{B}(I) \right) \cup \left( f_{A}(I) \cap {\hat{f}}_{B}(I/J) \right)} \\
 & {= \{ 0\} \cup \varnothing} \\
 & {= \{ 0\}}
\end{matrix}$$

Taken together, we have

$$\begin{array}{r}
{f_{A}(I) \cap {\hat{f}}_{B}(I) = \{ 0\}}
\end{array}$$

This concludes the proof. 

A short note on the original proof of Corollary 2.8 in the [paper](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf) is that Jay Shah claimed $f_{A}(I \cap J) \cap f_{B}(I \cap J) = \{ 0\}$, which is insufficient to show the proof. The sufficient statement should be $f_{A}(I) \cap f_{B}(J) = \{ 0\}$.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Remark-2-9-Complementation-Disjointness-Ordering-and-Boundedness "Remark 2.9 Complementation Disjointness, Ordering, and Boundedness"){.headerlink}Remark 2.9 Complementation Disjointness, Ordering, and Boundedness {#Remark-2-9-Complementation-Disjointness-Ordering-and-Boundedness style="scroll-margin: 1em;"}

The complement $B$ of a layout $A$ with respect to an integer $M$ should satisfy three properties:

1.  $A$ and $B$ are *disjoint* in the sense that $f_{A}(x) \neq {\hat{f}}_{B}(y)$ for all $x \neq 0$ and $y$ in the domain of $f_{A}$.
2.  $B$ is *ordered* in the sense that $f_{B}(x)$ is a strictly increasing function.
3.  $B$ is *bounded* by $M$ in the sense that $\text{size}(B) \geq \frac{M}{\text{size}(A)}$ and $\text{cosize}(B) \leq \left\lfloor \frac{M}{\text{cosize}(A)} \right\rfloor \cdot \text{cosize}(A)$. Here the cosize of a layout $L$ is defined as $\text{cosize}(L) = f_{L}(\text{size}(L) - 1) + 1$.

The property 1 and 2 have been proved in the Corollary 2.8 and Definition 2.6. We will show a proof of the property 3.

*Proof*

By Definition 2.6, we have $\text{size}(B) = \frac{M}{\text{size}(A)}$.

Because cosize is insensitive to the ordering of the layout, without loss of generality, we sorted $A$ so that $A = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$.

By the definition of cosize, we have

$$\begin{matrix}
{\text{cosize}(B)} & {= f_{B}(\text{size}(B) - 1) + 1} \\
 & {= f_{B}\left( d_{0} - 1,\frac{d_{1}}{N_{0}d_{0}} - 1,\ldots,\frac{d_{\alpha}}{N_{\alpha - 1}d_{\alpha - 1}} - 1,\frac{M}{N_{\alpha}d_{\alpha}} - 1 \right) + 1} \\
 & {= (d_{0} - 1) + \left( \frac{d_{1}}{N_{0}d_{0}} - 1 \right) \cdot N_{0}d_{0} + \ldots + \left( \frac{d_{\alpha}}{N_{\alpha - 1}d_{\alpha - 1}} - 1 \right) \cdot N_{\alpha - 1}d_{\alpha - 1} + \left( \frac{M}{N_{\alpha}d_{\alpha}} - 1 \right) \cdot N_{\alpha}d_{\alpha} + 1} \\
 & {= d_{0} + d_{1} + \ldots + d_{\alpha} - N_{0}d_{0} - N_{1}d_{1} - \ldots - N_{\alpha}d_{\alpha} + M} \\
 & {= M - \left( \left( N_{0} - 1 \right)d_{0} + \left( N_{1} - 1 \right)d_{1} + \ldots + \left( N_{\alpha} - 1 \right)d_{\alpha} \right)} \\
 & {= M - f_{A}(\text{size}(A) - 1)} \\
 & {= M - \left( \text{cosize}(A) - 1 \right)}
\end{matrix}$$

To obtain the inequality $\text{cosize}(B) \leq \left\lfloor \frac{M}{\text{cosize}(A)} \right\rfloor \cdot \text{cosize}(A)$, we divide the above equation by $\text{cosize}(A)$.

$$\begin{matrix}
\frac{\text{cosize}(B)}{\text{cosize}(A)} & {= \frac{M - \left( \text{cosize}(A) - 1 \right)}{\text{cosize}(A)}} \\
 & {= \frac{M}{\text{cosize}(A)} - 1 + \frac{1}{\text{cosize}(A)}}
\end{matrix}$$

and we have to show that

$$\begin{array}{r}
{\frac{M}{\text{cosize}(A)} - 1 + \frac{1}{\text{cosize}(A)} \leq \left\lfloor \frac{M}{\text{cosize}(A)} \right\rfloor}
\end{array}$$

In fact, for any $a,b \in \mathbb{N}$ and $a \geq 1$, we have

$$\begin{array}{r}
{\frac{b}{a} - 1 + \frac{1}{a} \leq \left\lfloor \frac{b}{a} \right\rfloor}
\end{array}$$

To see this, suppose $\frac{b}{a} = \left\lfloor \frac{b}{a} \right\rfloor + c$, where $c = \frac{k}{a}$ and $k$ is an integer such that $0 \leq k < a$. Then we have

$\frac{1}{a} \leq c < 1$ and $1 \leq ac < a$. Then we want to show that

$$\begin{array}{r}
{\frac{b}{a} - 1 + \frac{1}{a} \leq \frac{b}{a} - c}
\end{array}$$

$$\begin{array}{r}
{- a + 1 \leq - ac}
\end{array}$$

$$\begin{array}{r}
{a - ac \geq 1}
\end{array}$$

$$\begin{array}{r}
{a - k \geq 1}
\end{array}$$

Because $a$ and $k$ are both integers and $0 \leq k < a$, we have $a - k \geq 1$. Thus the inequality holds.

This concludes the proof.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Non-Integral-Layout-Complementation "Non-Integral Layout Complementation"){.headerlink}Non-Integral Layout Complementation {#Non-Integral-Layout-Complementation style="scroll-margin: 1em;"}

All the properties and proofs of complementation above assumes that the layout being complemented is a layout whose mode is integral, i.e., $A = (N_{0},N_{1},\ldots,N_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$. In the case where the layout is non-integral, i.e., some of the modes are not integers, the layout shall be coalesced to an integral layout before the complementation is applied. This is valid because coalescence does not change the layout function.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Implication-of-Complementation "Implication of Complementation"){.headerlink}Implication of Complementation {#Implication-of-Complementation style="scroll-margin: 1em;"}

The complementation of a layout finds a complement layout with a positive integer so that when the two layouts are concatenated, such as $\left( B,\text{complement}(B,M) \right)$, the new layout is a bijection $\lbrack 0,M) \cong \lbrack 0,M)$. This is also saying, if the original layout is repeated using the complement layout, the new layout is still a bijection.

## [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Composition "Composition"){.headerlink}Composition {#Composition style="scroll-margin: 1em;"}

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-11-Left-Divisibility "Definition 2.11 Left Divisibility"){.headerlink}Definition 2.11 Left Divisibility {#Definition-2-11-Left-Divisibility style="scroll-margin: 1em;"}

Let $M,d > 0$ be positive integers and let $M = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha}$ be a given factorization of $M$ by integers $M_{k} > 1$ for $k \in \lbrack 0,\alpha\rbrack$. Replacing $M_{\alpha}$ by $\infty$, let

$$\begin{array}{r}
{\hat{M} = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1} \cdot \infty}
\end{array}$$

and consider $\infty$ to be divisible by every positive integer. We say that $M$ is *left divisible* by $d$ (implicitly, with respect to the given factorization) if there exists $0 \leq i \leq \alpha$ such that:

1.  $M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1}$ divides $d$.
2.  Suppose the first condition is satisfied. Let $c = \frac{d}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1}}$. Then if $i < \alpha$, we require in addition that $1 \leq c < M_{i}$.
3.  For the second condition in the case $i < \alpha$, we require in addition that $c$ also divides $M_{i}$.

Here $i$ is necessarily unique if it exists. We could prove this by contradiction.

*Proof*

Suppose there exists two distinct $i$ and $j$ such that the three conditions are satisfied. Without loss of generality, suppose $i < j$.

There are two cases to consider.

In the case where $j < \alpha$, we will also have $i < \alpha$. Then we have

$$\begin{matrix}
d & {= c \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1}}
\end{matrix}$$

where $c$ is some positive integer such that $1 \leq c < M_{i}$.

Similarly,

$$\begin{matrix}
d & {= c^{\prime} \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{j - 1}}
\end{matrix}$$

where $c^{\prime}$ is some positive integer such that $1 \leq c^{\prime} < M_{j}$.

Thus,

$$\begin{matrix}
{c \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1}} & {= c^{\prime} \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{j - 1}}
\end{matrix}$$

$$\begin{matrix}
c & {= c^{\prime} \cdot M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}}
\end{matrix}$$

To make the above equation valid, we must show

$$\begin{array}{r}
{c^{\prime} \cdot \frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1} = 1}
\end{array}$$

However, because $M_{k} > 1$ for $k \in \lbrack 0,\alpha\rbrack$, $\frac{M_{i}}{c} > 1$, and $c^{\prime} \geq 1$, it is not possible to have the above equation valid. This raises a contradiction. Therefore, $i$ is unique.

In the case where $j = \alpha$, we will also have $i < \alpha$. Then we have

$$\begin{matrix}
d & {= c \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1}}
\end{matrix}$$

where $c$ is some positive integer such that $1 \leq c < M_{i}$.

Similarly,

$$\begin{matrix}
d & {= c^{\prime} \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}}
\end{matrix}$$

where $c^{\prime}$ is some positive integer.

Thus,

$$\begin{matrix}
{c \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1}} & {= c^{\prime} \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}}
\end{matrix}$$

$$\begin{matrix}
c & {= c^{\prime} \cdot M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 1}}
\end{matrix}$$

To make the above equation valid, we must show

$$\begin{array}{r}
{c^{\prime} \cdot \frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 1} = 1}
\end{array}$$

However, because $M_{k} > 1$ for $k \in \lbrack 0,\alpha\rbrack$, $\frac{M_{i}}{c} > 1$, and $c^{\prime} \geq 1$, it is not possible to have the above equation valid. This raises a contradiction. Therefore, $i$ is unique.

Taken together, $i$ is unique if it exists.

This concludes the proof. 

If $i$ exists, we will refer to $i$ as the *division index* and write $\hat{M} = d \cdot {\hat{M}}^{\prime}$, where ${\hat{M}}^{\prime}$ is endowed with the following induced factorization:

1.  If $0 \leq i < \alpha$, then ${\hat{M}}^{\prime} = {\hat{M}}_{0}^{\prime} \cdot {\hat{M}}_{1}^{\prime} \cdot \ldots \cdot {\hat{M}}_{\alpha - i - 1}^{\prime} \cdot \infty$ with ${\hat{M}}_{0}^{\prime} = \frac{M_{i}}{c} > 1$ and ${\hat{M}}_{j}^{\prime} = M_{i + j}$ for $0 < j < \alpha - i$.
2.  If $i = \alpha$, then $\hat{M} = d \cdot \infty$ and we will let ${\hat{M}}^{\prime} = \infty$.

To see this, in the case where $0 \leq i < \alpha$, we have

$$\begin{matrix}
\hat{M} & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1} \cdot \infty} \\
 & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1} \cdot M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 1} \cdot \infty} \\
 & {= \frac{d}{c} \cdot M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 1} \cdot \infty} \\
 & {= d \cdot \frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 1} \cdot \infty} \\
 & {= d \cdot M_{0}^{\prime} \cdot M_{1}^{\prime} \cdot \ldots \cdot M_{\alpha - i - 1}^{\prime} \cdot \infty} \\
 & {= d \cdot {\hat{M}}^{\prime}}
\end{matrix}$$

where ${\hat{M}}^{\prime} = M_{0}^{\prime} \cdot M_{1}^{\prime} \cdot \ldots \cdot M_{\alpha - i - 1}^{\prime} \cdot \infty$ with $M_{0}^{\prime} = \frac{M_{i}}{c} > 1$ and $M_{j}^{\prime} = M_{i + j}$ for $0 < j < \alpha - i$.

In the case where $i = \alpha$, we have

$$\begin{matrix}
\hat{M} & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1} \cdot \infty} \\
 & {= \frac{d}{c} \cdot \infty} \\
 & {= d \cdot \infty} \\
 & {= d \cdot {\hat{M}}^{\prime}}
\end{matrix}$$

where ${\hat{M}}^{\prime} = \infty$.

Furthermore, we say that $M$ is *weakly left divisible* by $d$ if there exists $0 \leq i \leq \alpha$ such that the conditions 1 and 2 are satisfied for left divisibility, but not necessarily the condition 3.

Notice that in the proof of the uniqueness of division index $i$, we have never used the condition 3. Therefore, we could still the necessarily unique $i$ the division index for weak left divisibility, but we no longer have the factorization of $\hat{M}$, because the factorization assumes the condition 3 of left divisibility.

Also notice that ${\hat{M}}^{\prime}$ with its induced factorization can itself be considered for left divisibility or weak left divisibility (with the step or replacing the last factor by $\infty$ now being superfluous). More specifically, because ${\hat{M}}^{\prime} > 0$, ${\hat{M}}_{j}^{\prime} > 1$ for $j \in \lbrack 0,\alpha - i - 1\rbrack$, and ${\hat{M}}^{\prime} = {\hat{M}}_{0}^{\prime} \cdot {\hat{M}}_{1}^{\prime} \cdot \ldots \cdot {\hat{M}}_{\alpha - i - 1}^{\prime} \cdot \infty$, given another positive integer $d^{\prime} > 0$, we could completely test whether the properties of left divisibility or weak left divisibility hold for ${\hat{M}}^{\prime}$ with respect to $d^{\prime}$. Replacing the last factor by $\infty$ is not necessary as it is already $\infty$.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-12-Admission-for-Composition-Restricted-Case "Definition 2.12 Admission for Composition - Restricted Case"){.headerlink}Definition 2.12 Admission for Composition - Restricted Case {#Definition-2-12-Admission-for-Composition-Restricted-Case style="scroll-margin: 1em;"}

We first consider composition in the restricted case of length 1 layouts for the second layout.

Let $\mathbf{S} = (M_{0},M_{1},\ldots,M_{\alpha})$ be a shape tuple, let $M = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha}$, and let $B = (N):(r)$ be a layout of length 1. Then we say that the pair $\{\mathbf{S},B\}$ is *admissible for composition* (or simply admissible) if:

1.  $M$ is left divisible by $r$. Write $\hat{M} = r \cdot {\hat{M}}^{\prime}$.
2.  With respect to its induced factorization, ${\hat{M}}^{\prime}$ is weakly left divisible by $N$.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-13-Composition-Restricted-Case "Definition 2.13 Composition - Restricted Case"){.headerlink}Definition 2.13 Composition - Restricted Case {#Definition-2-13-Composition-Restricted-Case style="scroll-margin: 1em;"}

The idea of admissibility is that the composition $A \circ B$ of layouts will entail "dividing $B$ along the modes of $A$". More preciously, we have the following:

Suppose that $\mathbf{S} = (M_{0},M_{1},\ldots,M_{\alpha})$ is a shape tuple, and $B = (N):(r)$ is a layout of length 1 such that $\{\mathbf{S},B\}$ is admissible for composition. Let $\mathbf{D} = (d_{0},d_{1},\ldots,d_{\alpha})$ be any stride tuple and let $A = (\mathbf{S}:\mathbf{D})$ be a **coalesced** layout.

Note that in Jay Shah's original paper, the layout $A$ was not specified to be coalesced. It will result in some compositions not being valid.

As in Definition 2.11, let $M = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha}$ and $\hat{M} = r \cdot {\hat{M}}^{\prime}$ with division index $0 \leq i \leq \alpha$. We separate the definition of $A \circ B$ into two cases.

First suppose $0 \leq i < \alpha$, so that

$$\begin{matrix}
r & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1} \cdot c} \\
{\hat{M}}^{\prime} & {= \frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 1} \cdot \infty}
\end{matrix}$$

Then if $N \leq \frac{M_{i}}{c}$, we let $A \circ B = (N):(cd_{i})$.

Otherwise, there exists a $j \in \lbrack i + 1,\alpha\rbrack$ such that $N = \frac{M_{i}}{c} \cdot \ldots \cdot M_{j - 1} \cdot c^{\prime}$, where $1 \leq c^{\prime} < M_{j}$ if $j \neq \alpha$ (when $j = i + 1$, $N = \frac{M_{i}}{c} \cdot c^{\prime}$).

Note that here is an important fact that $c^{\prime}$ must be an integer because of the second condition for admission for composition, that is, ${\hat{M}}^{\prime}$ is weakly left divisible by $N$. We must have $\frac{M_{i}}{c} \cdot \ldots \cdot M_{j - 1}$ divides $N$, resulting in $c^{\prime}$ being an integer.

We let

$$\begin{array}{r}
{A \circ B = \left\{ \begin{matrix}
{\left( \frac{M_{i}}{c},M_{i + 1},\ldots,M_{j - 1},c^{\prime} \right):\left( cd_{i},d_{i + 1},\ldots,d_{j - 1},d_{j} \right)} & {\text{if~}c^{\prime} > 1} \\
{\left( \frac{M_{i}}{c},M_{i + 1},\ldots,M_{j - 1} \right):\left( cd_{i},d_{i + 1},\ldots,d_{j - 1} \right)} & {\text{if~}c^{\prime} = 1}
\end{matrix} \right.}
\end{array}$$

If instead $i = \alpha$, then we have $r = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1} \cdot c$ as before but ${\hat{M}}^{\prime} = \infty$, and we let $A \circ B = (N):(cd_{\alpha})$.

Let's look at this definition more closely.

Essentially, we are taking the one-dimensional coordinates $k \cdot r$ along the layout $A$ where $k \in \lbrack 0,N - 1\rbrack$. Because we have $r = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1} \cdot c$, and $c$ divides $M_{i}$.

Let first consider the case of $0 \leq i < \alpha$.

If $N \leq \frac{M_{i}}{c}$, then the first mode in the layout $A$ is sufficient for dividing $B$. Consequently, the composition layout $A \circ B = (N):(cd_{i})$.

Otherwise if $N > \frac{M_{i}}{c}$, more modes in the layout $A$ will be involved for dividing $B$, and consequently the composition layout

$$\begin{array}{r}
{A \circ B = \left\{ \begin{matrix}
{\left( \frac{M_{i}}{c},M_{i + 1},\ldots,M_{j - 1},c^{\prime} \right):\left( cd_{i},d_{i + 1},\ldots,d_{j - 1},d_{j} \right)} & {\text{if~}c^{\prime} > 1} \\
{\left( \frac{M_{i}}{c},M_{i + 1},\ldots,M_{j - 1} \right):\left( cd_{i},d_{i + 1},\ldots,d_{j - 1} \right)} & {\text{if~}c^{\prime} = 1}
\end{matrix} \right.}
\end{array}$$

Let's then consider the case of $i = \alpha$. We have $r = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1} \cdot c$ and ${\hat{M}}^{\prime} = \infty$. Here $c$ is an positive integer that can be infinitely large. So only the last mode in the layout $A$ is involved for dividing $B$, and consequently the composition layout $A \circ B = (N):(cd_{\alpha})$.

Note that by this definition, $\text{size}(A \circ B) = \text{size}(B)$. This is a critical property which we will use later.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Proposition-2-14 "Proposition 2.14"){.headerlink}Proposition 2.14 {#Proposition-2-14 style="scroll-margin: 1em;"}

In the situation of Definition 2.13, we have that $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$.

*Proof*

This more formally proves the intuition we explained to Definition 2.13.

We carry over the notation from Definition 2.13.

Given an index $0 \leq k \leq \alpha$, let $\delta_{k} \in \mathbb{N}^{\times (\alpha + 1)}$ denote the coordinate that is zero everywhere except in the $k$-th position, where it is 1. Concretely,

$$\begin{matrix}
\delta_{0} & {= \underset{\alpha + 1}{\underbrace{(1,0,0,\ldots,0)}}} \\
\delta_{1} & {= \underset{\alpha + 1}{\underbrace{(0,1,0,\ldots,0)}}} \\
 & {\vdots} \\
\delta_{\alpha} & {= \underset{\alpha + 1}{\underbrace{(0,0,0,\ldots,1)}}}
\end{matrix}$$

With respect to the the isomorphism of the extended layout $A$, we have

$$\begin{array}{r}
{\hat{\iota}:\mathbb{N} \cong \lbrack 0,M_{0}) \times \lbrack 0,M_{1}) \times \ldots \times \lbrack 0,M_{\alpha - 1}) \times \mathbb{N}}
\end{array}$$

Because $B = (N):(r)$, we have

$$\begin{matrix}
{f_{B}(k)} & {= k \cdot r} \\
 & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1} \cdot k \cdot c}
\end{matrix}$$

where $k \in \lbrack 0,N - 1\rbrack$.

Let first consider the case of $0 \leq i < \alpha$.

If $N \leq \frac{M_{i}}{c}$, i.e. $N \cdot c \leq M_{i}$, then we must have $k \cdot c < M_{i}$ for all $k \in \lbrack 0,N - 1\rbrack$. Because of the isomorphism of the extended layout $A$, we have

$$\begin{array}{r}
{f_{B}(k)\mapsto\delta_{i} \cdot k \cdot c}
\end{array}$$

Then we have

$$\begin{matrix}
{\left( {\hat{f}}_{A} \circ f_{B} \right)(k)} & {= {\hat{f}}_{A}\left( f_{B}(k) \right)} \\
 & {= {\hat{f}}_{A}\left( \delta_{i} \cdot k \cdot c \right)} \\
 & {= k \cdot c \cdot d_{i}}
\end{matrix}$$

According to Definition 2.13, we have

$$\begin{matrix}
{f_{A \circ B}(k)} & {= k \cdot c \cdot d_{i}}
\end{matrix}$$

Therefore, $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$.

Otherwise if $N > \frac{M_{i}}{c}$, i.e. $N = \frac{M_{i}}{c} \cdot \ldots \cdot M_{j - 1} \cdot c^{\prime}$. Because of the isomorphism of the extended layout $A$, by definition, we have

$$\begin{matrix}
{f_{B}(k)} & {\mapsto\left( f_{B}(k)\quad{mod}\,\, M_{0},\left\lfloor \frac{f_{B}(k)}{M_{0}} \right\rfloor\quad{mod}\,\, M_{1},\left\lfloor \frac{f_{B}(k)}{M_{0} \cdot M_{1}} \right\rfloor\quad{mod}\,\, M_{2},\ldots,\left\lfloor \frac{f_{B}(k)}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1}} \right\rfloor\quad{mod}\,\, M_{i},\ldots,\left\lfloor \frac{f_{B}(k)}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 2}} \right\rfloor\quad{mod}\,\, M_{\alpha - 1},\left\lfloor \frac{f_{B}(k)}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor \right)} \\
 & {= \left( 0,0,\ldots,(k \cdot c)\quad{mod}\,\, M_{i},\left\lfloor \frac{k \cdot c}{M_{i}} \right\rfloor\quad{mod}\,\, M_{i + 1},\ldots,\left\lfloor \frac{k \cdot c}{M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, M_{j},\ldots,\left\lfloor \frac{k \cdot c}{M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 2}} \right\rfloor\quad{mod}\,\, M_{\alpha - 1},\left\lfloor \frac{k \cdot c}{M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor \right)} \\
 & {= \left( 0,0,\ldots,\left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c,\left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1},\ldots,\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, M_{j},\ldots,\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 2}} \right\rfloor\quad{mod}\,\, M_{\alpha - 1},\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor \right)}
\end{matrix}$$

Note that here we used the property $(k \cdot c) \equiv \left( k \bmod \frac{M_i}{c} \right) \cdot c \pmod{M_i}$, if $c$ divides $M_i$.

To see this, suppose $k \cdot c = p \cdot M_{i} + r$, where $0 \leq r < M_{i}$. Then we have $k = \frac{p \cdot M_{i} + r}{c} = p \cdot \frac{M_{i}}{c} + \frac{r}{c}$. Because $c$ divides $M_{i}$, $\frac{r}{c}$ must be an integer. Thus, we have $(k \cdot c) \mod M_{i} = r$, and $\left( k \mod \frac{M_{i}}{c} \right) \cdot c = \frac{r}{c} \cdot c = r$. Therefore, $(k \cdot c) \mod M_{i} = \left( k \mod \frac{M_{i}}{c} \right) \cdot c$.

Further more, because $0 \leq k < N$, we have $0 \leq k \cdot c < N \cdot c = M_{i} \cdot \ldots \cdot M_{j - 1} \cdot c^{\prime}$, where $1 \leq c^{\prime} < M_{j}$. Thus $0 \leq k \cdot c < M_{i} \cdot \ldots \cdot M_{j - 1} \cdot M_{j}$.

When $c^{\prime} > 1$, we have

$$\begin{matrix}
\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j}} \right\rfloor & {= \left\lfloor \frac{k \cdot c}{M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{j}} \right\rfloor} \\
 & {= 0}
\end{matrix}$$

$$\begin{matrix}
{\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j}} \right\rfloor\quad{mod}\,\, M_{j + 1}} & {= 0}
\end{matrix}$$

and of course for any $l \in \lbrack j + 1,\alpha - 1\rbrack$, we have

$$\begin{matrix}
\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{l}} \right\rfloor & {= \left\lfloor \frac{k \cdot c}{M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{l}} \right\rfloor} \\
 & {= 0}
\end{matrix}$$

$$\begin{matrix}
{\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{l}} \right\rfloor\quad{mod}\,\, M_{l + 1}} & {= 0}
\end{matrix}$$

Thus, we have

$$\begin{matrix}
{f_{B}(k)} & {\mapsto\left( 0,0,\ldots,\left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c,\left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1},\ldots,\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, M_{j},0,0,\ldots,0 \right)}
\end{matrix}$$

What's more, because $k \leq (N - 1)$ and $\frac{M_{i}}{c} \cdot \ldots \cdot M_{j - 1} = \frac{N}{c^{\prime}}$, we have

$$\begin{matrix}
\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor & {= \left\lfloor \frac{k}{\frac{N}{c^{\prime}}} \right\rfloor} \\
 & {= \left\lfloor \frac{k}{N} \cdot c^{\prime} \right\rfloor} \\
 & {\leq \left\lfloor \frac{N - 1}{N} \cdot c^{\prime} \right\rfloor} \\
 & {\leq \left\lfloor c^{\prime} \right\rfloor} \\
 & {\leq c^{\prime}}
\end{matrix}$$

Because $c^{\prime} < M_{j}$, we have

$$\begin{matrix}
{\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, M_{j}} & {= \left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, c^{\prime}}
\end{matrix}$$

Thus, we have

$$\begin{matrix}
{f_{B}(k)} & {\mapsto\left( 0,0,\ldots,\left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c,\left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1},\ldots,\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, c^{\prime},0,0,\ldots,0 \right)}
\end{matrix}$$

Then we have

$$\begin{matrix}
{\left( {\hat{f}}_{A} \circ f_{B} \right)(k)} & {= {\hat{f}}_{A}\left( f_{B}(k) \right)} \\
 & {= {\hat{f}}_{A}\left( 0,0,\ldots,\left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c,\left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1},\ldots,\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, c^{\prime},0,0,\ldots,0 \right)} \\
 & {= 0 \cdot d_{0} + 0 \cdot d_{1} + \ldots + \left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c \cdot d_{i} + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1} \right) \cdot d_{i + 1} + \ldots + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, c^{\prime} \right) \cdot d_{j} + 0 \cdot d_{j + 1} + \ldots + 0 \cdot d_{\alpha}} \\
 & {= \left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c \cdot d_{i} + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1} \right) \cdot d_{i + 1} + \ldots + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, c^{\prime} \right) \cdot d_{j}}
\end{matrix}$$

According to Definition 2.13, we have

$$\begin{matrix}
{f_{A \circ B}(k)} & {= \left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c \cdot d_{i} + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1} \right) \cdot d_{i + 1} + \ldots + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, c^{\prime} \right) \cdot d_{j}}
\end{matrix}$$

Therefore, $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$.

When $c^{\prime} = 1$, we have $0 \leq k \cdot c < N \cdot c = M_{i} \cdot \ldots \cdot M_{j - 1} \cdot c^{\prime} = M_{i} \cdot \ldots \cdot M_{j - 1}$.

Thus, we have

$$\begin{matrix}
\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor & {= \left\lfloor \frac{k \cdot c}{M_{i} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor} \\
 & {= 0}
\end{matrix}$$

$$\begin{matrix}
{\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 1}} \right\rfloor\quad{mod}\,\, M_{j}} & {= 0}
\end{matrix}$$

Thus, we have

$$\begin{matrix}
{f_{B}(k)} & {\mapsto\left( 0,0,\ldots,\left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c,\left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1},\ldots,\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 2}} \right\rfloor\quad{mod}\,\, M_{j - 1},0,0,\ldots,0 \right)}
\end{matrix}$$

Then we have

$$\begin{matrix}
{\left( {\hat{f}}_{A} \circ f_{B} \right)(k)} & {= {\hat{f}}_{A}\left( f_{B}(k) \right)} \\
 & {= {\hat{f}}_{A}\left( 0,0,\ldots,\left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c,\left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1},\ldots,\left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 2}} \right\rfloor\quad{mod}\,\, M_{j - 1},0,0,\ldots,0 \right)} \\
 & {= 0 \cdot d_{0} + 0 \cdot d_{1} + \ldots + \left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c \cdot d_{i} + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1} \right) \cdot d_{i + 1} + \ldots + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 2}} \right\rfloor\quad{mod}\,\, M_{j - 1} \right) \cdot d_{j - 1} + 0 \cdot d_{j} + 0 \cdot d_{j + 1} + \ldots + 0 \cdot d_{\alpha}} \\
 & {= \left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c \cdot d_{i} + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1} \right) \cdot d_{i + 1} + \ldots + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 2}} \right\rfloor\quad{mod}\,\, M_{j - 1} \right) \cdot d_{j - 1}}
\end{matrix}$$

According to Definition 2.13, we have

$$\begin{matrix}
{f_{A \circ B}(k)} & {= \left( k\quad{mod}\,\,\frac{M_{i}}{c} \right) \cdot c \cdot d_{i} + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c}} \right\rfloor\quad{mod}\,\, M_{i + 1} \right) \cdot d_{i + 1} + \ldots + \left( \left\lfloor \frac{k}{\frac{M_{i}}{c} \cdot M_{i + 1} \cdot \ldots \cdot M_{j - 2}} \right\rfloor\quad{mod}\,\, M_{j - 1} \right) \cdot d_{j - 1}}
\end{matrix}$$

Therefore, $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$.

Let's then consider the case of $i = \alpha$.

$$\begin{matrix}
{f_{B}(k)} & {= k \cdot r} \\
 & {= k \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1} \cdot c}
\end{matrix}$$

where $k \in \lbrack 0,N - 1\rbrack$.

Because of the isomorphism of the extended layout $A$, we have

$$\begin{matrix}
{f_{B}(k)} & {\mapsto\delta_{\alpha} \cdot k \cdot c}
\end{matrix}$$

Then we have

$$\begin{matrix}
{\left( {\hat{f}}_{A} \circ f_{B} \right)(k)} & {= {\hat{f}}_{A}\left( f_{B}(k) \right)} \\
 & {= {\hat{f}}_{A}\left( \delta_{\alpha} \cdot k \cdot c \right)} \\
 & {= k \cdot c \cdot d_{\alpha}}
\end{matrix}$$

According to Definition 2.13, we have

$$\begin{matrix}
{f_{A \circ B}(k)} & {= k \cdot c \cdot d_{\alpha}}
\end{matrix}$$

Therefore, $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$.

Taken together, we have $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$ for all the cases in Definition 2.13.

This concludes the proof. 

One might ask why the second condition for admission for composition is necessary. If we don't have it, $c^{\prime}$ can be fractional and we can still define $A \circ B$ to be

$$\begin{array}{r}
{A \circ B = \left\{ \begin{matrix}
{\left( \frac{M_{i}}{c},M_{i + 1},\ldots,M_{j - 1},\left\lceil c^{\prime} \right\rceil \right):\left( cd_{i},d_{i + 1},\ldots,d_{j - 1},d_{j} \right)} & {\text{if~}c^{\prime} > 1} \\
{\left( \frac{M_{i}}{c},M_{i + 1},\ldots,M_{j - 1} \right):\left( cd_{i},d_{i + 1},\ldots,d_{j - 1} \right)} & {\text{if~}c^{\prime} = 1}
\end{matrix} \right.}
\end{array}$$

It's not too difficult to show that we still have $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$ for the domain of $f_{B}$ when the length of $B$ is 1.

However, the critical property $\text{size}(A \circ B) = \text{size}(B)$ will not hold in this case. As we will see later, without having this property, $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$ cannot be true when $B$ is multi-modal, i.e., the length of $B$ is greater than 1.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-16-Interval-of-Definition "Definition 2.16 Interval of Definition"){.headerlink}Definition 2.16 Interval of Definition {#Definition-2-16-Interval-of-Definition style="scroll-margin: 1em;"}

In the situation of Definition 2.12, where layout $B$ is of length 1, let $f_{B}:\lbrack 0,N) \rightarrow \mathbb{N}$ be the layout function, and let $I = \lbrack r,r(N - 1)\rbrack$ be the interval given by the convex closure of the image $f_{B}(\lbrack 1,N))$. Let $M^{\prime} = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}$ and $J = I \cap \lbrack 1,M^{\prime})$ (so $J = \varnothing$ if $\alpha = 0$). Then the *interval of definition* for $\{\mathbf{S},B\}$ is $J$.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-17-Composition-General-Case "Definition 2.17 Composition - General Case"){.headerlink}Definition 2.17 Composition - General Case {#Definition-2-17-Composition-General-Case style="scroll-margin: 1em;"}

Let $\mathbf{S} = (M_{0},M_{1},\ldots,M_{\alpha})$ be a shape tuple, and let $B = (N_{0},N_{1},\ldots,N_{\beta}):(r_{0},r_{1},\ldots,r_{\beta})$ be a layout, let $B_{k} = (N_{k}):(r_{k})$ for $0 \leq k \leq \beta$. Then we say that the pair $\{\mathbf{S},B\}$ is *admissible for composition* if:

1.  For all $0 \leq k \leq \beta$, the pair $\{\mathbf{S},B_{k}\}$ is admissible for composition in the sense of Definition 2.12.
2.  The interval of definition for the pairs $\{\mathbf{S},B_{k}\}_{0 \leq k \leq \beta}$ are disjoint.

In this case, if $\mathbf{D} = (d_{0},d_{1},\ldots,d_{\alpha})$ is a stride tuple and $A = \mathbf{S}:\mathbf{D}$, then we define the composition $A \circ B$ to be the concatenated layout

$$\begin{array}{r}
{A \circ B:=\left( A \circ B_{0},A \circ B_{1},\ldots,A \circ B_{\beta} \right)}
\end{array}$$

where each $A \circ B_{k}$ is defined as in Definition 2.13.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Theorem-2-18-Composition-General-Case "Theorem 2.18 Composition - General Case"){.headerlink}Theorem 2.18 Composition - General Case {#Theorem-2-18-Composition-General-Case style="scroll-margin: 1em;"}

In the situation of Definition 2.17, we have that $f_{A \circ B} = {\hat{f}}_{A} \circ f_{B}$.

*Proof*

By Definition 2.13, $\text{size}(A \circ B_{k}) = \text{size}(B_{k}) = N_{k}$ for all $0 \leq k \leq \beta$. We have the following isomorphism for both the layout $A \circ B$ or the layout $B$.

$$\begin{array}{r}
{\iota:\lbrack 0,N_{0} \cdot N_{1} \cdot \ldots \cdot N_{\beta}) \cong \lbrack 0,N_{0}) \times \lbrack 0,N_{1}) \times \ldots \times \lbrack 0,N_{\beta})}
\end{array}$$

Given any $x \in \lbrack 0,N_{0} \cdot N_{1} \cdot \ldots \cdot N_{\beta})$, because of the isomorphism $\iota$, we have

$$\begin{matrix}
x & {\mapsto\left( x_{0},x_{1},\ldots,x_{\beta} \right)}
\end{matrix}$$

By Lemma 2.19, we have

$$\begin{matrix}
{{\hat{f}}_{A} \circ f_{B}(x)} & {= {\hat{f}}_{A}\left( f_{B}(x) \right)} \\
 & {= {\hat{f}}_{A}\left( f_{B_{0}}(x_{0}) + f_{B_{1}}(x_{1}) + \ldots + f_{B_{\beta}}(x_{\beta}) \right)}
\end{matrix}$$

By Definition 2.17, Lemma 2.19, and Definition 2.13, we have

$$\begin{matrix}
{f_{A \circ B}(x)} & {= f_{A \circ B_{0}}(x_{0}) + f_{A \circ B_{1}}(x_{1}) + \ldots + f_{A \circ B_{\beta}}(x_{\beta})} \\
 & {= {\hat{f}}_{A} \circ f_{B_{0}}(x_{0}) + {\hat{f}}_{A} \circ f_{B_{1}}(x_{1}) + \ldots + {\hat{f}}_{A} \circ f_{B_{\beta}}(x_{\beta})} \\
 & {= {\hat{f}}_{A}\left( f_{B_{0}}(x_{0}) \right) + {\hat{f}}_{A}\left( f_{B_{1}}(x_{1}) \right) + \ldots + {\hat{f}}_{A}\left( f_{B_{\beta}}(x_{\beta}) \right)}
\end{matrix}$$

Normally, we don't have ${\hat{f}}_{A}\left( x_{A,0} + x_{A,1} + \ldots + x_{A,\beta} \right) = {\hat{f}}_{A}\left( x_{A,0} \right) + {\hat{f}}_{A}\left( x_{A,1} \right) + \ldots + {\hat{f}}_{A}\left( x_{A,\beta} \right)$, because the layout function ${\hat{f}}_{A}$ is not linear. For example, suppose $A = (2,3):(1,4)$, and we have ${\hat{f}}_{A}(1) = 1$ and ${\hat{f}}_{A}(3) = 5$. ${\hat{f}}_{A}(3) = {\hat{f}}_{A}(1 + 1 + 1) \neq {\hat{f}}_{A}(1) + {\hat{f}}_{A}(1) + {\hat{f}}_{A}(1)$.

However, there are some special cases where the above equation holds. For example, for simplicity, suppose $\beta = \alpha$, if we have

$$\begin{matrix}
x_{A,0} & {\in \lbrack 0,M_{0})} \\
x_{A,1} & {\in \{ 0,1 \cdot M_{0},2 \cdot M_{0},\ldots,\infty \cdot M_{0}\} \cap \lbrack 0,M_{1})} \\
x_{A,2} & {\in \{ 0,1 \cdot M_{0} \cdot M_{1},2 \cdot M_{0} \cdot M_{1},\ldots,\infty \cdot M_{0} \cdot M_{1}\} \cap \lbrack 0,M_{2})} \\
 & {\vdots} \\
x_{A,\alpha} & {\in \{ 0,1 \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1},2 \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1},\ldots,\infty \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}\} \cap \lbrack 0,M_{\alpha})}
\end{matrix}$$

By definition,

$$\begin{matrix}
{{\hat{f}}_{A}(x)} & {= \left( x\quad{mod}\,\, M_{0} \right) \cdot d_{0} + \left( \left\lfloor \frac{x}{M_{0}} \right\rfloor\quad{mod}\,\, M_{1} \right) \cdot d_{1} + \ldots + \left( \left\lfloor \frac{x}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor\quad{mod}\,\, M_{\alpha} \right) \cdot d_{\alpha}}
\end{matrix}$$

So in our case, we have

$$\begin{matrix}
{{\hat{f}}_{A}\left( x_{A,0} + x_{A,1} + \ldots + x_{A,\beta} \right)} & {= \left( \left( x_{A,0} + x_{A,1} + \ldots + x_{A,\beta} \right)\quad{mod}\,\, M_{0} \right) \cdot d_{0} + \left( \left\lfloor \frac{x_{A,0} + x_{A,1} + \ldots + x_{A,\beta}}{M_{0}} \right\rfloor\quad{mod}\,\, M_{1} \right) \cdot d_{1} + \ldots + \left( \left\lfloor \frac{x_{A,0} + x_{A,1} + \ldots + x_{A,\beta}}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor\quad{mod}\,\, M_{\alpha} \right) \cdot d_{\alpha}} \\
 & {= \left( x_{A,0}\quad{mod}\,\, M_{0} \right) \cdot d_{0} + \left( \left\lfloor \frac{x_{A,1}}{M_{0}} \right\rfloor\quad{mod}\,\, M_{1} \right) \cdot d_{1} + \ldots + \left( \left\lfloor \frac{x_{A,\beta}}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha - 1}} \right\rfloor\quad{mod}\,\, M_{\alpha} \right) \cdot d_{\alpha}} \\
 & {= {\hat{f}}_{A}\left( x_{A,0} \right) + {\hat{f}}_{A}\left( x_{A,1} \right) + \ldots + {\hat{f}}_{A}\left( x_{A,\beta} \right)}
\end{matrix}$$

The idea of having the second condition for admission for composition, i.e., the interval of definition for the pairs $\{\mathbf{S},B_{k}\}_{0 \leq k \leq \beta}$ are disjoint, are exactly the same.

Because $r_{k} = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i_{k} - 1} \cdot c$, for $x_{k} \in \lbrack 0,N_{k})$, we have

$$\begin{matrix}
{f_{B_{k}}(x_{k})} & {\in \lbrack 0,1 \cdot r,2 \cdot r,\ldots,(N_{k} - 1) \cdot r\rbrack} \\
 & {= \lbrack 0,M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i_{k} - 1} \cdot c,2 \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i_{k} - 1} \cdot c,\ldots,(N_{k} - 1) \cdot M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i_{k} - 1} \cdot c\rbrack}
\end{matrix}$$

Because of the isomorphism of the layout $A$, we have

$$\begin{matrix}
{f_{B_{k}}(x_{k})} & {\mapsto\left( x_{A,0},x_{A,1},\ldots,x_{A,\alpha} \right)}
\end{matrix}$$

where

$$\begin{array}{r}
{x_{A,i} = \left\lfloor \frac{f_{B_{k}}(x_{k})}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i - 1}} \right\rfloor\quad{mod}\,\, M_{i}}
\end{array}$$

Then we must have some integer $p_{k}$ and $q_{k}$, $p_{k} < q_{k}$, where $x_{A,i} = 0$ for $i < p_{k}$ and $i > q_{k}$.

The second condition for admission for composition ensures that $\lbrack p_{k},q_{k}\rbrack$ are disjoint for all $0 \leq k \leq \beta$. Therefore, we can have the equation:

$$\begin{array}{r}
{{\hat{f}}_{A}\left( x_{A,0} + x_{A,1} + \ldots + x_{A,\beta} \right) = {\hat{f}}_{A}\left( x_{A,0} \right) + {\hat{f}}_{A}\left( x_{A,1} \right) + \ldots + {\hat{f}}_{A}\left( x_{A,\beta} \right)}
\end{array}$$

This concludes the proof. 

Going back to the discussion why the second condition for admission for composition in the restricted case in Proposition 2.14 is necessary.

Because the critical property $\text{size}(A \circ B) = \text{size}(B)$ will not hold, we will have two completely different isomorphisms for the layout $A \circ B$ and the layout $B$, respectively.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Lemma-2-19-Concatenation-of-Layouts "Lemma 2.19 Concatenation of Layouts"){.headerlink}Lemma 2.19 Concatenation of Layouts {#Lemma-2-19-Concatenation-of-Layouts style="scroll-margin: 1em;"}

Let $C = (C_{0},C_{1},\ldots,C_{\gamma})$ be a *concatenated* layout. Let

$$\begin{array}{r}
{\iota:\lbrack 0,\text{size}(C)) \cong \lbrack 0,\text{size}(C_{0})) \times \cdots \times \lbrack 0,\text{size}(C_{\gamma}))}
\end{array}$$

be the usual isomorphism (as in Definition 2.3). Then the following diagram commutes:

$$\begin{matrix}
{\lbrack 0,\text{size}(C))} & \underset{\cong}{\overset{\iota}{\rightarrow}} & {\lbrack 0,\text{size}(C_{0})) \times \cdots \times \lbrack 0,\text{size}(C_{\gamma}))} & \\
\left. f_{C}\downarrow \right. & & \left. \downarrow{(f_{C_{0}},\ldots,f_{C_{\gamma}})} \right. & \\
\mathbb{N} & \overset{+}{\leftarrow} & {\mathbb{N} \times \cdots \times \mathbb{N}} & 
\end{matrix}$$

*Proof*

If $C_{0},\ldots,C_{\gamma}$ are all length 1 layouts, then this is immediate from Definition 2.3.

Concretely, suppose $C_{k} = (M_{k}):(d_{k})$ for all $0 \leq k \leq \gamma$. The concatenated layout becomes

$$\begin{matrix}
C & {= (C_{0},C_{1},\ldots,C_{\gamma})} \\
 & {= (M_{0}:d_{0},M_{1}:d_{1},\ldots,M_{\gamma}:d_{\gamma})} \\
 & {= (M_{0},M_{1},\ldots,M_{\gamma}):(d_{0},d_{1},\ldots,d_{\gamma})}
\end{matrix}$$

Because of the isomorphism of the layout $C$, we have

$$\begin{matrix}
x & {\mapsto\left( x_{0},x_{1},\ldots,x_{\gamma} \right)}
\end{matrix}$$

Then by definition, the concatenated layout function is

$$\begin{matrix}
{f_{C}(x)} & {= x_{0} \cdot d_{0} + x_{1} \cdot d_{1} + \ldots + x_{\gamma} \cdot d_{\gamma}}
\end{matrix}$$

For each of the length 1 layouts $C_{k}$, by definition, the layout function is

$$\begin{matrix}
{f_{C_{k}}(x_{k})} & {= x_{k} \cdot d_{k}}
\end{matrix}$$

Therefore, we have

$$\begin{matrix}
{f_{C}(x)} & {= f_{C_{0}}(x_{0}) + f_{C_{1}}(x_{1}) + \ldots + f_{C_{\gamma}}(x_{\gamma})}
\end{matrix}$$

In the case where some of the layouts $C_{k}$ are not length 1, we can apply the same argument to each of the sublayouts $C_{k}$, and the result follows by induction.

Concretely, suppose $C_{k}$ are not length 1 and $C_{k} = (C_{k,0},C_{k,1},\ldots,C_{k,\gamma_{k}})$, where $C_{k,0},\ldots,C_{k,\gamma_{k}}$ are length 1 layouts. Based on what we have proved above, we have

$$\begin{matrix}
{f_{C_{k}}(x_{k})} & {= f_{C_{k,0}}(x_{k,0}) + f_{C_{k,1}}(x_{k,1}) + \ldots + f_{C_{k,\gamma_{k}}}(x_{k,\gamma_{k}})}
\end{matrix}$$

where

$$\begin{matrix}
x_{k} & {\mapsto\left( x_{k,0},x_{k,1},\ldots,x_{k,\gamma_{k}} \right)}
\end{matrix}$$

Suppose the layout $C$ can be maximally decomposed into layouts of length 1.

$$\begin{matrix}
C & {= (C_{0},C_{1},\ldots,C_{\gamma})} \\
 & {= (C_{0,0},C_{0,1},\ldots,C_{0,\gamma_{0}},C_{1,0},C_{1,1},\ldots,C_{1,\gamma_{1}},\ldots,C_{\gamma,0},C_{\gamma,1},\ldots,C_{\gamma,\gamma_{\gamma}})}
\end{matrix}$$

Then we have

$$\begin{matrix}
{f_{C}(x)} & {= f_{C_{0,0}}(x_{0,0}) + f_{C_{0,1}}(x_{0,1}) + \ldots + f_{C_{0,\gamma_{0}}}(x_{0,\gamma_{0}}) + f_{C_{1,0}}(x_{1,0}) + f_{C_{1,1}}(x_{1,1}) + \ldots + f_{C_{1,\gamma_{1}}}(x_{1,\gamma_{1}}) + \ldots + f_{C_{\gamma,0}}(x_{\gamma,0}) + f_{C_{\gamma,1}}(x_{\gamma,1}) + \ldots + f_{C_{\gamma,\gamma_{\gamma}}}(x_{\gamma,\gamma_{\gamma}})} \\
 & {= f_{C_{0}}(x_{0}) + f_{C_{1}}(x_{1}) + \ldots + f_{C_{\gamma}}(x_{\gamma})}
\end{matrix}$$

where

$$\begin{matrix}
x & {\mapsto\left( x_{0},x_{1},\ldots,x_{\gamma} \right)}
\end{matrix}$$

This concludes the proof. 

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-21-CUTLASS-Admission-for-Composition-Restricted-Case "Definition 2.21 CUTLASS Admission for Composition - Restricted Case"){.headerlink}Definition 2.21 CUTLASS Admission for Composition - Restricted Case {#Definition-2-21-CUTLASS-Admission-for-Composition-Restricted-Case style="scroll-margin: 1em;"}

The CUTLASS admission for composition in the restricted case is more restrictive.

Let $\mathbf{S} = (M_{0},M_{1},\ldots,M_{\alpha})$ be a shape tuple, let $M = M_{0} \cdot M_{1} \cdot \ldots \cdot M_{\alpha}$, and let $B = (N):(r)$ be a layout of length 1. Then we say that the pair $\{\mathbf{S},B\}$ is *admissible for composition* (or simply admissible) if:

1.  $M$ is left divisible by $r$. Write $\hat{M} = r \cdot {\hat{M}}^{\prime}$.
2.  With respect to its induced factorization, ${\hat{M}}^{\prime}$ is left divisible by $N$.

Note that the second condition is the left divisibility, instead of the weak left divisibility in Definition 2.12.

For example, suppose $A = (8,6,8):(1,16,108)$ and $B = (8):(4)$. According to Definition 2.12, $A \circ B = (2,4):(4,16)$. However, if we run composition for $A$ and $B$ in CUTLASS, we will encounter an error because CUTLASS requires left divisibility for the second condition.

More specifically, in the [CUTLASS composition layout algebra implementation](https://github.com/NVIDIA/cutlass/blob/v3.5.1/media/docs/cute/02_layout_algebra.md#computing-composition){target="_blank" rel="noopener"}, we have

<figure id="code-1766573911346145" class="highlight c++ hljs">
<div class="highlight-body">
<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td class="gutter"><pre><code>1
2
3
4
5
6
7
8
9
10</code></pre></td>
<td class="code"><pre><code>void shape_div(int* shapeA, int N, int&amp; strideB) {
   for (int i = 0; i &lt; N; ++i) {
      assert(shapeA[i] %   strideB == 0 or
               strideB % shapeA[i] == 0);
      int new_shape  = ceil_div(shapeA[i], strideB);
      int new_stride = ceil_div(strideB, shapeA[i]);
      shapeA[i] = new_shape;
      strideB   = new_stride;
   }
}</code></pre></td>
</tr>
</tbody>
</table>
</div>
<figcaption><div class="level-left">
<span class="fold"><em></em></span>
</div>
<div class="level-right">
<a href="javascript:;" class="copy" data-clipboard-target="#code-1766573911346145 .code" title="Copy"><em></em></a>
</div></figcaption>
</figure>

<figure id="code-1766573911346658" class="highlight c++ hljs">
<div class="highlight-body">
<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td class="gutter"><pre><code>1
2
3
4
5
6
7
8
9
10</code></pre></td>
<td class="code"><pre><code>void shape_mod(int* shapeA, int N, int&amp; shapeB) {
   for (int i = 0; i &lt; N; ++i) {
      assert(shapeA[i] %    shapeB == 0 or
                shapeB % shapeA[i] == 0);
      int new_shapeA =      min(shapeA[i], shapeB);
      int new_shapeB = ceil_div(shapeB, shapeA[i]);
      shapeA[i] = new_shapeA;
      shapeB    = new_shapeB;
   }
}</code></pre></td>
</tr>
</tbody>
</table>
</div>
<figcaption><div class="level-left">
<span class="fold"><em></em></span>
</div>
<div class="level-right">
<a href="javascript:;" class="copy" data-clipboard-target="#code-1766573911346658 .code" title="Copy"><em></em></a>
</div></figcaption>
</figure>

The reason why CUTLASS enforces this is because of the logical division operation. Without this restriction, the logical division operation will not be defined in some cases.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#By-Mode-Composition "By-Mode Composition"){.headerlink}By-Mode Composition {#By-Mode-Composition style="scroll-margin: 1em;"}

In some cases, we would like to perform composition for each mode of the layout separately.

Let the layout $A$ be a concatenation of layouts $A = (A_{0},A_{1},\ldots,A_{\alpha})$, and $B$ be a tile of layouts $B = \langle B_{0},B_{1},\ldots,B_{\alpha}\rangle$. Note that $B$, a tiler, is just a tuple of layouts instead of a concatenated layout. A tiler may consists of one or more than one layouts. Then we define the *by-mode composition* $A \circ B$ to be the layout

$$\begin{array}{r}
{A \circ B:=\left( A_{0} \circ B_{0},A_{1} \circ B_{1},\ldots,A_{\alpha} \circ B_{\alpha} \right)}
\end{array}$$

In some cases, we also have the layout $A$ be a concatenation of layouts $A = (A_{0},A_{1},\ldots,A_{\alpha})$, and $B$ be a tile of layouts $B = \langle B_{0},B_{1},\ldots,B_{\beta}\rangle$, where $\beta \geq \alpha$. In this case, we define the *by-mode composition* $A \circ B$ to be the layout

$$\begin{array}{r}
{A \circ B:=\left( A_{0} \circ B_{0},A_{1} \circ B_{1},\ldots,A_{\alpha} \circ B_{\alpha} \right)}
\end{array}$$

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Non-Integral-Layout-Composition "Non-Integral Layout Composition"){.headerlink}Non-Integral Layout Composition {#Non-Integral-Layout-Composition style="scroll-margin: 1em;"}

Similar to the non-integral layout complementation, the non-integral layout composition can be derived by coalescing the the layout so that the layout becomes integral. All the properties of the integral layout composition derived above can then be used.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Implication-of-Composition "Implication of Composition"){.headerlink}Implication of Composition {#Implication-of-Composition style="scroll-margin: 1em;"}

The composition operation is usually used for selecting a sublayout from a layout. The sublayout can be strided even for each mode if the by-mode composition is performed.

## [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Logical-Division "Logical Division"){.headerlink}Logical Division {#Logical-Division style="scroll-margin: 1em;"}

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-2-22-Logical-Division "Definition 2.22 Logical Division"){.headerlink}Definition 2.22 Logical Division {#Definition-2-22-Logical-Division style="scroll-margin: 1em;"}

Let $A = \mathbf{S}:\mathbf{D}$ and $B$ be layouts, and let $M$ be the size of $A$. Suppose that the pairs $\{ B,M\}$ and $\{\mathbf{S},B\}$ are admissible (for complementation and composition, respectively). Then we define the *logical division* $A/B$ to be the layout

$$\begin{array}{r}
{A/B:=A \circ \left( B,\text{complement}(B,M) \right)}
\end{array}$$

Note that here the conditions of admission for composition follows Definition 2.21 rather than Definition 2.12.

Implicitly Lemma 2.23 is used in Definition 2.22.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Lemma-2-23-Logical-Division-Implication "Lemma 2.23 Logical Division Implication"){.headerlink}Lemma 2.23 Logical Division Implication {#Lemma-2-23-Logical-Division-Implication style="scroll-margin: 1em;"}

Suppose $A = \mathbf{S}:\mathbf{D}$, $M = \text{size}(A)$, and $B$ are as in Definition 2.22. Then $\{\mathbf{S},\left( B,\text{complement}(B,M) \right)\}$ is admissible for composition.

*Proof*

We denote $A = \mathbf{S}:\mathbf{D} = (M_{0},M_{1},\ldots,M_{\alpha}):(d_{0},d_{1},\ldots,d_{\alpha})$, and $B = (N_{0},N_{1},\ldots,N_{\beta}):(r_{0},r_{1},\ldots,r_{\beta})$. Let

$$\begin{array}{r}
{\varphi:\lbrack 0,\beta\rbrack\overset{\cong}{\rightarrow}\lbrack 0,\beta\rbrack}
\end{array}$$

be the [automorphism](https://en.wikipedia.org/wiki/Automorphism){target="_blank" rel="noopener"} such that $B^{\varphi}:=(N_{\varphi(0)},N_{\varphi(1)},\ldots,N_{\varphi(\beta)}):(r_{\varphi(0)},r_{\varphi(1)},\ldots,r_{\varphi(\beta)})$ is sorted.

Then by Definition 2.6, we have

$$\begin{matrix}
B^{\prime} & {= \text{complement}(B,M)} \\
 & {= \left( r_{\varphi(0)},\frac{r_{\varphi(1)}}{N_{\varphi(0)}r_{\varphi(0)}},\frac{r_{\varphi(2)}}{N_{\varphi(1)}r_{\varphi(1)}},\ldots,\frac{r_{\varphi(\beta)}}{N_{\varphi(\beta - 1)}r_{\varphi(\beta - 1)}},\frac{M}{N_{\varphi(\beta)}r_{\varphi(\beta)}} \right):(1,N_{\varphi(0)}r_{\varphi(0)},N_{\varphi(1)}r_{\varphi(1)},\ldots,N_{\varphi(\beta - 1)}r_{\varphi(\beta - 1)},N_{\varphi(\beta)}r_{\varphi(\beta)})}
\end{matrix}$$

Now we denote each mode of $B^{\prime}$ as

$$\begin{matrix}
B_{k}^{\prime} & {= \left\{ \begin{matrix}
{\left( r_{\varphi(0)} \right):(1)} & {\text{if~}k = 0} \\
{\left( \frac{r_{\varphi(k)}}{N_{\varphi(k - 1)}r_{\varphi(k - 1)}} \right):(N_{\varphi(k - 1)}r_{\varphi(k - 1)})} & {\text{if~}1 \leq k \leq \beta} \\
{\left( \frac{M}{N_{\varphi(\beta)}r_{\varphi(\beta)}} \right):(N_{\varphi(\beta)}r_{\varphi(\beta)})} & {\text{if~}k = \beta + 1}
\end{matrix} \right.}
\end{matrix}$$

for $k \in \lbrack 0,\beta + 1\rbrack$.

Because the pair $\{\mathbf{S},B\}$ is admissible for composition, for each mode in $B$, $B_{k} = (N_{k}):(r_{k})$ for $k \in \lbrack 0,\beta\rbrack$, by Definition 2.17, the pair $\{\mathbf{S},B_{k}\}$ is admissible for composition. Therefore, by Definition 2.12, $M$ is left divisible by $r_{k}$ and the quotient $\frac{M}{r_{k}}$ is left divisible (not weakly left divisible) by $N_{k}$ for all $k \in \lbrack 0,\beta\rbrack$.

It is trivial to see $M$ is left divisible by $1$. Let's see if $M$ is also left divisible by $N_{\varphi(k - 1)}r_{\varphi(k - 1)}$ for all $k \in \lbrack 1,\beta + 1\rbrack$.

Suppose $\varphi(k - 1) = h$ and Because $M$ is left divisible by $r_{h}$, we have

$$\begin{matrix}
r_{\varphi(k - 1)} & {= r_{h}} \\
 & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i_{k - 1} - 1} \cdot c_{k - 1}}
\end{matrix}$$

where $c_{k}$ divides $M_{i_{k}}$.

$$\begin{matrix}
N_{\varphi(k - 1)} & {= N_{h}} \\
 & {= \frac{M_{i_{k}}}{c_{k - 1}} \cdot M_{i_{k - 1} + 1} \cdot \ldots \cdot M_{j_{k - 1} - 1} \cdot c_{k - 1}^{\prime}}
\end{matrix}$$

where $c_{k - 1}^{\prime}$ divides $M_{j_{k - 1}}$.

Thus, $M$ is also left divisible by $N_{\varphi(k - 1)}r_{\varphi(k - 1)}$, because

$$\begin{matrix}
{N_{\varphi(k - 1)}r_{\varphi(k - 1)}} & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i_{k - 1} - 1} \cdot c_{k - 1} \cdot \frac{M_{i_{k}}}{c_{k - 1}} \cdot M_{i_{k} + 1} \cdot \ldots \cdot M_{j_{k - 1} - 1} \cdot c_{k - 1}^{\prime}} \\
 & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{j_{k - 1} - 1} \cdot c_{k - 1}^{\prime}}
\end{matrix}$$

where $c_{k - 1}^{\prime}$ divides $M_{j_{k - 1}}$.

Next, we will have to show $M$ is left divisible by $\frac{r_{\varphi(k)}}{N_{\varphi(k - 1)}r_{\varphi(k - 1)}}$ for all $k \in \lbrack 1,\beta\rbrack$.

$$\begin{matrix}
r_{\varphi(k)} & {= M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i_{k} - 1} \cdot c_{k}}
\end{matrix}$$

Because $r_{\varphi(k)} \geq N_{\varphi(k - 1)}r_{\varphi(k - 1)}$, we must have $i_{k} \geq j_{k - 1}$. Thus,

$$\begin{matrix}
\frac{r_{\varphi(k)}}{N_{\varphi(k - 1)}r_{\varphi(k - 1)}} & {= \frac{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{i_{k} - 1} \cdot c_{k}}{M_{0} \cdot M_{1} \cdot \ldots \cdot M_{j_{k - 1} - 1} \cdot c_{k - 1}^{\prime}}} \\
 & {= \frac{M_{j_{k - 1}} \cdot M_{j_{k - 1} + 1} \cdot \ldots \cdot M_{i_{k} - 1} \cdot c_{k}}{c_{k - 1}^{\prime}}} \\
 & {= \frac{M_{j_{k - 1}}}{c_{k - 1}^{\prime}} \cdot M_{j_{k - 1} + 1} \cdot \ldots \cdot M_{i_{k} - 1} \cdot c_{k}}
\end{matrix}$$

Thus, $M$ is left divisible by $\frac{r_{\varphi(k)}}{N_{\varphi(k - 1)}r_{\varphi(k - 1)}}$ for all $k \in \lbrack 1,\beta\rbrack$.

It is trivial to see $M$ is left divisible by $\frac{M}{N_{\varphi(\beta)}r_{\varphi(\beta)}}$.

Therefore, the pair $\{\mathbf{S},B_{k}^{\prime}\}$ is admissible for composition for all $k \in \lbrack 0,\beta + 1\rbrack$.

By Definition 2.17, in order to show $\{\mathbf{S},(B,\text{complement}(B,M))\}$ is admissible for composition, we also have to show the interval of definition for the pairs $\{\mathbf{S},B_{k}\}_{0 \leq k \leq \beta}$ and $\{\mathbf{S},B_{k}^{\prime}\}_{0 \leq k \leq \beta + 1}$ are disjoint.

By Proposition 2.7, the concatenated layout $(B,\text{complement}(B,M))$ is automatically satisfied with the disjoint argument.

Therefore, $\{\mathbf{S},(B,\text{complement}(B,M))\}$ is admissible for composition.

This concludes the proof. 

Note that in Definition 2.22, if the conditions of admission for composition follows Definition 2.12, our proof above will not be valid. That's why CUTLASS enforces the conditions of admission for composition follows Definition 2.21.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#By-Mode-Logical-Division "By-Mode Logical Division"){.headerlink}By-Mode Logical Division {#By-Mode-Logical-Division style="scroll-margin: 1em;"}

In some cases, we would like to perform logical division for each mode of the layout separately.

Let the layout $A$ be a concatenation of layouts $A = (A_{0},A_{1},\ldots,A_{\alpha})$, and $B$ be a tile of layouts $B = \langle B_{0},B_{1},\ldots,B_{\alpha}\rangle$. Note that $B$ is just a tuple of layouts instead of a concatenated layout. Then we define the *by-mode logical division* $A/B$ to be the layout

$$\begin{matrix}
{A/B} & {:=\left( A_{0}/B_{0},A_{1}/B_{1},\ldots,A_{\alpha}/B_{\alpha} \right)} \\
 & {= \left( A_{0} \circ \left( B_{0},\text{complement}(B_{0},M) \right),A_{1} \circ \left( B_{1},\text{complement}(B_{1},M) \right),\ldots,A_{\alpha} \circ \left( B_{\alpha},\text{complement}(B_{\alpha},M) \right) \right)} \\
 & {= \left( \left( A_{0} \circ B_{0},A_{0} \circ \text{complement}(B_{0},M) \right),\left( A_{1} \circ B_{1},A_{1} \circ \text{complement}(B_{1},M) \right),\ldots,\left( A_{\alpha} \circ B_{\alpha},A_{\alpha} \circ \text{complement}(B_{\alpha},M) \right) \right)}
\end{matrix}$$

In some cases, we also have $A$ be a concatenation of layouts $A = (A_{0},A_{1},\ldots,A_{\alpha})$, and $B$ be a tile of layouts $B = \langle B_{0},B_{1},\ldots,B_{\beta}\rangle$, where $\alpha \geq \beta$. In this case, we define the *by-mode logical division* $A/B$ to be the layout

$$\begin{matrix}
{A/B} & {:=\left( A_{0}/B_{0},A_{1}/B_{1},\ldots,A_{\beta}/B_{\beta},A_{\beta + 1},\ldots,A_{\alpha} \right)} \\
 & {= \left( A_{0} \circ \left( B_{0},\text{complement}(B_{0},M) \right),A_{1} \circ \left( B_{1},\text{complement}(B_{1},M) \right),\ldots,A_{\beta} \circ \left( B_{\beta},\text{complement}(B_{\beta},M) \right),A_{\beta + 1},\ldots,A_{\alpha} \right)} \\
 & {= \left( \left( A_{0} \circ B_{0},A_{0} \circ \text{complement}(B_{0},M) \right),\left( A_{1} \circ B_{1},A_{1} \circ \text{complement}(B_{1},M) \right),\ldots,\left( A_{\beta} \circ B_{\beta},A_{\beta} \circ \text{complement}(B_{\beta},M) \right),A_{\beta + 1},\ldots,A_{\alpha} \right)}
\end{matrix}$$

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Logical-Division-Variants "Logical Division Variants"){.headerlink}Logical Division Variants {#Logical-Division-Variants style="scroll-margin: 1em;"}

In by-mode logical division, it is inconvenient to select a multi-dimensional tile. Given logical division $A/B$,

$$\begin{matrix}
{A/B} & {= \left( \left( A_{0} \circ B_{0},A_{0} \circ \text{complement}(B_{0},M) \right),\left( A_{1} \circ B_{1},A_{1} \circ \text{complement}(B_{1},M) \right),\ldots,\left( A_{\beta} \circ B_{\beta},A_{\beta} \circ \text{complement}(B_{\beta},M) \right),A_{\beta + 1},\ldots,A_{\alpha} \right)}
\end{matrix}$$

We would like to iterate through each multi-dimensional sublayout $\left( A_{0} \circ B_{0},A_{1} \circ B_{1},\ldots,A_{\beta} \circ B_{\beta} \right)$ from $A/B$ by indexing.

So zipped division is introduced by rearranging the layout $A/B$ to

$$\begin{matrix}
{\text{zipped\_division}(A,B)} & {= \left( \left( A_{0} \circ B_{0},A_{1} \circ B_{1},\ldots,A_{\beta} \circ B_{\beta} \right),\left( A_{0} \circ \text{complement}(B_{0},M),A_{1} \circ \text{complement}(B_{1},M),\ldots,A_{\beta} \circ \text{complement}(B_{\beta},M) \right),A_{\beta + 1},\ldots,A_{\alpha} \right)} \\
 & {= \left( A \circ B,\left( A_{0} \circ \text{complement}(B_{0},M),A_{1} \circ \text{complement}(B_{1},M),\ldots,A_{\beta} \circ \text{complement}(B_{\beta},M) \right),A_{\beta + 1},\ldots,A_{\alpha} \right)}
\end{matrix}$$

In this way, we can iterate through each multi-dimensional sublayout $A \circ B = \left( A_{0} \circ B_{0},A_{1} \circ B_{1},\ldots,A_{\beta} \circ B_{\beta} \right)$ from $\text{zipped\_division}(A,B)$ by indexing on the second mode. Note because $A_{0} \circ B_{0}$ has the same domain as $B_{0}$, $A_{1} \circ B_{1}$ has the same domain as $B_{1}$, and so on, the domain or shape of $A \circ B$ becomes very predictable, which is $(\text{shape}(B_{0}),\text{shape}(B_{1}),\ldots,\text{shape}(B_{\beta}))$. The shape of $\left( A_{0} \circ \text{complement}(B_{0},M),A_{1} \circ \text{complement}(B_{1},M),\ldots,A_{\beta} \circ \text{complement}(B_{\beta},M) \right)$ is less predictable, but its 1D size is predictable, which is $\left( \frac{M}{\text{size}(B_{0})},\frac{M}{\text{size}(B_{1})},\ldots,\frac{M}{\text{size}(B_{\beta})} \right)$.

CuTe implemented four logical division variants, including logical divide, zipped divide, tiled divide, and flat divide. Assuming the tiler is a tuple of two layouts, the logical division variants are defined as follows:

<figure id="code-1766573911346681" class="highlight plaintext hljs">
<div class="highlight-body">
<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td class="gutter"><pre><code>1
2
3
4
5
6
7</code></pre></td>
<td class="code"><pre><code>Layout Shape : (M, N, L, ...)
Tiler Shape  : &lt;TileM, TileN&gt;

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)</code></pre></td>
</tr>
</tbody>
</table>
</div>
<figcaption><div class="level-left">
<span class="fold"><em></em></span>
</div>
<div class="level-right">
<a href="javascript:;" class="copy" data-clipboard-target="#code-1766573911346681 .code" title="Copy"><em></em></a>
</div></figcaption>
</figure>

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Implication-of-Logical-Division "Implication of Logical Division"){.headerlink}Implication of Logical Division {#Implication-of-Logical-Division style="scroll-margin: 1em;"}

Because we have previously proved that $\left( B,\text{complement}(B,M) \right)$ is a layout that is a bijection $\lbrack 0,M) \cong \lbrack 0,M)$, the logical division $A/B:=A \circ \left( B,\text{complement}(B,M) \right)$, where $M$ is the size of $A$, is a layout that has the same domain as $A$ and consequently also the same codomain as $A$. This means that the way the original layout $A$ maps from the coordinates $\lbrack 0,M)$ to the integers is scrambled or permutated to a new layout, i.e., the logical division $A/B$.

Because of the definition of composition,

$$\begin{matrix}
{A/B} & {:=A \circ \left( B,\text{complement}(B,M) \right)} \\
 & {= \left( A \circ B,A \circ \text{complement}(B,M) \right)}
\end{matrix}$$

$A \circ B$ is the layout that selects a sublayout, i.e., a tile, from $A$ based on the layout $B$, and $A \circ \text{complement}(B,M)$ that repeats the $A \circ B$ layout to fill the domain and codomain of $A$.

Logical division informs us how to extract a tile using the tiler $B$ from a layout $A$, and how to repeat the tile to fill the domain and codomain of $A$, or how to select a tile from the consequent repeated layout using indexing.

## [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Logical-Product "Logical Product"){.headerlink}Logical Product {#Logical-Product style="scroll-margin: 1em;"}

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-4-1-Logical-Product "Definition 4.1 Logical Product"){.headerlink}Definition 4.1 Logical Product {#Definition-4-1-Logical-Product style="scroll-margin: 1em;"}

Given a tiler and a layout, we could compute how the repeat layout that repeats the tiler to fill the domain and codomain of the layout using logical division. Similarly, given a tiler and a repeat layout, we could compute the resulting layout that is a tile of the repeat layout using logical product.

Let $A$ and $B$ be layouts, $M = \text{size}(A)\text{cosize}(B)$. Suppose that the pairs $\{ A,M\}$ and $\{\text{complement}(A,M),B\}$ are admissible (for complementation and composition, respectively). Then we define the *logical product* $A \times B$ to be the layout

$$\begin{array}{r}
{A \times B:=\left( A,\text{complement}(A,M) \circ B \right)}
\end{array}$$

The complementation of a layout $A$ finds a complement layout $\text{complement}(A,M)$ with a positive integer $M$ so that when the two layouts are concatenated, such as $\left( A,\text{complement}(A,M) \right)$, the new layout is a bijection $\lbrack 0,M) \cong \lbrack 0,M)$. However, the codomain of the resulting layout we want to create might not need to be $\lbrack 0,M)$. The layout $B$ can have $\text{cosize}(B)$ that is much larger than $\text{size}(B)$. Consequently, after repeating the layout $A$ using $\text{complement}(A,M) \circ B$, the resulting layout $A \times B$ will fill up to the codomain $M$ but potentially strided. In addition, even if $\text{size}(B) = \text{cosize}(B)$, we could permute $\text{complement}(A,M)$ by using the composition operation, because there can be multiple layouts whose cosize is $\text{cosize}(B)$.

Because $\text{complement}(A,M) \circ B$ is a sublayout of $\text{complement}(A,M)$, the layout concatenation $\left( A,\text{complement}(A,M) \circ B \right)$ remains a valid layout.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#By-Mode-Logical-Product "By-Mode Logical Product"){.headerlink}By-Mode Logical Product {#By-Mode-Logical-Product style="scroll-margin: 1em;"}

In some cases, we would like to perform logical division for each mode of the layout separately. For example, if we would like to tile a 2D layout with a 2D tiler, we would like to perform logical product for each mode of the layout separately.

Let the layout $A$ be a concatenation of layouts $A = (A_{0},A_{1},\ldots,A_{\alpha})$, and $B$ be a tile of layouts $B = \langle B_{0},B_{1},\ldots,B_{\alpha}\rangle$. Note that $B$ is just a tuple of layouts instead of a concatenated layout. Then we define the *by-mode logical product* $A \times B$ to be the layout

$$\begin{matrix}
{A \times B} & {:=\left( A_{0} \times B_{0},A_{1} \times B_{1},\ldots,A_{\alpha} \times B_{\alpha} \right)} \\
 & {= \left( A_{0},\text{complement}(A_{0},M) \circ B_{0},A_{1},\text{complement}(A_{1},M) \circ B_{1},\ldots,A_{\alpha},\text{complement}(A_{\alpha},M) \circ B_{\alpha} \right)} \\
 & {= \left( \left( A_{0},\text{complement}(A_{0},M) \circ B_{0} \right),\left( A_{1},\text{complement}(A_{1},M) \circ B_{1} \right),\ldots,\left( A_{\alpha},\text{complement}(A_{\alpha},M) \circ B_{\alpha} \right) \right)}
\end{matrix}$$

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Logical-Product-Variants "Logical Product Variants"){.headerlink}Logical Product Variants {#Logical-Product-Variants style="scroll-margin: 1em;"}

Similar to logical division, there are also variants of logical product, including logical product, zipped product, tiled product, and flat product, for the convenience of tile selection and iteration. Assuming the tiler is a tuple of two layouts, the logical product variants are defined as follows:

<figure id="code-1766573911346923" class="highlight plaintext hljs">
<div class="highlight-body">
<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td class="gutter"><pre><code>1
2
3
4
5
6
7</code></pre></td>
<td class="code"><pre><code>Layout Shape : (M, N, L, ...)
Tiler Shape  : &lt;TileM, TileN&gt;

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)</code></pre></td>
</tr>
</tbody>
</table>
</div>
<figcaption><div class="level-left">
<span class="fold"><em></em></span>
</div>
<div class="level-right">
<a href="javascript:;" class="copy" data-clipboard-target="#code-1766573911346923 .code" title="Copy"><em></em></a>
</div></figcaption>
</figure>

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Implication-of-Logical-Product "Implication of Logical Product"){.headerlink}Implication of Logical Product {#Implication-of-Logical-Product style="scroll-margin: 1em;"}

Similar to logical division, the logical product $A \times B$ informs the us what the layout is after repeating the original layout using the tiler.

## [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Permutation-Expressible-As-Layout-Functions "Permutation Expressible As Layout Functions"){.headerlink}Permutation Expressible As Layout Functions {#Permutation-Expressible-As-Layout-Functions style="scroll-margin: 1em;"}

This section explains how to retrieve all permutations that are expressible as layout functions in a structured way. This is important because some permutation algebra used in CUTLASS and CuTe, such as swizzle, does not seem to be expressed as layout function. The basic language of [category theory](https://en.wikipedia.org/wiki/Category_theory){target="_blank" rel="noopener"} is used to describe the process.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-3-1-Ordered-Factorization "Definition 3.1 Ordered Factorization"){.headerlink}Definition 3.1 Ordered Factorization {#Definition-3-1-Ordered-Factorization style="scroll-margin: 1em;"}

We define the set $\text{ob}(\textbf{Fact})$ of *ordered factorizations* to consists of all expressions $\lbrack p_{1}\ldots p_{k}\rbrack$ where $k \geq 0$ and the $p_{i}$ are primes (not necessarily distinct). The case $k = 0$ corresponds to the empty factorization, which we denote as $\lbrack\ \rbrack$.

For example, the set $\text{ob}(\textbf{Fact})$ includes expressions such as $\lbrack\ \rbrack$, $\lbrack 2\rbrack$, $\lbrack 3\rbrack$, $\lbrack 22\rbrack$, $\lbrack 23\rbrack$, $\lbrack 32\rbrack$, $\lbrack 232\rbrack$, etc.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Notation-3-3 "Notation 3.3"){.headerlink}Notation 3.3 {#Notation-3-3 style="scroll-margin: 1em;"}

Let $\underset{―}{k}$ denote the set $\{ 1,2,\ldots,k\}$ consisting of $k$ elements. (If $k = 0$, then $\underset{―}{0} = \varnothing$ is the empty set.)

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-3-4-Category-of-Ordered-Factorizations "Definition 3.4 Category of Ordered Factorizations"){.headerlink}Definition 3.4 Category of Ordered Factorizations {#Definition-3-4-Category-of-Ordered-Factorizations style="scroll-margin: 1em;"}

We define the category $\textbf{Fact}$ of ordered factorizations as follows:

1.  $\text{ob}(\textbf{Fact})$ is the set of objects of $\textbf{Fact}$.
2.  For every expression $E = \lbrack p_{1}\ldots p_{k}\rbrack$ in $\text{ob}(\textbf{Fact})$ and every morphism of finite sets $\alpha:\underset{―}{n} \rightarrow \underset{―}{k}$, we have a morphism

$$\begin{array}{r}
{E^{\alpha} = \lbrack p_{\alpha(1)}\ldots p_{\alpha(n)}\rbrack\overset{\alpha_{E}}{\rightarrow}E = \lbrack p_{1}\ldots p_{k}\rbrack}
\end{array}$$

in $\textbf{Fact}$. This defines the set of all morphisms with [codomain](https://en.wikipedia.org/wiki/Codomain){target="_blank" rel="noopener"} $E$, and ranging over all $E$ thus defines the set of all morphisms in $\textbf{Fact}$.

3.  The composition of morphisms is defined as follows. Suppose we have morphisms of finite sets $\alpha:\underset{―}{n} \rightarrow \underset{―}{k}$ and $\beta:\underset{―}{m} \rightarrow \underset{―}{n}$, and expressions $E = \lbrack p_{1}\ldots p_{k}\rbrack$. Write

$$\begin{array}{r}
{E^{\alpha} = \lbrack p_{\alpha(1)}\ldots p_{\alpha(n)}\rbrack = \lbrack q_{1}\ldots q_{n}\rbrack}
\end{array}$$

Let $\gamma = \alpha \circ \beta:\underset{―}{m} \rightarrow \underset{―}{k}$. Then the composition of morphisms

$$\begin{array}{r}
{\alpha_{E}:E^{\alpha} = \lbrack p_{\alpha(1)}\ldots p_{\alpha(n)}\rbrack\overset{}{\rightarrow}E = \lbrack p_{1}\ldots p_{k}\rbrack}
\end{array}$$

$$\begin{array}{r}
{\beta_{E^{\alpha}}:E^{\beta} = \lbrack q_{\beta(1)}\ldots q_{\beta(m)}\rbrack\overset{}{\rightarrow}E^{\alpha} = \lbrack q_{1}\ldots q_{n}\rbrack}
\end{array}$$

is given by $\gamma_{E}:E^{\gamma}\overset{}{\rightarrow}E$, where we used that $\lbrack q_{\beta(1)}\ldots q_{\beta(m)}\rbrack = \lbrack p_{\gamma(1)}\ldots p_{\gamma(m)}\rbrack$.

It's easy to check that the composition of morphisms in $\textbf{Fact}$ is associative and has identities, which are the two axioms that composition in a category must satisfy, so Definition 3.4 really does define a category.

To see why the composition of morphisms is associative, suppose we have morphisms of finite sets $\alpha:\underset{―}{n} \rightarrow \underset{―}{k}$, $\beta:\underset{―}{m} \rightarrow \underset{―}{n}$, and $\gamma:\underset{―}{l} \rightarrow \underset{―}{m}$. Then we have

$$\begin{array}{r}
{\alpha \circ (\beta \circ \gamma) = (\alpha \circ \beta) \circ \gamma}
\end{array}$$

To see why the composition of morphisms has identities, suppose for every $n$ we have a morphism of finite sets $\text{id}_{\underset{―}{n}}:\underset{―}{n} \rightarrow \underset{―}{n}$, such that $\text{id}_{\underset{―}{n}}(i) = i$ for all $i \in \underset{―}{n}$. Then we have

$$\begin{array}{r}
{E^{\text{id}_{\underset{―}{n}}} = \lbrack p_{\text{id}_{\underset{―}{n}}(1)}\ldots p_{\text{id}_{\underset{―}{n}}(n)}\rbrack\overset{}{\rightarrow}E = \lbrack p_{1}\ldots p_{n}\rbrack}
\end{array}$$

For every morphism $\alpha:\underset{―}{n} \rightarrow \underset{―}{k}$, we have

$$\begin{array}{r}
{\alpha \circ \text{id}_{\underset{―}{n}} = \text{id}_{\underset{―}{k}} \circ \alpha = \alpha}
\end{array}$$

Therefore, $\text{id}_{\underset{―}{n}}$ is the identity morphism for $\underset{―}{n}$.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Notation-3-5 "Notation 3.5"){.headerlink}Notation 3.5 {#Notation-3-5 style="scroll-margin: 1em;"}

Let $\Sigma_{k}$ denote the [symmetric group](https://en.wikipedia.org/wiki/Symmetric_group){target="_blank" rel="noopener"} on $k$ letters. Given an element $\varphi \in \Sigma_{k}$, we also denote the associated automorphism of $\underset{―}{k}$ by $\varphi$.

In mathematics, a [group](https://en.wikipedia.org/wiki/Group_(mathematics)){target="_blank" rel="noopener"} is a set with an operation that associates an element of the set to every pair of elements of the set (as does every binary operation) and satisfies the following constraints: the operation is associative, it has an identity element, and every element of the set has an inverse element.

In this sense, the symmetric group is a set of all permutations of a set of $k$ elements with an operation of composition of permutations (applying one permutation after another).

Suppose $E = \lbrack 222\rbrack$. Then every permutation $\varphi \in \Sigma_{3}$ defines an automorphism $E^{\varphi} = E\overset{}{\rightarrow}E$ in $\textbf{Fact}$.

Suppose $E = \lbrack 232\rbrack$. Then the transposition $\sigma = (13) \in \Sigma_{3}$ defines an automorphism $E^{\sigma} = E\overset{}{\rightarrow}E$ in $\textbf{Fact}$. On the other hand, the transposition $\tau = (12) \in \Sigma_{3}$ defines a morphism $E^{\tau} = \lbrack 322\rbrack\overset{}{\rightarrow}E = \lbrack 232\rbrack$ in $\textbf{Fact}$.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Remark-3-7 "Remark 3.7"){.headerlink}Remark 3.7 {#Remark-3-7 style="scroll-margin: 1em;"}

Let $\textbf{FinSet}$ denote the category of finite sets (or rather a skeleton, with objects given by the sets $\underset{―}{n}$ for $n \geq 0$). Given an object $\underset{―}{k} \in \textbf{FinSet}$, let $\textbf{FinSet}^{/\underset{―}{k}}$ denote the [overcategory](https://en.wikipedia.org/wiki/Overcategory){target="_blank" rel="noopener"}, whose objects are morphisms $\lbrack\alpha:\underset{―}{n} \rightarrow \underset{―}{k}\rbrack$ and whose morphisms are commuting triangles. Recall that this category has a [final object](https://en.wikipedia.org/wiki/Initial_and_terminal_objects){target="_blank" rel="noopener"} given by the identity morphism $\lbrack\text{id}_{\underset{―}{k}}\rbrack$.

Then for every expression $E = \lbrack p_{1}\ldots p_{k}\rbrack$ of length $k$, we have a [functor](https://en.wikipedia.org/wiki/Functor){target="_blank" rel="noopener"}

$$\begin{array}{r}
{F_{E}:\textbf{FinSet}^{/\underset{―}{k}} \rightarrow \textbf{Fact}}
\end{array}$$

that sends the object $\lbrack\alpha:\underset{―}{n} \rightarrow \underset{―}{k}\rbrack$ to the expression $E^{\alpha}$ and the unique morphism $\lbrack\alpha\rbrack\overset{}{\rightarrow}\lbrack\text{id}_{\underset{―}{k}}\rbrack$ to $\alpha_{E}:E^{\alpha}\overset{}{\rightarrow}E$. This functor has every morphism in $\textbf{Fact}$ with codomain $E$ in its image.

Suppose we have an object of morphism $\lbrack\alpha:\underset{―}{n} \rightarrow \underset{―}{k}\rbrack$ and another object of morphism $\lbrack\beta:\underset{―}{m} \rightarrow \underset{―}{k}\rbrack$ in $\textbf{FinSet}^{/\underset{―}{k}}$. Then the morphism of the overcategory is a commuting triangle, whose remaining morphism is $\lbrack\gamma:\underset{―}{m} \rightarrow \underset{―}{n}\rbrack$, that maps $\lbrack\alpha:\underset{―}{n} \rightarrow \underset{―}{k}\rbrack$ to $\lbrack\beta:\underset{―}{m} \rightarrow \underset{―}{k}\rbrack$.

The identity morphism $\lbrack\text{id}_{\underset{―}{k}}\rbrack$ is the final object of $\textbf{FinSet}^{/\underset{―}{k}}$ because every other object of morphism in $\textbf{FinSet}^{/\underset{―}{k}}$ has a unique morphism to $\lbrack\text{id}_{\underset{―}{k}}\rbrack$.

In this case, given an object of morphism $\lbrack\alpha:\underset{―}{n} \rightarrow \underset{―}{k}\rbrack$, the commuting triangle has a remaining morphism $\lbrack\gamma:\underset{―}{k} \rightarrow \underset{―}{n}\rbrack$, and we will have to show that this remaining morphism in the commuting triangle is unique.

Given $\underset{―}{k} = \{ 1,2,\ldots,k\}$, the identity morphism maps $i \in \underset{―}{k}$ to $i \in \underset{―}{k}$. Given a morphism $\lbrack\alpha:\underset{―}{n} \rightarrow \underset{―}{k}\rbrack$, in which without loss of generality we assume $\alpha(i) = i$ for all $i \in \lbrack 1,k\rbrack$, the remaining morphism $\lbrack\gamma:\underset{―}{k} \rightarrow \underset{―}{n}\rbrack$ must have $i = \alpha(i)$ for all $i \in \lbrack 1,k\rbrack$, so that we have a commuting triangle that $i \in \underset{―}{k}\overset{\gamma}{\rightarrow}\overset{\alpha}{\rightarrow}i \in \underset{―}{k}$. Otherwise, for the remaining morphism $\lbrack\gamma:\underset{―}{k} \rightarrow \underset{―}{n}\rbrack$, if there exists an $i \in \underset{―}{k}$ such that $i \neq \alpha(i)$, then the commuting triangle will not be valid because $i \in \underset{―}{k}\overset{\gamma}{\rightarrow}\overset{\alpha}{\rightarrow}j \in \underset{―}{k}$ and $i \neq j$. Therefore, the morphism of commuting triangle is unique.

By [definition](https://en.wikipedia.org/wiki/Functor#Definition){target="_blank" rel="noopener"}, let $C$ and $D$ be categories. A functor $F$ from $C$ to $D$ is a mapping that

-   associates each object $X$ in $C$ to an object $F(X)$ in $D$,
-   associates each morphism $f:X \rightarrow Y$ in $C$ to a morphism $F(f):F(X) \rightarrow F(Y)$ in $D$ such that
    -   $F(\text{id}_{X}) = \text{id}_{F(X)}$ for every object $X$ in $C$, and
    -   $F(g \circ f) = F(g) \circ F(f)$ for all morphisms $f:X \rightarrow Y$ and $g:Y \rightarrow Z$ in $C$.

In the category $\textbf{FinSet}^{/\underset{―}{k}}$, we have $X = \lbrack\alpha:\underset{―}{n} \rightarrow \underset{―}{k}\rbrack$ and $Y = \lbrack\gamma:\underset{―}{m} \rightarrow \underset{―}{k}\rbrack$, $Z = \lbrack\text{id}_{\underset{―}{k}}\rbrack$, and the morphisms are commuting triangles $f = \lbrack\alpha\rbrack\overset{}{\rightarrow}\lbrack\gamma\rbrack$, $g = \lbrack\gamma\rbrack\overset{}{\rightarrow}\lbrack\text{id}_{\underset{―}{k}}\rbrack$, and $g \circ f = \lbrack\alpha\rbrack\overset{}{\rightarrow}\lbrack\text{id}_{\underset{―}{k}}\rbrack$.

In the category $\textbf{Fact}$, by the functor $F_{E}$, we have $F_{E}(X) = E^{\alpha} = \lbrack p_{\alpha(1)}\ldots p_{\alpha(n)}\rbrack$, $F_{E}(Y) = E^{\gamma} = \lbrack p_{\gamma(1)}\ldots p_{\gamma(m)}\rbrack$, $F_{E}(Z) = E = \lbrack p_{1}\ldots p_{k}\rbrack$, and the morphisms are $F_{E}(f) = \alpha_{E}:E^{\alpha}\overset{}{\rightarrow}E^{\gamma}$, $F_{E}(g) = \gamma_{E}:E^{\gamma}\overset{}{\rightarrow}E$, and $F_{E}(g) \circ F_{E}(f) = \alpha_{E}:E^{\alpha}\overset{}{\rightarrow}E$.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Remark-3-8 "Remark 3.8"){.headerlink}Remark 3.8 {#Remark-3-8 style="scroll-margin: 1em;"}

In fact, we can identify $\textbf{Fact}$ itself as a certain overcategory (or rather, a full [subcategory](https://en.wikipedia.org/wiki/Subcategory){target="_blank" rel="noopener"} thereof). Namely, let $\mathcal{P}$ denote the the infinite set of primes $\{ 2,3,5\ldots\}$, let $\textbf{Set}$ be the category of sets, and let $\textbf{FinSet}^{/\mathcal{P}}$ be the full subcategory of $\textbf{Set}^{/\mathcal{P}}$ on those morphisms $X\overset{}{\rightarrow}\mathcal{P}$ where $X$ is a finite set. Then we have an equivalence of categories

$$\begin{array}{r}
{\textbf{Fact} \simeq \textbf{FinSet}^{/\mathcal{P}}}
\end{array}$$

that sends an expression $E = \lbrack p_{1}\ldots p_{k}\rbrack$ to the morphism $E_{\bullet}:k\overset{}{\rightarrow}\mathcal{P}$ given by $i\mapsto p_{i}$.

Under this equivalence, the functor $F_{E}$ of Remark 3.7 identifies with the functor

$$\begin{array}{r}
{\textbf{FinSet}^{/k} \simeq \left( \textbf{FinSet}^{/\mathcal{P}} \right)^{/E_{\bullet}}\overset{}{\rightarrow}\textbf{FinSet}^{/\mathcal{P}}}
\end{array}$$

that forgets the map to $E_{\bullet}$.

To understand this, let's consider the following example.

Suppose we have an object $E = \lbrack 232\rbrack$ from $\textbf{Fact}$. Then we have a morphism $E_{\bullet}:\underset{―}{3}\overset{}{\rightarrow}\mathcal{P}$ given by

$$\begin{matrix}
{i = 1} & {\mapsto p_{1} = 2} \\
{i = 2} & {\mapsto p_{2} = 3} \\
{i = 3} & {\mapsto p_{3} = 2}
\end{matrix}$$

Every object of morphisms in $\textbf{FinSet}^{/\mathcal{P}}$ can be mapped from an object from $\textbf{Fact}$ and thus we have an equivalence of categories $\textbf{Fact} \simeq \textbf{FinSet}^{/\mathcal{P}}$.

Because of the functor $F_{E}$ of Remark 3.7, with the equivalence above, we have

$$\begin{array}{r}
{\textbf{FinSet}^{/\underset{―}{k}} \rightarrow \textbf{FinSet}^{/\mathcal{P}}}
\end{array}$$

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-3-9 "Definition 3.9"){.headerlink}Definition 3.9 {#Definition-3-9 style="scroll-margin: 1em;"}

Suppose $E = \lbrack p_{1}\ldots p_{k}\rbrack$ and $\alpha:\underset{―}{n} \rightarrow \underset{―}{k}$. We define a layout $L_{(E,\alpha)}$ as follows:

1.  Its shape tuple is $(p_{\alpha(1)},p_{\alpha(2)},\ldots,p_{\alpha(n)})$.
2.  Its stride tuple is $(d_{1},d_{2},\ldots,d_{n})$, where $d_{i} = \prod\limits_{j < \alpha(i)}p_{j}$.

We also let $f_{(E,\alpha)}$ denote the associated layout function.

Suppose $E = \lbrack 23\rbrack$ and $\varphi = (12) \in \Sigma_{2}$ is the nontrivial transposition. Then $L_{(E,\varphi)} = (3,2):(2,1)$.

Suppose $E = \lbrack 23\rbrack$, $\varphi(1) = 2$, $\varphi(2) = 1$, $\varphi(3) = 2$. Then $L_{(E,\varphi)} = (3,2,3):(2,1,2)$. This layout seems strange because its layout function is not an injection mapping. However, it is still a valid layout by Definition 2.2.

Suppose $E = \lbrack 222\rbrack$ and $\varphi = (231) \in \Sigma_{3}$, so $\varphi$ is a cycle of order $3$ with $\varphi(1) = 2$, $\varphi(2) = 3$, and $\varphi(3) = 1$. Then $L_{(E,\varphi)} = (2,2,2):(2,4,1)$.

We could now see why $p_{i}$ is prime number. It's used for constructing the stride tuple of any kind for the layout, because any natural number can be uniquely factored into a product of prime numbers.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Remark-3-11 "Remark 3.11"){.headerlink}Remark 3.11 {#Remark-3-11 style="scroll-margin: 1em;"}

Let $E = \lbrack p_{1}\ldots p_{k}\rbrack$ and $\alpha:\underset{―}{n} \rightarrow \underset{―}{k}$. Let $N = p_{1} \cdot p_{2} \cdot \ldots \cdot p_{k}$ and $N^{\alpha} = p_{\alpha(1)} \cdot p_{\alpha(2)} \cdot \ldots \cdot p_{\alpha(n)}$. In what follows, consider the canonical isomorphisms

$$\begin{matrix}
{\lbrack 0,N)} & {\cong \lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})} \\
{\lbrack 0,N^{\alpha})} & {\cong \lbrack 0,p_{\alpha(1)}) \times \lbrack 0,p_{\alpha(2)}) \times \ldots \times \lbrack 0,p_{\alpha(n)})}
\end{matrix}$$

Then the associated layout function $f_{(E,\alpha)}:\lbrack 0,N^{\alpha}) \rightarrow \lbrack 0,N) \subset \mathbb{N}$ can be described as the [multilinear function](https://en.wikipedia.org/wiki/Multilinear_map){target="_blank" rel="noopener"}

$$\begin{array}{r}
{\lbrack 0,p_{\alpha(1)}) \times \lbrack 0,p_{\alpha(2)}) \times \ldots \times \lbrack 0,p_{\alpha(n)})\overset{}{\rightarrow}\lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})}
\end{array}$$

that sends the basis vector $\delta_{i}$ of one vector space to the basis vector $\beta_{\alpha(i)}$ of the other for $i \in \lbrack 1,n\rbrack$.

In particular, if $\alpha$ is itself a bijection, then $f_{(E,\alpha)}$ restricts to an automorphism of $\lbrack 0,N)$.

*Proof*

We noticed that $f_{(E,\alpha)}:\lbrack 0,N^{\alpha}) \rightarrow \lbrack 0,N) \subset \mathbb{N}$. The domain of $f_{(E,\alpha)}$ is $\lbrack 0,N^{\alpha})$ and it is obvious. The codomain of $f_{(E,\alpha)}$ is $\lbrack 0,N) \subset \mathbb{N}$, however, is less obvious.

$$\begin{array}{r}
{f_{(E,\alpha)} = x_{\alpha_{(1)}}d_{1} + x_{\alpha_{(2)}}d_{2} + \ldots + x_{\alpha_{(n)}}d_{n}}
\end{array}$$

where $x_{\alpha_{(i)}} \in \lbrack 0,p_{\alpha(i)})$ and $d_{i} = \prod\limits_{j < \alpha(i)}p_{j}$.

So we have

$$\begin{matrix}
{\max\left( f_{(E,\alpha)} \right)} & {= (p_{\alpha(1)} - 1)d_{1} + (p_{\alpha(2)} - 1)d_{2} + \ldots + (p_{\alpha(n)} - 1)d_{n}} \\
 & {= (p_{\alpha(1)} - 1)\prod\limits_{j < \alpha(1)}p_{j} + (p_{\alpha(2)} - 1)\prod\limits_{j < \alpha(2)}p_{j} + \ldots + (p_{\alpha(n)} - 1)\prod\limits_{j < \alpha(n)}p_{j}}
\end{matrix}$$

Without losing generality, we assume $p_{\alpha(1)} \leq p_{\alpha(2)} \leq \ldots \leq p_{\alpha(n)}$. Then we have

$$\begin{matrix}
{\max\left( f_{(E,\alpha)} \right)} & {= (p_{\alpha(1)} - 1)\prod\limits_{j < \alpha(1)}p_{j} + (p_{\alpha(2)} - 1)\prod\limits_{j < \alpha(2)}p_{j} + \ldots + (p_{\alpha(n)} - 1)\prod\limits_{j < \alpha(n)}p_{j}} \\
 & {\leq p_{\alpha(1)}\prod\limits_{j < \alpha(1)}p_{j} + (p_{\alpha(2)} - 1)\prod\limits_{j < \alpha(2)}p_{j} + \ldots + (p_{\alpha(n)} - 1)\prod\limits_{j < \alpha(n)}p_{j}} \\
 & {= \prod\limits_{j < \alpha(2)}p_{j} + (p_{\alpha(2)} - 1)\prod\limits_{j < \alpha(2)}p_{j} + \ldots + (p_{\alpha(n)} - 1)\prod\limits_{j < \alpha(n)}p_{j}} \\
 & {= p_{\alpha(2)}\prod\limits_{j < \alpha(2)}p_{j} + \ldots + (p_{\alpha(n)} - 1)\prod\limits_{j < \alpha(n)}p_{j}} \\
 & {\leq \prod\limits_{j < \alpha(3)}p_{j} + \ldots + (p_{\alpha(n)} - 1)\prod\limits_{j < \alpha(n)}p_{j}} \\
 & {\leq \ldots} \\
 & {\leq p_{\alpha(n)}\prod\limits_{j < \alpha(n - 1)}p_{j} + (p_{\alpha(n)} - 1)\prod\limits_{j < \alpha(n)}p_{j}} \\
 & {= \prod\limits_{j < \alpha(n)}p_{j} + (p_{\alpha(n)} - 1)\prod\limits_{j < \alpha(n)}p_{j}} \\
 & {= p_{\alpha(n)}\prod\limits_{j < \alpha(n)}p_{j}} \\
 & {= \prod\limits_{j \leq \alpha(n)}p_{j}} \\
 & {\leq \prod\limits_{j \leq k}p_{j}} \\
 & {= N}
\end{matrix}$$

Thus, we have $f_{(E,\alpha)}:\lbrack 0,N^{\alpha}) \rightarrow \lbrack 0,N) \subset \mathbb{N}$.

Because $f_{(E,\alpha)}$ is a multilinear function, and because of the canonical isomorphisms, $f_{(E,\alpha)}$ can be described as the multilinear function

$$\begin{array}{r}
{\lbrack 0,p_{\alpha(1)}) \times \lbrack 0,p_{\alpha(2)}) \times \ldots \times \lbrack 0,p_{\alpha(n)})\overset{}{\rightarrow}\lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})}
\end{array}$$

We denote a vector space $V = \lbrack 0,p_{\alpha(1)}) \times \lbrack 0,p_{\alpha(2)}) \times \ldots \times \lbrack 0,p_{\alpha(n)})$ and a vector space $W = \lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})$. Then the layout function $f_{(E,\alpha)}$ is a linear map $V\overset{}{\rightarrow}W$.

Suppose $v_{1},v_{2},av_{1},bv_{2},av_{1} + bv_{2} \in V$, $f_{(E,\alpha)}(v_{1}) = w_{1}$, and $f_{(E,\alpha)}(v_{2}) = w_{2}$. Then we have

$$\begin{matrix}
{f_{(E,\alpha)}(v_{1})} & {= v_{1,1}d_{1} + v_{1,2}d_{2} + \ldots + v_{1,n}d_{n}} \\
{f_{(E,\alpha)}(v_{2})} & {= v_{2,1}d_{1} + v_{2,2}d_{2} + \ldots + v_{2,n}d_{n}} \\
{f_{(E,\alpha)}(av_{1})} & {= av_{1,1}d_{1} + av_{1,2}d_{2} + \ldots + av_{1,n}d_{n}} \\
 & {= af_{(E,\alpha)}(v_{1})} \\
{f_{(E,\alpha)}(bv_{2})} & {= bv_{2,1}d_{1} + bv_{2,2}d_{2} + \ldots + bv_{2,n}d_{n}} \\
 & {= bf_{(E,\alpha)}(v_{2})} \\
{f_{(E,\alpha)}(av_{1} + bv_{2})} & {= (av_{1} + bv_{2})_{1}d_{1} + (av_{1} + bv_{2})_{2}d_{2} + \ldots + (av_{1} + bv_{2})_{n}d_{n}} \\
 & {= af_{(E,\alpha)}(v_{1}) + bf_{(E,\alpha)}(v_{2})}
\end{matrix}$$

So $f_{(E,\alpha)}:V\overset{}{\rightarrow}W$ is indeed a linear (multilinear) map.

Given an index $1 \leq i \leq \alpha$, let $\delta_{i} \in \mathbb{N}^{\times \alpha}$ denote the coordinate that is zero everywhere except in the $k$-th position, where it is 1. Note here the indexing is 1-based, instead of the similar one used in Proposition 2.14 which is 0-based. $\delta_{i}$ is the basis vector of the vector space $V$ for $1 \leq i \leq \alpha$.

We send $\delta_{i}$ to $f_{(E,\alpha)}$ for $1 \leq i \leq \alpha$. Then we have

$$\begin{matrix}
{f_{(E,\alpha)}(\delta_{i})} & {= d_{i}} \\
 & {= \prod\limits_{j < \alpha(i)}p_{j}}
\end{matrix}$$

Given the canonical isomorphism $\lbrack 0,N) \cong \lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})$, we have the multilinear function $g:W\overset{}{\rightarrow}\mathbb{N}$

$$\begin{array}{r}
{g(w) = w_{1} + w_{2}p_{1} + w_{3}p_{1}p_{2} + \ldots + w_{k}\prod\limits_{j < k}p_{j}}
\end{array}$$

Given an index $1 \leq i \leq k$, let $\beta_{i} \in \mathbb{N}^{\times k}$ denote the coordinate that is zero everywhere except in the $k$-th position, where it is 1. $\beta_{i}$ is the basis vector of the vector space $W$ for $1 \leq i \leq k$.

Thus, we have

$$\begin{matrix}
{f_{(E,\alpha)}(\delta_{i})} & {= \prod\limits_{j < \alpha(i)}p_{j}} \\
 & {= g(\beta_{\alpha(i)})}
\end{matrix}$$

This means the basis vector $\delta_{i}$ in the vector space $V$ is sent to the basis vector $\beta_{\alpha(i)}$ in the vector space $W$ by the multilinear function.

Suppose $v = c_{1}\delta_{1} + c_{2}\delta_{2} + \ldots + c_{\alpha}\delta_{\alpha} \in V$. Then we have

$$\begin{matrix}
{f_{(E,\alpha)}(v)} & {= f_{(E,\alpha)}(c_{1}\delta_{1} + c_{2}\delta_{2} + \ldots + c_{\alpha}\delta_{\alpha})} \\
 & {= (c_{1}\delta_{1} + c_{2}\delta_{2} + \ldots + c_{\alpha}\delta_{\alpha})_{1}d_{1} + (c_{1}\delta_{1} + c_{2}\delta_{2} + \ldots + c_{\alpha}\delta_{\alpha})_{2}d_{2} + \ldots + (c_{1}\delta_{1} + c_{2}\delta_{2} + \ldots + c_{\alpha}\delta_{\alpha})_{\alpha}d_{\alpha}} \\
 & {= c_{1}d_{1} + c_{2}d_{2} + \ldots + c_{\alpha}d_{\alpha}} \\
 & {= c_{1}f_{(E,\alpha)}(\delta_{1}) + c_{2}f_{(E,\alpha)}(\delta_{2}) + \ldots + c_{\alpha}f_{(E,\alpha)}(\delta_{\alpha})} \\
 & {= c_{1}g(\beta_{\alpha(1)}) + c_{2}g(\beta_{\alpha(2)}) + \ldots + c_{\alpha}g(\beta_{\alpha(\alpha)})} \\
 & {= g(c_{1}\beta_{\alpha(1)} + c_{2}\beta_{\alpha(2)} + \ldots + c_{\alpha}\beta_{\alpha(\alpha)})}
\end{matrix}$$

Therefore, we have set up the basis vector mapping for the multilinear function $f_{(E,\alpha)}:V\overset{}{\rightarrow}W$. Given $v = c_{1}\delta_{1} + c_{2}\delta_{2} + \ldots + c_{\alpha}\delta_{\alpha} \in V$, it maps to $w = c_{1}\beta_{\alpha(1)} + c_{2}\beta_{\alpha(2)} + \ldots + c_{\alpha}\beta_{\alpha(\alpha)} \in W$.

This concludes the proof. 

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Lemma-3-12 "Lemma 3.12"){.headerlink}Lemma 3.12 {#Lemma-3-12 style="scroll-margin: 1em;"}

Elaborating on Remark 3.11, we have the following lemma, which indicates that composition in the category $\textbf{Fact}$ is compatible with the composition of layout functions.

Suppose we have morphisms of finite sets $\alpha:\underset{―}{n} \rightarrow \underset{―}{k}$, $\beta:\underset{―}{m} \rightarrow \underset{―}{n}$, and an expression $E = \lbrack p_{1}p_{2}\ldots p_{k}\rbrack$. Write $\gamma = \alpha \circ \beta$. Consider the composition

$$\begin{array}{r}
{\gamma_{E}:E^{\gamma} = (E^{\alpha})^{\beta}\overset{\beta_{E^{\alpha}}}{\rightarrow}E^{\alpha}\overset{\alpha_{E}}{\rightarrow}E}
\end{array}$$

in $\textbf{Fact}$. Then the associated layout functions satisfy the composition equality

$$\begin{array}{r}
{f_{(E,\gamma)} = f_{(E,\alpha)} \circ f_{(E^{\alpha},\beta)}}
\end{array}$$

*Proof*

Let $N = p_{1} \cdot p_{2} \cdot \ldots \cdot p_{k}$, $N^{\alpha} = p_{\alpha(1)} \cdot p_{\alpha(2)} \cdot \ldots \cdot p_{\alpha(n)}$, and $N^{\gamma} = p_{\gamma(1)} \cdot p_{\gamma(2)} \cdot \ldots \cdot p_{\gamma(m)}$. We use the canonical isomorphisms

$$\begin{matrix}
{\lbrack 0,N)} & {\cong \lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})} \\
{\lbrack 0,N^{\alpha})} & {\cong \lbrack 0,p_{\alpha(1)}) \times \lbrack 0,p_{\alpha(2)}) \times \ldots \times \lbrack 0,p_{\alpha(n)})} \\
{\lbrack 0,N^{\gamma})} & {\cong \lbrack 0,p_{\gamma(1)}) \times \lbrack 0,p_{\gamma(2)}) \times \ldots \times \lbrack 0,p_{\gamma(m)})}
\end{matrix}$$

to write the domains and codomains of the layout functions in question.

More specifically, we have

$$\begin{matrix}
{f_{(E,\alpha)}:\lbrack 0,N^{\alpha})} & {\rightarrow \lbrack 0,N)} \\
{f_{(E^{\alpha},\beta)}:\lbrack 0,N^{\gamma})} & {\rightarrow \lbrack 0,N^{\alpha})} \\
{f_{(E,\gamma)}:\lbrack 0,N^{\gamma})} & {\rightarrow \lbrack 0,N)}
\end{matrix}$$

We are trying to equate the multilinear function

$$\begin{array}{r}
{f_{(E,\gamma)}:\lbrack 0,p_{\gamma(1)}) \times \lbrack 0,p_{\gamma(2)}) \times \ldots \times \lbrack 0,p_{\gamma(m)})\overset{}{\rightarrow}\lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})}
\end{array}$$

with the composition of the two multilinear functions

$$\begin{matrix}
f_{(E,\alpha)} & {:\lbrack 0,p_{\alpha(1)}) \times \lbrack 0,p_{\alpha(2)}) \times \ldots \times \lbrack 0,p_{\alpha(n)})\overset{}{\rightarrow}\lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})} \\
f_{(E^{\alpha},\beta)} & {:\lbrack 0,p_{\gamma(1)}) \times \lbrack 0,p_{\gamma(2)}) \times \ldots \times \lbrack 0,p_{\gamma(m)})\overset{}{\rightarrow}\lbrack 0,p_{\alpha(1)}) \times \lbrack 0,p_{\alpha(2)}) \times \ldots \times \lbrack 0,p_{\alpha(n)})}
\end{matrix}$$

We denote a vector space $V = \lbrack 0,p_{\alpha(1)}) \times \lbrack 0,p_{\alpha(2)}) \times \ldots \times \lbrack 0,p_{\alpha(n)})$, a vector space $W = \lbrack 0,p_{1}) \times \lbrack 0,p_{2}) \times \ldots \times \lbrack 0,p_{k})$, and a vector space $U = \lbrack 0,p_{\gamma(1)}) \times \lbrack 0,p_{\gamma(2)}) \times \ldots \times \lbrack 0,p_{\gamma(m)})$. The basis vectors of $V$, $W$, and $U$ are $\delta_{i}$, $\sigma_{j}$, and $\tau_{l}$ for $1 \leq i \leq n$, $1 \leq j \leq k$, and $1 \leq l \leq m$.

Based on the basis vector mapping by Remark 3.11, given $u = c_{1}\tau_{1} + c_{2}\tau_{2} + \ldots + c_{m}\tau_{m} \in U$, by $f_{(E^{\alpha},\beta)}$, it maps to $v = c_{1}\delta_{\beta(1)} + c_{2}\delta_{\beta(2)} + \ldots + c_{m}\delta_{\beta(m)} \in V$. Then by $f_{(E,\alpha)}$, it maps to $w = c_{1}\sigma_{\alpha(\beta(1))} + c_{2}\sigma_{\alpha(\beta(2))} + \ldots + c_{m}\sigma_{\alpha(\beta(m))} \in W$.

Given $u = c_{1}\tau_{1} + c_{2}\tau_{2} + \ldots + c_{m}\tau_{m} \in U$, by $f_{(E,\gamma)}$, because $\gamma = \alpha \circ \beta$, $\gamma(i) = \alpha(\beta(i))$, it maps to $w^{\prime} = c_{1}\sigma_{\gamma(1)} + c_{2}\sigma_{\gamma(2)} + \ldots + c_{m}\sigma_{\gamma(m)} = c_{1}\sigma_{\alpha(\beta(1))} + c_{2}\sigma_{\alpha(\beta(2))} + \ldots + c_{m}\sigma_{\alpha(\beta(m))} \in W$.

Because $w = w^{\prime}$, we have $f_{(E,\gamma)} = f_{(E,\alpha)} \circ f_{(E^{\alpha},\beta)}$.

This concludes the proof. 

In Lemma 3.12, the per-mode condition of admissibility for composition (Definition 2.12) is satisfied. To see this, we have

$$\begin{matrix}
E & {= \lbrack p_{1}p_{2}\ldots p_{k}\rbrack} \\
E^{\alpha} & {= \lbrack p_{\alpha(1)}p_{\alpha(2)}\ldots p_{\alpha(n)}\rbrack} \\
E^{\gamma} & {= \lbrack p_{\gamma(1)}p_{\gamma(2)}\ldots p_{\gamma(m)}\rbrack}
\end{matrix}$$

$$\begin{matrix}
L_{(E,\alpha)} & {= \left( p_{\alpha(1)},p_{\alpha(2)},\ldots,p_{\alpha(n)} \right):\left( d_{1},d_{2},\ldots,d_{n} \right)}
\end{matrix}$$

where $d_{i} = \prod\limits_{j < \alpha(i)}p_{j}$.

$$\begin{matrix}
L_{(E^{\alpha},\beta)} & {= \left( p_{\alpha(\beta(1))},p_{\alpha(\beta(2))},\ldots,p_{\alpha(\beta(m))} \right):\left( d_{1}^{\prime},d_{2}^{\prime},\ldots,d_{m}^{\prime} \right)}
\end{matrix}$$

where $d_{i}^{\prime} = \prod\limits_{j < \beta(i)}p_{\alpha(j)}$.
CuTe Layout Algebra - Lei Mao&#39;s Log Book_files
Because

$$\begin{matrix}
M & {= p_{\alpha(1)} \cdot p_{\alpha(2)} \cdot \ldots \cdot p_{\alpha(n)}} \\
 & {= \left( \prod\limits_{j < \beta(i)}p_{\alpha(j)} \right) \cdot p_{\alpha(\beta(i))} \cdot p_{\alpha(\beta(i) + 1)} \cdot \ldots \cdot p_{\alpha(n)}} \\
 & {= \left( \prod\limits_{j < \beta(i)}p_{\alpha(j)} \right) \cdot M^{\prime}}
\end{matrix}$$

Thus, $M$ is left divisible by $d_{i}^{\prime}$, $M^{\prime}$ is weakly left divisible and also left divisible by $p_{\alpha(\beta(i))}$, and the per-mode condition of admissibility for composition is satisfied.

The disjointness condition in Definition 2.17 is satisfied when $\beta:\underset{―}{m} \rightarrow \underset{―}{n}$ is an injective function and may be violated when it is not.

When $\beta:\underset{―}{m} \rightarrow \underset{―}{n}$ is injective, we have $m \leq n$, and $\beta{(i)} \neq \beta{(j)}$ for $i \neq j$. By Definition 2.16, for each mode $i \in \lbrack 1,m\rbrack$, we have $N_{i} = p_{\alpha(\beta(i))}$, $I_{i} = \lbrack d_{i}^{\prime},d_{i}^{\prime}(N_{i} - 1)\rbrack$. $M^{\prime} = p_{\alpha(1)} \cdot p_{\alpha(2)} \cdot \ldots \cdot p_{\alpha(n - 1)}$. So the interval of definition is $J_{i} = I_{i} \cap \lbrack 1,M^{\prime})$. Because $d_{i}^{\prime} = \prod\limits_{j < \beta(i)}p_{\alpha(j)} \geq 1$, $d_{i}^{\prime}(N_{i} - 1) = \prod\limits_{j < \beta(i)}p_{\alpha(j)} \cdot (p_{\alpha(\beta(i))} - 1) = \prod\limits_{j \leq \beta(i)}p_{\alpha(j)} - \prod\limits_{j < \beta(i)}p_{\alpha(j)} < M^{\prime}$. Thus, $J_{i} = I_{i} = \lbrack\prod\limits_{j < \beta(i)}p_{\alpha(j)},\prod\limits_{j \leq \beta(i)}p_{\alpha(j)} - \prod\limits_{j < \beta(i)}p_{\alpha(j)}\rbrack$. Suppose we have a different mode $k$, $k \neq i$. Then $J_{k} = I_{k} = \lbrack\prod\limits_{j < \beta(k)}p_{\alpha(j)},\prod\limits_{j \leq \beta(k)}p_{\alpha(j)} - \prod\limits_{j < \beta(k)}p_{\alpha(j)}\rbrack$. Because $\beta(i) \neq \beta(k)$, without losing generality, we assume $\beta(i) < \beta(k)$. Then we have

$$\begin{matrix}
{\prod\limits_{j < \beta(k)}p_{\alpha(j)} - \left( \prod\limits_{j \leq \beta(i)}p_{\alpha(j)} - \prod\limits_{j < \beta(i)}p_{\alpha(j)} \right)} & {= \prod\limits_{j < \beta(k)}p_{\alpha(j)} - \prod\limits_{j \leq \beta(i)}p_{\alpha(j)} + \prod\limits_{j < \beta(i)}p_{\alpha(j)}} \\
 & {> 0}
\end{matrix}$$

Thus, $J_{i} \cap J_{k} = \varnothing$ for any $i \neq k$. The disjointness condition is satisfied.

When $\beta:\underset{―}{m} \rightarrow \underset{―}{n}$ is not injective, we don't have $\beta{(i)} \neq \beta{(j)}$ for $i \neq j$. The disjointness condition may be violated.

So Lemma 3.12 actually proves Theorem 2.18 for layouts that has any *arbitrary strides* (not yet any arbitrary shapes) of the second layout that satisfies Definition 2.17.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Definition-3-14 "Definition 3.14"){.headerlink}Definition 3.14 {#Definition-3-14 style="scroll-margin: 1em;"}

We now define a "realization" functor from the category $\textbf{Fact}$ to the category $\textbf{FinSet}$ that sends morphisms of ordered factorizations to their associated layout functions.

Let $R:\textbf{Fact} \rightarrow \textbf{FinSet}$ be the functor defined as follows:

1.  Let $E = \lbrack p_{1}p_{2}\ldots p_{k}\rbrack$ be an object of $\textbf{Fact}$ and let $N = p_{1} \cdot p_{2} \cdot \ldots \cdot p_{k}$. Then $R(E) = \lbrack 0,N)$.
2.  For every morphism $\alpha_{E}:E^{\alpha} \rightarrow E$, let $R(\alpha_{E}) = f_{(E,\alpha)}:\lbrack 0,N^{\alpha}) \rightarrow \lbrack 0,N)$ be as in Definition 3.9.

By Lemma 3.12, $R:\textbf{Fact} \rightarrow \textbf{FinSet}$ does indeed define a functor since it respects the composition of morphisms and identities as well.

We note that, as mentioned previously, $R$ does not contain every possible function expressible as a layout function in its image. However, it does contain every automorphism of $\lbrack 0,N)\overset{\cong}{\rightarrow}\lbrack 0,N)$ expressible as a layout function in its image.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Proposition-3-15 "Proposition 3.15"){.headerlink}Proposition 3.15 {#Proposition-3-15 style="scroll-margin: 1em;"}

Let $N > 0$ be a positive integer and let $f:\lbrack 0,N) \rightarrow \lbrack 0,N)$ be an automorphism such that there exists a layout $L$ of size $N$ with $f = f_{L}$. Then $f_{L}$ is in the image of the realization functor $R$.

*Proof*

Without loss of generality, we may suppose that the shape tuple of $L$ is given by $(p_{1},p_{2},\ldots,p_{k})$ where the $p_{i}$ are all prime numbers and $N = p_{1} \cdot p_{2} \cdot \ldots \cdot p_{k}$.

In order for $f_{L}$ to be an automorphism of $\lbrack 0,N)$, the sorted $L$, $L^{\varphi}$, must be of the form

$$\begin{array}{r}
{L^{\varphi} = \left( p_{\varphi(1)},p_{\varphi(2)},\ldots,p_{\varphi(k)} \right):\left( 1,p_{\varphi(1)},p_{\varphi(1)}p_{\varphi(2)},\ldots,\prod\limits_{1 \leq i < k}p_{\varphi(i)} \right)}
\end{array}$$

for some permutation $\varphi \in \Sigma_{k}$. This means that if we let $\psi = \varphi^{- 1}$ be the inverse permutation, then

$$\begin{array}{r}
{\psi_{E}:E^{\psi} = \lbrack p_{1}p_{2}\ldots p_{k}\rbrack = \lbrack p_{\psi(\varphi(1))}p_{\psi(\varphi(2))}\ldots p_{\psi(\varphi(k))}\rbrack\overset{}{\rightarrow}E = \lbrack p_{\varphi(1)}p_{\varphi(2)}\ldots p_{\varphi(k)}\rbrack}
\end{array}$$

is a morphism in $\textbf{Fact}$ that $R(\psi_{E}) = f_{L}$.

This concludes the proof.

### [](https://leimao.github.io/article/CuTe-Layout-Algebra/#Remark-3-16 "Remark 3.16"){.headerlink}Remark 3.16 {#Remark-3-16 style="scroll-margin: 1em;"}

One way to interpret Proposition 3.15 is that if we take the maximal subgroupoid $\textbf{Fact}^{\simeq}$ inside $\textbf{Fact}$, i.e., the subcategory of all invertible morphisms, then

$$\begin{array}{r}
{R:\textbf{Fact}^{\simeq} \rightarrow \textbf{FinSet}}
\end{array}$$

carves out exactly those permutations expressible as layouts. Our motivation for this description is that for a fixed integer $N > 0$, the subset $\Sigma_{N}^{L}$ of $\Sigma_{N}$ on those automorphisms expressible as layout functions is typically not a subgroup (being not generally closed under the group multiplication, i.e., composition).

Instead, if we let

$$\begin{array}{r}
{\textbf{Fact}_{N}^{\simeq} \subset \textbf{Fact}^{\simeq}}
\end{array}$$

be the full subgroupoid of those objects $\lbrack p_{1}p_{2}\ldots p_{k}\rbrack$ with $N = p_{1} \cdot p_{2} \cdot \ldots \cdot p_{k}$, then the $\Sigma_{N}^{L}$ consists of those morphisms in the image of $R$ on $\textbf{Fact}_{N}^{\simeq}$. However, we see that $\Sigma_{N}^{L}$ is closed under the operation of taking the group inverse (the objects taking permutations to their inverses are also in $\textbf{Fact}_{N}^{\simeq}$). Moreover, in the special case that $N$ is a prime power $p^{k}$, then $\Sigma_{N}^{L}$ is in fact a subgroup and is isomorphic to the symmetric group $\Sigma_{k}$. This corresponds to $\textbf{Fact}_{p^{k}}^{\simeq}$ being a groupoid with a single object $\lbrack pp\ldots p\rbrack$, i.e., a group.