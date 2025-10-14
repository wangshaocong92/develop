# FlashAttention：具有IO感知的快速且内存高效的精确注意力机制

Tri Dao

斯坦福大学计算机科学系

Daniel Y. Fu

纽约州立大学布法罗分校计算机科学与工程系

Stefano Ermon

斯坦福大学计算机科学系

Atri Rudra

{trid,danfu}@cs.stanford.edu, ermon@stanford.edu, atri@buffalo.edu, chrismre@cs.stanford.edu

Christopher Re

斯坦福大学计算机科学系

###### 摘要

Transformer 在长序列上速度慢且内存消耗大，因为自注意力模块的时间和内存复杂度是序列长度的二次方。许多近似注意力方法试图通过牺牲模型质量来降低计算复杂度以解决这个问题，但通常无法实现实际时钟速度的提升。我们认为，一个缺失的原则是使注意力算法具有 **IO 感知**——考虑 GPU 内存层次结构之间的读写。我们提出了 FlashAttention，一种 IO 感知的精确注意力算法，它使用平铺技术来减少 GPU 高带宽内存（HBM）和 GPU 片上 SRAM 之间的内存读写次数。我们分析了 FlashAttention 的 IO 复杂度，表明它比标准注意力需要更少的 HBM 访问次数，并且对于一定范围的 SRAM 大小是最优的。我们还将 FlashAttention 扩展到块稀疏注意力，产生了一种比任何现有近似注意力方法更快的近似注意力算法。FlashAttention 训练 Transformer 的速度比现有基线更快：在 BERT-large（序列长度 512）上相比 MLPerf 1.1 训练速度记录实现了 15% 的端到端时钟加速，在 GPT-2（序列长度 1K）上实现了 3 倍加速，在长距离竞技场（序列长度 1K-4K）上实现了 2.4 倍加速。FlashAttention 和块稀疏 FlashAttention 使得 Transformer 能够处理更长的上下文，从而产生更高质量的模型（在 GPT-2 上困惑度提升 0.7，在长文档分类上提升 6.4 个百分点）并带来全新的能力：首个在 Path-X 挑战（序列长度 16K，61.4% 准确率）和 Path-256（序列长度 64K，63.1% 准确率）上实现优于随机猜测性能的 Transformer。

## 1 引言

Transformer 模型 [82] 已成为自然语言处理和图像分类等应用中最广泛使用的架构。Transformer 变得越来越大 [5] 和越来越深 [83]，但为其配备更长的上下文仍然很困难 [80]，因为其核心的自注意力模块具有序列长度二次方的时间和内存复杂度。一个重要的问题是，使注意力更快、内存效率更高是否能帮助 Transformer 模型解决其在长序列上的运行时和内存挑战。

许多近似注意力方法旨在减少注意力的计算和内存需求。这些方法范围从稀疏近似 [51, 74] 到低秩近似 [12, 50, 84]，以及它们的组合 [9, 32, 3]。尽管这些方法将计算需求降低到序列长度的线性或近线性，但其中许多方法相比标准注意力并未显示出实际时钟速度的提升，并且未获得广泛采用。一个主要原因是它们专注于 FLOP 的减少（这可能与时钟速度无关），并且往往忽略了内存访问（IO）的开销。

在本文中，我们认为一个缺失的原则是使注意力算法具有 **IO 感知** [1]——即，仔细考虑对不同级别快慢内存（例如，在快速的 GPU 片上 SRAM 和相对较慢的 GPU 高带宽内存或 HBM [45] 之间，图 1 左）的读写。在现代

===== 第 2 页 =====

GPU 上，计算速度已经超过了内存速度 [61, 62, 63]，并且 Transformer 中的大多数操作都受限于内存访问 [43]。对于类似的内存受限操作，当读写数据可能占运行时的大部分时，IO 感知算法一直至关重要——例如数据库连接 [71]、图像处理 [70]、数值线性代数 [4] 等等 [40, 85]。然而，深度学习的常见 Python 接口（如 PyTorch 和 TensorFlow）不允许对内存访问进行细粒度控制。

我们提出了 FlashAttention，一种新的注意力算法，它能够以少得多的内存访问次数计算精确注意力。我们的主要目标是避免将注意力矩阵读写到 HBM。这需要 (i) 在没有整个输入的情况下计算 softmax 归约；(ii) 不为反向传播存储大的中间注意力矩阵。我们应用两种成熟的技术来解决这些挑战。(i) 我们重构注意力计算，将输入分成块并对输入块进行多次遍历，从而增量地执行 softmax 归约（也称为 **平铺**）。(ii) 我们存储前向传播中的 softmax 归一化因子，以便在反向传播中在芯片上快速 **重新计算** 注意力，这比从 HBM 读取中间注意力矩阵的标准方法更快。我们在 CUDA 中实现 FlashAttention 以实现对内存访问的细粒度控制，并将所有注意力操作融合到一个 GPU 核中。即使由于重新计算导致 FLOPs 增加，我们的算法由于 HBM 访问量的大幅减少，相比标准注意力既 **运行更快**（在 GPT-2 [67] 上高达 \(7.6\times\)，图 1 右）又 **使用更少的内存**——与序列长度呈线性关系。

我们分析了 FlashAttention 的 IO 复杂度 [1]，证明它需要 \(O(N^{2}d^{2}\bm{M}^{-1})\) 次 HBM 访问，其中 \(d\) 是头维度，\(\bm{M}\) 是 SRAM 的大小，而标准注意力需要 \(\Omega(Nd+N^{2})\) 次。对于典型的 \(d\) 和 \(\bm{M}\) 值，FlashAttention 需要的 HBM 访问次数比标准注意力少许多倍（最多减少 \(9\times\)，如图 2 所示）。此外，我们提供了一个下界，表明在所有 SRAM 大小上，没有精确注意力算法能在 HBM 访问次数上渐进地改进。

我们还展示了 FlashAttention 可以通过克服内存访问开销问题，作为实现近似注意力算法潜力的有用原语。作为概念验证，我们实现了块稀疏 FlashAttention，这是一种稀疏注意力算法，比 FlashAttention 快 \(2\)-\(4\times\)，可扩展到 64k 的序列长度。我们证明块稀疏 FlashAttention 的 IO 复杂度比 FlashAttention 好一个与稀疏比率成比例的因子。我们在第 5 节讨论了扩展到其他操作（多 GPU 上的注意力、核回归、块稀疏矩阵

图 1：**左：** FlashAttention 使用平铺来防止在（相对）较慢的 GPU HBM 上具体化大的 \(N\times N\) 注意力矩阵（虚线框）。在外层循环（红色箭头）中，FlashAttention 遍历 **K** 和 **V** 矩阵的块并将其加载到快速的片上 SRAM。在每个块中，FlashAttention 遍历 **Q** 矩阵的块（蓝色箭头），将其加载到 SRAM，并将注意力计算的输出写回 HBM。**右：** 在 GPT-2 上相对于 PyTorch 注意力实现的速度提升。FlashAttention 不读写大的 \(N\times N\) 注意力矩阵到 HBM，从而在注意力计算上实现了 \(7.6\times\) 的加速。

===== 第 3 页 =====

乘法）的进一步扩展。我们开源 FlashAttention 以便更容易地基于此原语进行构建。¹

脚注 1：FlashAttention 代码可在 https://github.com/HazyResearch/flash-attention 获取。

我们通过经验验证了 FlashAttention 通过建模更长的上下文来加速模型训练并提高模型质量。我们还将 FlashAttention 和块稀疏 FlashAttention 的运行时和内存占用与先前的注意力实现进行了基准测试。

*   **更快的模型训练。** FlashAttention 在实际时钟时间上更快地训练 Transformer 模型。我们训练 BERT-large（序列长度 512）比 MLPerf 1.1 [58] 中的训练速度记录快 15%，训练 GPT2（序列长度 1K）比 HuggingFace [87] 和 Megatron-LM [77] 的基线实现快 3 倍，训练长距离竞技场（序列长度 1K-4K）比基线快 2.4 倍。
*   **更高质量的模型。** FlashAttention 将 Transformer 扩展到更长的序列，从而提高了它们的质量并实现了新的能力。我们观察到在 GPT-2 上困惑度提高了 0.7，在长文档分类 [13] 上通过建模更长的序列提升了 6.4 个百分点。FlashAttention 使得第一个 Transformer 能够仅在使用了更长序列长度（16K）的情况下，在 Path-X [80] 挑战上实现优于随机猜测的性能。块稀疏 FlashAttention 使得 Transformer 能够扩展到更长的序列（64K），产生了第一个在 Path-256 上能够实现优于随机猜测性能的模型。
*   **注意力基准测试。** FlashAttention 在从 128 到 2K 的常见序列长度上比标准注意力实现快达 3 倍，并可扩展到 64K。在序列长度达到 512 时，FlashAttention 比任何现有的注意力方法都快且内存效率更高，而对于超过 1K 的序列长度，一些近似注意力方法（例如 Linformer）开始变得更快。另一方面，块稀疏 FlashAttention 比我们所知的所有现有近似注意力方法都要快。

## 2 背景

我们提供了现代硬件（GPU）上常见深度学习操作性能特性的一些背景。我们还描述了注意力的标准实现。

### 2.1 硬件性能

我们这里主要关注 GPU。其他硬件加速器的性能类似 [46, 48]。

**GPU 内存层次结构。** GPU 内存层次结构（图 1 左）包括多种不同大小和速度的内存，较小的内存更快。例如，A100 GPU 具有 40-80GB 的高带宽内存（HBM），带宽为 1.5-2.0TB/s，以及每个流式多处理器 192KB 的片上 SRAM（共 108 个），带宽估计约为 19TB/s [44, 45]。片上 SRAM 比 HBM 快一个数量级，但大小小了许多数量级。随着计算速度相对于内存速度越来越快 [61, 62, 63]，操作越来越受限于内存（HBM）访问。因此，利用快速 SRAM 变得更加重要。

**执行模型。** GPU 有大量线程来执行一个操作（称为核）。每个核从 HBM 加载输入到寄存器和 SRAM，进行计算，然后将输出写入 HBM。

**性能特性。** 根据计算和内存访问的平衡，操作可以分为计算受限或内存受限。这通常通过 **算术强度** [85] 来衡量，即每次内存访问字节数的算术操作数。

1.  计算受限：操作所花费的时间由算术操作的数量决定，而访问 HBM 的时间要小得多。典型的例子是具有大内部维度的矩阵乘法和具有大量通道的卷积。
2.  内存受限：操作所花费的时间由内存访问的次数决定，而计算所花费的时间要小得多。例子包括大多数其他操作：逐元素操作（例如，激活、dropout）和归约操作（例如，求和、softmax、批量归一化、层归一化）。

**核融合。** 加速内存受限操作的最常见方法是核融合：如果对同一输入应用多个操作，则可以从 HBM 加载一次输入，而不是为每个操作加载多次。编译器可以自动融合许多逐元素操作 [53, 65, 75]。

===== 第 4 页 =====

然而，在模型训练的背景下，中间值仍然需要写入 HBM 以保存用于反向传播，这降低了朴素核融合的有效性。

### 2.2 标准注意力实现

给定输入序列 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\)，其中 \(N\) 是序列长度，\(d\) 是头维度，我们想要计算注意力输出 \(\mathbf{O}\in\mathbb{R}^{N\times d}\)：

\[\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad \mathbf{P}=\text{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad \mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},\]

其中 softmax 按行应用。

标准注意力实现将矩阵 \(\mathbf{S}\) 和 \(\mathbf{P}\) 具体化到 HBM，这需要 \(O(N^{2})\) 内存。通常 \(N\gg d\)（例如，对于 GPT2，\(N=1024\) 且 \(d=64\)）。我们在算法 3.1 中描述了标准注意力实现。由于部分或大部分操作是内存受限的（例如，softmax），大量的内存访问转化为较慢的时钟时间。

应用于注意力矩阵的其他逐元素操作（例如应用于 \(\mathbf{S}\) 的掩码或应用于 \(\mathbf{P}\) 的 dropout）加剧了这个问题。因此，有许多尝试来融合几个逐元素操作，例如将掩码与 softmax 融合 [77]。

在第 3.2 节中，我们将展示标准注意力实现执行了序列长度 \(N\) 的二次方次 HBM 访问。我们还比较了标准注意力和我们的方法（FlashAttention）的 FLOPs 数量和 HBM 访问次数。

[ht] 标准注意力实现
**要求：** 矩阵 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\) 在 HBM 中。
1.  从 HBM 按块加载 \(\mathbf{Q},\mathbf{K}\)，计算 \(\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\)，将 \(\mathbf{S}\) 写入 HBM。
2.  从 HBM 读取 \(\mathbf{S}\)，计算 \(\mathbf{P}=\text{softmax}(\mathbf{S})\)，将 \(\mathbf{P}\) 写入 HBM。
3.  从 HBM 按块加载 \(\mathbf{P}\) 和 \(\mathbf{V}\)，计算 \(\mathbf{O}=\mathbf{P}\mathbf{V}\)，将 \(\mathbf{O}\) 写入 HBM。
4.  返回 \(\mathbf{O}\)。

## 3 FlashAttention：算法、分析和扩展

我们展示了如何用更少的 HBM 读写次数计算精确注意力，并且不为反向传播存储大的中间矩阵。这产生了一种既内存高效又在时钟时间上更快的注意力算法。我们分析了其 IO 复杂度，表明我们的方法相比标准注意力需要少得多的 HBM 访问次数。我们进一步展示了 FlashAttention 可以通过扩展到处理块稀疏注意力而成为一个有用的原语。

我们这里为阐述方便主要关注前向传播；附录 B 包含反向传播的细节。

### 3.1 具有平铺和重新计算的高效注意力算法

给定 HBM 中的输入 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\)，我们的目标是计算注意力输出 \(\mathbf{O}\in\mathbb{R}^{N\times d}\) 并将其写入 HBM。我们的目标是减少 HBM 访问量（达到 \(N\) 的次二次方）。

我们应用两种成熟的技术（平铺、重新计算）来克服在次二次方 HBM 访问中计算精确注意力的技术挑战。我们在算法 3.1 中描述了这一点。主要思想是我们将输入 \(\mathbf{Q},\mathbf{K},\mathbf{V}\) 分成块，将它们从慢速 HBM 加载到快速 SRAM，然后计算相对于这些块的注意力输出。通过在将每个块的输出相加之前用正确的归一化因子进行缩放，我们在最后得到正确的结果。

**平铺。** 我们分块计算注意力。Softmax 耦合了 \(\mathbf{K}\) 的列，因此我们使用缩放 [51, 60, 66] 来分解大的 softmax。为了数值稳定性，向量 \(x\in\mathbb{R}^{B}\) 的 softmax 计算为：

\[m(x):=\max_{i} x_{i},\quad f(x):=\begin{bmatrix}e^{x_{1}-m(x)}&\ldots&e^{x_{B}-m(x)}\end{bmatrix},\quad \ell(x):=\sum_{i}f(x)_{i},\quad \text{softmax}(x):=\frac{f(x)}{\ell(x)}.\]

===== 第 5 页 =====

对于向量 \(x^{(1)},x^{(2)}\in\mathbb{R}^{B}\)，我们可以将拼接后的 \(x=\left[x^{(1)} x^{(2)}\right]\in\mathbb{R}^{2B}\) 的 softmax 分解为：

\[m(x)=m(\left[x^{(1)} x^{(2)}\right])=\max(m(x^{(1)}),m(x^{(2)})),\quad f(x)=\left[e^{m(x^{(1)})-m(x)}f(x^{(1)})\quad e^{m(x^{(2)})-m(x)}f(x^{(2)})\right],\]
\[\ell(x)=\ell(\left[x^{(1)} x^{(2)}\right])=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)}),\quad \text{softmax}(x)=\frac{f(x)}{\ell(x)}.\]

因此，如果我们跟踪一些额外的统计量 \((m(x),\ell(x))\)，我们可以逐块计算 softmax。² 因此，我们将输入 \(\mathbf{Q},\mathbf{K},\mathbf{V}\) 分成块（算法 3.2 第 3 行），计算 softmax 值以及额外的统计量（算法 3.2 第 9 行），并组合结果（算法 3.2 第 12 行）。
脚注 2：这种聚合风格称为 **代数聚合** [33]。

**重新计算。** 我们的目标之一是不为反向传播存储 \(O(N^{2})\) 的中间值。反向传播通常需要矩阵 \(\mathbf{S},\mathbf{P}\in\mathbb{R}^{N\times N}\) 来计算关于 \(\mathbf{Q},\mathbf{K},\mathbf{V}\) 的梯度。然而，通过存储输出 \(\mathbf{O}\) 和 softmax 归一化统计量 \((m,\ell)\)，我们可以在反向传播中从 SRAM 中的 \(\mathbf{Q},\mathbf{K},\mathbf{V}\) 块轻松地重新计算注意力矩阵 \(\mathbf{S}\) 和 \(\mathbf{P}\)。这可以看作是一种选择性梯度检查点 [34, 10] 的形式。虽然梯度检查点已被建议用于减少所需的最大内存量 [66]，但所有（我们所知的）实现都必须以速度换取内存。相比之下，即使有更多的 FLOPs，由于减少了 HBM 访问，我们的重新计算加速了反向传播（图 2）。完整的反向传播描述在附录 B 中。

**实现细节：核融合。** 平铺使我们能够在一个 CUDA 核中实现我们的算法，从 HBM 加载输入，执行所有计算步骤（矩阵乘法、softmax、可选的掩码和 dropout、矩阵乘法），然后将结果写回 HBM（掩码和 dropout 在附录 B 中）。这避免了从 HBM 重复读写输入和输出。

FlashAttention

矩阵 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\) 在 HBM 中，片上 SRAM 大小为 \(M\)。

1.  设置块大小 \(B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)\)。
2.  在 HBM 中初始化 \(\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}\)。
3.  将 \(\mathbf{Q}\) 分成 \(T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil\) 个块 \(\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}\)，每个块大小为 \(B_{r}\times d\)，并将 \(\mathbf{K},\mathbf{V}\) 分成 \(T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil\) 个块 \(\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}\) 和 \(\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}\)，每个块大小为 \(B_{c}\times d\)。
4.  将 \(\mathbf{O}\) 分成 \(T_{r}\) 个块 \(\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}\)，每个块大小为 \(B_{r}\times d\)，将 \(\ell\) 分成 \(T_{r}\) 个块 \(\ell_{i},\ldots,\ell_{T_{r}}\)，每个块大小为 \(B_{r}\)，将 \(m\) 分成 \(T_{r}\) 个块 \(m_{1},\ldots,m_{T_{r}}\)，每个块大小为 \(B_{r}\)。
5.  **for** \(1\leq j\leq T_{c}\) **do**
6.      从 HBM 加载 \(\mathbf{K}_{j},\mathbf{V}_{j}\) 到片上 SRAM。
7.      **for** \(1\leq i\leq T_{r}\) **do**
8.          从 HBM 加载 \(\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}\) 到片上 SRAM。
9.          在芯片上，计算 \(\mathbf{S}_{ij}=\mathbf{Q}_{i}\mathbf{K}_{j}^{T}\in\mathbb{R}^{B_{r}\times B_{c}}\)。
10.         在芯片上，计算 \(\tilde{m}_{ij}=\text{rowmax}(\mathbf{S}_{ij})\in\mathbb{R}^{B_{r}}\)，\(\tilde{\mathbf{P}}_{ij}=\exp(\mathbf{S}_{ij}-\tilde{m}_{ij})\in\mathbb{R}^{B_{r}\times B_{c}}\)（逐点），\(\tilde{\ell}_{ij}=\text{rowsum}(\tilde{\mathbf{P}}_{ij})\in\mathbb{R}^{B_{r}}\)。
11.         在芯片上，计算 \(m^{\text{new}}_{i}=\max(m_{i},\tilde{m}_{ij})\in\mathbb{R}^{B_{r}}\)，\(\ell^{\text{new}}_{i}=e^{m_{i}-m^{\text{new}}_{i}}\ell_{i}+e^{\tilde{m}_{ij}-m^{\text{new}}_{i}}\tilde{\ell}_{ij}\in\mathbb{R}^{B_{r}}\)。
12.         写入 \(\mathbf{O}_{i}\leftarrow \text{diag}(\ell^{\text{new}}_{i})^{-1}(\text{diag}(\ell_{i})e^{m_{i}-m^{\text{new}}_{i}}\mathbf{O}_{i}+e^{\tilde{m}_{ij}-m^{\text{new}}_{i}}\tilde{\mathbf{P}}_{ij}\mathbf{V}_{j})\) 到 HBM。
13.         写入 \(\ell_{i}\leftarrow \ell^{\text{new}}_{i}\)，\(m_{i}\leftarrow m^{\text{new}}_{i}\) 到 HBM。
14.     **end for**
15. **end for**
16. 返回 \(\mathbf{O}\)。

我们展示了 FlashAttention 的正确性、运行时间和内存需求（证明在附录 C）。

**定理 1.** _算法 3.2 返回 \(\mathbf{O}=\textnormal{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}\)，具有 \(O(N^{2}d)\) FLOPs，并且除了输入和输出之外需要 \(O(N)\) 的额外内存。_

### 3.2 分析：FlashAttention 的 IO 复杂度

我们分析了 FlashAttention 的 IO 复杂度，显示与标准注意力相比 HBM 访问次数显著减少。我们还提供了一个下界，证明没有精确注意力算法可以

===== 第 6 页 =====

在所有 SRAM 大小上渐进地改进 HBM 访问次数。证明在附录 C 中。

**定理 2.** _设 \(N\) 为序列长度，\(d\) 为头维度，\(\bm{M}\) 为 SRAM 大小，满足 \(d\leq \bm{M}\leq Nd\)。标准注意力（算法 3）需要 \(\Theta(Nd+N^{2})\) 次 HBM 访问，而 FlashAttention（算法 3）需要 \(\Theta(N^{2}d^{2}\bm{M}^{-1})\) 次 HBM 访问。_

对于典型的 \(d\)（64-128）和 \(\bm{M}\)（约 100KB）值，\(d^{2}\) 比 \(\bm{M}\) 小许多倍，因此 FlashAttention 需要的 HBM 访问次数比标准实现少许多倍。这导致更快的执行和更低的内存占用，我们在第 4.3 节对此进行了验证。

证明的主要思想是，给定 SRAM 大小 \(\bm{M}\)，我们可以加载大小为 \(\Theta(\bm{M})\) 的 \(\mathbf{K}\)、\(\mathbf{V}\) 块（算法 3 第 6 行）。对于 \(\mathbf{K}\) 和 \(\mathbf{V}\) 的每个块，我们遍历 \(\mathbf{Q}\) 的所有块（算法 3 第 7 行）以计算中间值，这导致对 \(\mathbf{Q}\) 进行 \(\Theta(Nd\bm{M}^{-1})\) 次遍历。每次遍历加载 \(\Theta(Nd)\) 个元素，总计 \(\Theta(N^{2}d^{2}\bm{M}^{-1})\) 次 HBM 访问。我们类似地证明了标准注意力的反向传播需要 \(\Theta(Nd+N^{2})\) 次 HBM 访问，而 FlashAttention 的反向传播需要 \(\Theta(N^{2}d^{2}\bm{M}^{-1})\) 次 HBM 访问（附录 B）。

我们证明了一个下界：对于所有 \(\bm{M}\)（SRAM 大小）的值，当计算精确注意力时，不能在 HBM 访问次数上渐进地改进。

**命题 3.** _设 \(N\) 为序列长度，\(d\) 为头维度，\(\bm{M}\) 为 SRAM 大小，满足 \(d\leq \bm{M}\leq Nd\)。不存在一种算法能在范围 \([d,Nd]\) 内的所有 \(\bm{M}\) 上以 \(o(N^{2}d^{2}\bm{M}^{-1})\) 次 HBM 访问计算精确注意力。_

证明依赖于这样一个事实：对于 \(\bm{M}=\Theta(Nd)\)，任何算法必须执行 \(\bm{\Omega}(N^{2}d^{2}\bm{M}^{-1})=\bm{\Omega}(Nd)\) 次 HBM 访问。这种在 \(\bm{M}\) 的子范围内的下界在流算法文献中很常见 [88]。我们将证明关于 \(\bm{M}\) 的参数化复杂度 [27] 下界作为令人兴奋的未来工作。

我们验证了 HBM 访问次数是注意力运行时间的主要决定因素。在图 2（左）中，我们看到即使 FlashAttention 与标准注意力相比具有更高的 FLOP 计数（由于反向传播中的重新计算），它的 HBM 访问次数要少得多，从而导致更快的运行时间。在图 2（中），我们改变 FlashAttention 的块大小 \(B_{c}\)，这导致不同数量的 HBM 访问，并测量前向传播的运行时间。随着块大小的增加，HBM 访问次数减少（因为我们对输入进行的遍历次数减少），运行时间减少。对于足够大的块大小（超过 256），运行时间随后受其他因素（例如，算术操作）的限制。此外，更大的块大小将无法放入小的 SRAM 中。

### 3.3 扩展：块稀疏 FlashAttention

我们将 FlashAttention 扩展到近似注意力：我们提出了块稀疏 FlashAttention，其 IO 复杂度比 FlashAttention 小一个与稀疏度成比例的因子。

给定输入 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\) 和一个掩码矩阵 \(\widetilde{\mathbf{M}}\in\{0,1\}^{N\times N}\)，我们想要计算：

\[\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad \mathbf{P}=\operatorname{softmax}(\mathbf{S}\odot\mathbbm{1}_{\widetilde{\mathbf{M}}})\in\mathbb{R}^{N\times N},\quad \mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},\]

其中 \((\mathbf{S}\odot\mathbbm{1}_{\widetilde{\mathbf{M}}})_{kl}=\mathbf{S}_{kl}\) 如果 \(\widetilde{\mathbf{M}}_{kl}=1\)，否则为 \(-\infty\)。我们要求 \(\widetilde{\mathbf{M}}\) 具有块形式：对于某个块大小 \(B_{r},B_{c}\)，对于所有 \(k,l\)，\(\widetilde{\mathbf{M}}_{k,l}=\mathbf{M}_{lj}\)，其中 \(i=\lfloor k/B_{r}\rfloor,j=\lfloor l/B_{c}\rfloor\)，对于某个 \(\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}\)。

图 2：**左：** 标准注意力和 FlashAttention 在 GPT-2 medium（序列长度 1024，头维度 64，16 个头，批量大小 64）上的前向+反向运行时间，在 A100 GPU 上。HBM 访问是影响运行时间的主要因素。**中：** FlashAttention（序列长度 1024，头维度 64，16 个头，批量大小 64）在 A100 GPU 上的前向运行时间。更少的 HBM 访问导致更快的运行时间，但达到一个点后不再明显。**右：** 块稀疏 FlashAttention 的运行时间（对于序列长度 4K）比 FlashAttention 快一个与稀疏度成比例的因子。

===== 第 7 页 =====

给定一个预定义的块稀疏掩码 \(\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}\)，我们可以轻松地调整算法 1 以仅计算注意力矩阵的非零块。该算法与算法 1 相同，只是我们跳过零块。我们在附录 B 的算法 5 中重现了算法描述。

我们还分析了块稀疏 FlashAttention 的 IO 复杂度。

**命题 4.** _设 \(N\) 为序列长度，\(d\) 为头维度，\(M\) 为 SRAM 大小，满足 \(d\leq M\leq Nd\)。块稀疏 FlashAttention（算法 5）需要 \(\Theta(Nd+N^{2}d^{2}M^{-1}s)\) 次 HBM 访问，其中 \(s\) 是块稀疏掩码中非零块的比例。_

我们看到应用块稀疏性对 IO 复杂度中较大的项直接带来了稀疏度的改进。对于大序列长度 \(N\)，\(s\) 通常设置为 \(N^{-1/2}\) [11] 或 \(N^{-1}\log N\) [3, 17, 92]，导致 \(\Theta(N\sqrt{N})\) 或 \(\Theta(N\log N)\) 的 IO 复杂度。对于下游实验，我们使用固定的蝴蝶稀疏模式 [17]，它已被证明能够近似任意稀疏性 [16]。

在图 2（右）中，我们验证了随着稀疏度的增加，块稀疏 FlashAttention 的运行时间成比例地提高。在 LRA 基准测试中，块稀疏 FlashAttention 实现了 \(2.8\times\) 的加速，同时性能与标准注意力相当（第 4 节）。

## 4 实验

我们评估了使用 FlashAttention 训练 Transformer 模型的影响。我们验证了关于训练时间和模型准确性的两个主张，并报告了注意力运行时间和内存基准测试。

*   **训练速度。** FlashAttention 在 BERT 上的训练速度超过了 MLPerf 1.1 [58] 速度记录 \(15\%\)，并且相比标准 Transformer，在 GPT-2 上比 HuggingFace [87] 快达 \(3\times\)，比 Megatron [77] 快 \(1.8\times\)。FlashAttention 将长距离竞技场（LRA）基准测试加速 \(2.4\times\)。
*   **质量。** FlashAttention 将 Transformer 扩展到更长的序列，产生更高质量。FlashAttention 以上下文长度 \(4\)K 训练 GPT-2 比 Megatron 以上下文长度 \(1\)K 训练 GPT-2 更快，同时实现了 \(0.7\) 更好的困惑度。建模更长的序列在两个长文档分类任务上产生了 \(6.4\) 个百分点的提升。最后，FlashAttention 产生了 **第一个 Transformer** 能够在具有挑战性的 Path-X 任务（序列长度 \(16\)K）上实现优于随机猜测的性能，而块稀疏 FlashAttention 产生了我们所知的 **第一个序列模型** 能够在 Path-256（序列长度 \(64\)K）上实现优于随机猜测的性能。
*   **注意力基准测试。** 我们根据序列长度测量 FlashAttention 和块稀疏 FlashAttention 的运行时间和内存性能。我们确认 FlashAttention 的内存占用与序列长度成线性比例，并且在常见序列长度（最多 \(2\)K）上比标准注意力快达 \(3\times\)。我们确认块稀疏 FlashAttention 的运行时间与序列长度成线性比例，并且比所有现有的近似注意力基线更快。额外的实验细节在附录 E 中。

### 4.1 使用 FlashAttention 实现更快的模型

**BERT。** FlashAttention 产生了我们所知的最快的单节点 BERT 训练速度。我们使用 FlashAttention 在 Wikipedia 上训练一个 BERT-large [22] 模型。表 1 将我们的训练时间与设置了 MLPerf 1.1 [58] 训练速度记录的 Nvidia 实现进行了比较。我们的实现快 \(15\%\)。

**GPT-2。** FlashAttention 在大型 OpenWebtext 数据集 [32] 上为 GPT-2 [67] 提供了比广泛使用的 HuggingFace [87] 和 Megatron-LM [77] 实现更快的训练时间。表 2 显示与 Huggingface 相比端到端加速高达 \(3\times\)，与 Megatron-LM 相比加速 \(1.7\times\)。FlashAttention

===== 第 8 页 =====

实现了与其他两种实现相同的困惑度，因为我们没有改变模型定义。附录 E 包含了整个训练过程中的验证困惑度图，确认 FlashAttention 与基线一样数值稳定，并产生相同的训练/验证曲线。

**长距离竞技场。** 我们在长距离竞技场（LRA [80]）基准测试上比较了普通 Transformer（使用标准实现或 FlashAttention）。我们测量了所有模型的准确率、吞吐量和训练时间。每个任务具有不同的序列长度，范围在 1024 到 4096 之间。我们遵循 Tay 等人 [80] 和 Xiong 等人 [90] 的实现和实验设置。³ 表 3 显示 FlashAttention 相比标准注意力实现了高达 2.4 倍的加速。块稀疏 FlashAttention 比我们测试过的所有近似注意力方法都要快。
脚注 3：LRA 准确率结果已知高度依赖于调优过程 [90]。我们复现的基线性能优于原始比较 [80] 中报告的结果。

### 4.2 具有更长序列的更好模型

**具有长上下文的语言建模。** FlashAttention 的运行时间和内存效率使我们能够将 GPT-2 的上下文长度增加 4 倍，同时仍然比 Megatron-LM 的优化实现运行得更快。表 4 显示，具有 FlashAttention 和上下文长度 4K 的 GPT-2 仍然比来自 Megatron 的上下文长度 1K 的 GPT-2 快 30%，同时实现了 0.7 更好的困惑度。

**长文档分类。** 使用 FlashAttention 以更长的序列训练 Transformer 提高了在 MIMIC-III [47] 和 ECtHR [6, 7] 数据集上的性能。MIMIC-III 包含重症监护室患者出院摘要，每个摘要都有多个标签。ECtHR 包含来自

表 4：使用 FlashAttention 的 GPT-2 small，与 Megatron-LM 相比上下文长度大 4 倍，仍然快 30%，同时实现了 0.7 更好的困惑度。报告了在 8 个 A100 GPU 上的训练时间。

表 3：标准注意力、FlashAttention、块稀疏 FlashAttention 和近似注意力基线在长距离竞技场基准测试上的性能。

表 2：使用 FlashAttention 的 GPT-2 small 和 medium 相比 Huggingface 实现实现了高达 3 倍的加速，相比 Megatron-LM 实现了高达 1.7 倍的加速。报告了在 8 个 A100 GPU 上的训练时间。

===== 第 9 页 =====

欧洲人权法院的法律案件，每个案件都被映射到据称被侵犯的《人权公约》条款。这两个数据集都包含非常长的文本文档；MIMIC 中的平均令牌数为 2,395 个，最长的文档包含 14,562 个令牌，而 ECtHR 中的平均和最长数量分别为 2,197 和 49,392。我们评估了增加预训练 RoBERTa 模型 [56] 序列长度带来的提升（我们重复位置嵌入，如 Beltagy 等人 [3]）。

表 5 显示，序列长度 16K 在 MIMIC 上比长度 512 高出 4.3 个百分点，而长度 8K 在 ECtHR 上比长度 512 高出 8.5 个百分点。这些差异可能是由于细微的分布偏移：MIMIC-III 包含专业的医学文本，因此可能更容易受到文档长度分布偏移的影响，而 ECtHR 包含通用语言。

**Path-X 和 Path-256。** Path-X 和 Path-256 基准测试来自长距离竞技场基准测试，旨在测试长上下文。任务是分类黑白 128×128（或 256×256）图像中的两个点是否有路径连接，图像被逐个像素地输入到 transformer。在先前的工作中，所有 transformer 模型要么内存不足，要么仅达到随机性能 [80]。人们一直在寻找能够建模如此长上下文的其他架构 [37]。我们在这里首次展示了 Transformer 模型能够解决 Path-X 和 Path-256 的结果（表 6）。我们在 Path-64 上预训练一个 transformer，然后通过空间插值位置嵌入转移到 Path-X。FlashAttention 在 Path-X 上实现了 61.4 的准确率。此外，块稀疏 FlashAttention 使得 Transformer 能够扩展到序列长度 64K，在 Path-256 上实现了 63.1 的准确率⁴。
脚注 4：Path-256 需要更长的序列，但路径相对 Path-X 较短，因此更容易获得更高的准确率。

### 4.3 注意力基准测试

我们改变序列长度，并在一个具有 40 GB HBM 的 A100 GPU 上，使用 dropout 和填充掩码，测量 FlashAttention 和块稀疏 FlashAttention 与各种注意力基线的运行时间和内存使用情况。我们与精确注意力、近似注意力和稀疏注意力的参考实现进行比较。我们在正文中报告了基线的子集；附录 E 包含更多基线和完整细节。

表 5：使用 FlashAttention 在不同序列长度下的长文档性能（微 \(F_{1}\)）。

图 3：**左：** 前向传播 + 反向传播的运行时间。**右：** 注意力内存使用情况。

表 6：我们报告了第一个能够在 Path-X 和 Path-256 上实现非随机性能的 Transformer 模型。

===== 第 10 页 =====

**运行时间。** 图 3（左）报告了 FlashAttention 和块稀疏 FlashAttention 与精确、近似和稀疏注意力基线相比的前向 + 反向传播运行时间（以毫秒为单位）（具体数字在附录 E）。运行时间随序列长度二次增长，但 FlashAttention 的运行速度显著快于 **精确注意力** 基线，比 PyTorch 实现快达 \(3\times\)。许多近似/稀疏注意力机制的运行时间随序列长度线性增长，但由于内存访问次数更少，对于短序列，FlashAttention 仍然比近似和稀疏注意力运行得更快。**近似注意力** 的运行时间在序列长度介于 512 和 1024 之间时开始与 FlashAttention 交叉。另一方面，块稀疏 FlashAttention 在我们所知的所有序列长度上，比所有精确、稀疏和近似注意力的实现都要快。

**内存占用。** 图 3（右）显示了 FlashAttention 和块稀疏 FlashAttention 与各种精确、近似和稀疏注意力基线相比的内存占用。FlashAttention 和块稀疏 FlashAttention 具有相同的内存占用，随序列长度线性增长。FlashAttention 比 **精确注意力** 基线内存效率高多达 \(20\times\)，并且比 **近似注意力** 基线更内存高效。除 Linformer 外，所有其他算法在达到 64K 之前在 A100 GPU 上内存不足，而 FlashAttention 仍然比 Linformer 高效 \(2\times\)。

## 5 局限性与未来方向

我们讨论了我们方法的局限性和未来方向。相关工作在附录 A 中给出。

**编译到 CUDA。** 我们当前构建 IO 感知注意力实现的方法需要为每个新的注意力实现编写一个新的 CUDA 核。这需要用比 PyTorch 低得多的语言编写注意力算法，并且需要大量的工程努力。实现也可能无法跨 GPU 架构移植。这些局限性表明需要一种方法，支持用高级语言（例如 PyTorch）编写注意力算法，并编译成 CUDA 中的 IO 感知实现——类似于图像处理中的 Halide 等努力 [70]。

**IO 感知深度学习。** 我们相信 IO 感知方法可以扩展到注意力之外。注意力是 Transformer 中内存最密集的计算，但深度网络中的每一层都会接触 GPU HBM。我们希望我们的工作能激发对其他模块的 IO 感知实现。我们在附录 D 中讨论了这些潜在的扩展。

**多 GPU IO 感知方法。** 我们的 IO 感知注意力实现在单个 GPU 上计算注意力时在常数因子内是最优的。然而，注意力计算可能可以在多个 GPU 之间并行化 [72]。使用多个 GPU 增加了 IO 分析的另一个层次——考虑 GPU 之间的数据传输。我们希望我们的工作能激发未来在这个方向上的工作。

### 致谢

我们的实现以 Apex 的 FMHA 代码 (https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha) 作为起点。我们感谢 Young-Jun Ko 对其 FMHA 实现的深入解释以及他对我们关于 CUDA 问题的深思熟虑的回答。我们感谢 Sabri Eyuboglu, Megan Leszczynski, Laurel Orr, Yuhuai Wu, Beidi Chen 和 Xun Huang 他们对论文早期草稿的建设性反馈和建议。我们感谢 Markus Rabe 和 Charles Staats 对他们注意力算法的有益讨论。

我们衷心感谢 NIH under No. U54EB020405 (Mobilize), NSF under Nos. CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); ARL under No. W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under No. N000141712266 (Unifying Weak Supervision); ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry in Machine Learning; N000142012275 (NEPTUNE); NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, the HAI-GCP & HAI-Azure Cloud Credits for Research program, the Stanford Data Science Initiative (SDSI), Department of Defense (DoD) through the National Defense Science and Engineering Graduate Fellowship (NDSEG) Program, and members of the Stanford DAWN project: Facebook, Google, and VMWare 的支持。美国政府被授权为政府目的复制和分发再版，

===== 第 11 页 =====

无论其上是否有任何版权标记。本材料中表达的任何意见、发现、结论或建议均为作者的观点，不一定反映 NIH、ONR 或美国政府的观点、政策或认可，无论明示或暗示。Atri Rudra 的研究得到了 NSF grant CCF-1763481 的支持。

## 参考文献

*   [1] Alok Aggarwal and S Vitter, Jeffrey. The input/output complexity of sorting and related problems. _Communications of the ACM_, 31(9):1116-1127, 1988.
*   [2] Irwan Bello. LambdaNetworks: Modeling long-range interactions without attention. _arXiv preprint arXiv:2102.08602_, 2021.
*   [3] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_, 2020.
*   [4] L Susan Blackford, Antoine Petitet, Roldan Pozo, Karin Remington, R Clint Whaley, James Demmel, Jack Dongarra, Iain Duff, Sven Hammarling, Greg Henry, et al. An updated set of basic linear algebra subprograms (blas). _ACM Transactions on Mathematical Software_, 28(2):135-151, 2002.
*   [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_, 33:1877-1901, 2020.
*   [6] Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. Neural legal judgment prediction in English. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_, pages 4317-4323, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1424. URL https://www.aclweb.org/anthology/P19-1424.
*   [7] Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and Prodromos Malakasiotis. Paragraph-level rationale extraction through regularization: A case study on european court of human rights cases. In _Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics_, Mexico City, Mexico, 2021. Association for Computational Linguistics.
*   [8] Benjamin Charlier, Jean Feydy, Joan Alexis Glaunes, Francois-David Collin, and Ghislain Durif. Kernel operations on the gpu, with autodiff, without memory overflows. _Journal of Machine Learning Research_, 22(74):1-6, 2021. URL http://jmlr.org/papers/v22/20-275.html.
*   [9] Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Re. Scatterbrain: Unifying sparse and low-rank attention. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2021.
*   [10] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. _arXiv preprint arXiv:1604.06174_, 2016.
*   [11] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. _arXiv preprint arXiv:1904.10509_, 2019.
*   [12] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In _International Conference on Learning Representations (ICLR)_, 2020.
*   [13] Xiang Dai, Ilias Chalkidis, Sune Darkner, and Desmond Elliott. Revisiting transformer-based models for long document classification. _arXiv preprint arXiv:2204.06683_, 2022.
*   [14] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_, pages 2978-2988, 2019.

===== 第 12 页 =====

*   [15] Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, and Christopher Re. Learning fast algorithms for linear transforms using butterfly factorizations. In _International Conference on Machine Learning (ICML)_, 2019.
*   [16] Tri Dao, Nimit Sohoni, Albert Gu, Matthew Eichhorn, Amit Blonder, Megan Leszczynski, Atri Rudra, and Christopher Re. Kaleidoscope: An efficient, learnable representation for all structured linear maps. In _International Conference on Learning Representations (ICLR)_, 2020.
*   [17] Tri Dao, Beidi Chen, Kaizhao Liang, Jiaming Yang, Zhao Song, Atri Rudra, and Christopher Re. Pixelated butterfly: Simple and efficient sparse training for neural network models. In _International Conference on Learning Representations (ICLR)_, 2022.
*   [18] Tri Dao, Beidi Chen, Nimit Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, and Christopher Re. Monarch: Expressive structured matrices for efficient and accurate training. In _International Conference on Machine Learning (ICML)_, 2022.
*   [19] Giannis Daras, Nikita Kitaev, Augustus Odena, and Alexandros G Dimakis. Smyrf-efficient attention using asymmetric clustering. _Advances in Neural Information Processing Systems_, 33:6476-6489, 2020.
*   [20] Christopher De Sa, Albert Gu, Rohan Puttagunta, Christopher Re, and Atri Rudra. A two-pronged progress in structured dense matrix vector multiplication. In _Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms_, pages 1060-1079. SIAM, 2018.
*   [21] Peter J Denning. The working set model for program behavior. _Communications of the ACM_, 11(5): 323-333, 1968.
*   [22] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. 2019.
*   [23] Xin Dong, Shangyu Chen, and Sinno Jialin Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. _arXiv preprint arXiv:1705.07565_, 2017.
*   [24] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_, 2020.
*   [25] Y Eidelman and I Gohberg. On a new class of structured matrices. _Integral Equations and Operator Theory_, 34(3):293-324, 1999.
*   [26] Jean Feydy, Joan Glaunes, Benjamin Charlier, and Michael Bronstein. Fast geometric learning with symbolic matrices. _Advances in Neural Information Processing Systems_, 33, 2020.
*   [27] Jorg Flum and Martin Grohe. _Parameterized Complexity Theory_. Springer, 2006.
*   [28] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In _International Conference on Learning Representations_, 2018.
*   [29] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin. Stabilizing the lottery ticket hypothesis. _arXiv preprint arXiv:1903.01611_, 2019.
*   [30] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In _International Conference on Machine Learning_, pages 3259-3269. PMLR, 2020.
*   [31] Karan Goel, Albert Gu, Chris Donahue, and Christopher Re. It's raw! audio generation with state-space models. In _International Conference on Machine Learning (ICML)_, 2022.
*   [32] Aaron Gokaslan, Vanya Cohen, Pavlick Ellie, and Stefanie Tellex. Openwebtext corpus, 2019

===== 第 13 页 =====

*   [33] Jim Gray, Surajit Chaudhuri, Adam Bosworth, Andrew Layman, Don Reichart, Murali Venkatrao, Frank Pellow, and Hamid Pirahesh. Data cube: A relational aggregation operator generalizing group-by, cross-tab, and sub-totals. _Data mining and knowledge discovery_, 1(1):29-53, 1997.
*   [34] Andreas Griewank and Andrea Walther. _Evaluating derivatives: principles and techniques of algorithmic differentiation_. SIAM, 2008.
*   [35] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Re. Hippo: Recurrent memory with optimal polynomial projections. In _Advances in neural information processing systems (NeurIPS)_, 2020.
*   [36] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Re. Combining recurrent, convolutional, and continuous-time models with linear state space layers. _Advances in Neural Information Processing Systems_, 34, 2021.
*   [37] Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces. In _The International Conference on Learning Representations (ICLR)_, 2022.
*   [38] Song Han, Jeff Pool, John Tran, and William J Dally. Learning both weights and connections for efficient neural networks. _arXiv preprint arXiv:1506.02626_, 2015.
*   [39] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In _International Conference on Learning Representations_, 2016.
*   [40] John Hennessy and David Patterson. Memory hierarchy design. _Computer Architecture: A Quantitative Approach_, pages 390-525, 2003.
*   [41] Sara Hooker. The hardware lottery. _arXiv preprint arXiv:2009.06489_, 2020.
*   [42] Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V Le. Transformer quality in linear time. _arXiv preprint arXiv:2202.10447_, 2022.
*   [43] Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shigang Li, and Torsten Hoefler. Data movement is all you need: A case study on optimizing transformers. _Proceedings of Machine Learning and Systems_, 3:711-732, 2021.
*   [44] Zhe Jia and Peter Van Sandt. Dissecting the Ampere GPU architecture via microbenchmarking. GPU Technology Conference, 2021.
*   [45] Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza. Dissecting the nvidia Volta GPU architecture via microbenchmarking. _arXiv preprint arXiv:1804.06826_, 2018.
*   [46] Zhe Jia, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. Dissecting the grapheore IPU architecture via microbenchmarking. _arXiv preprint arXiv:1912.03413_, 2019.
*   [47] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. _Scientific data_, 3(1):1-9, 2016.
*   [48] Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. In-datacenter performance analysis of a tensor processing unit. In _Proceedings of the 44th annual international symposium on computer architecture_, pages 1-12, 2017.
*   [49] Thomas Kailath, Sun-Yuan Kung, and Martin Morf. Displacement ranks of matrices and linear equations. _Journal of Mathematical Analysis and Applications_, 68(2):395-407, 1979.
*   [50] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Francois Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In _International Conference on Machine Learning_, pages 5156-5165. PMLR, 2020

===== 第 14 页 =====

*   [51] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In _The International Conference on Machine Learning (ICML)_, 2020.
*   [52] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite BEDRT for self-supervised learning of language representations. In _The International Conference on Learning Representations (ICLR)_, 2020.
*   [53] Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. The deep learning compiler: A comprehensive survey. _IEEE Transactions on Parallel and Distributed Systems_, 32(3):708-727, 2020.
*   [54] Valerii Likhosherstov, Krzysztof Choromanski, Jared Davis, Xingyou Song, and Adrian Weller. Sub-linear memory: How to make performers slim. _arXiv preprint arXiv:2012.11346_, 2020.
*   [55] Ji Lin, Yongming Rao, Jiwen Lu, and Jie Zhou. Runtime neural pruning. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems_, volume 30. Curran Associates, Inc., 2017.
*   [56] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_, 2019.
*   [57] Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer. Luna: Linear unified nested attention. _Advances in Neural Information Processing Systems_, 34, 2021.
*   [58] Peter Mattson, Christine Cheng, Gregory Diamos, Cody Coleman, Paulius Micikevicius, David Patterson, Hanlin Tang, Gu-Yeon Wei, Peter Bailis, Victor Bittorf, et al. Mlperf training benchmark. _Proceedings of Machine Learning and Systems_, 2:336-349, 2020.
*   [59] Frank McSherry, Michael Isard, and Derek G Murray. Scalability! but at what {COST}? In _15th Workshop on Hot Topics in Operating Systems (HotOS XV)_, 2015.
*   [60] Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. _arXiv preprint arXiv:1805.02867_, 2018.
*   [61] NVIDIA. Nvidia Tesla V100 GPU architecture, 2017.
*   [62] NVIDIA. Nvidia A100 tensor core GPU architecture, 2020.
*   [63] NVIDIA. Nvidia H100 tensor core GPU architecture, 2022.
*   [64] D Stott Parker. Random butterfly transformations with applications in computational linear algebra. 1995.
*   [65] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems_, 32, 2019.
*   [66] Markus N Rabe and Charles Staats. Self-attention does not need \(O(n^{2})\) memory. _arXiv preprint arXiv:2112.05682_, 2021.
*   [67] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_, 1(8):9, 2019.
*   [68] Jack Rae and Ali Razavi. Do transformers need deep long-range memory? In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_, Online, July 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.acl-main.672.
*   [69] Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap. Compressive transformers for long-range sequence modelling. In _The International Conference on Learning Representations (ICLR)_, 2020

===== 第 15 页 =====

*   [70] Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Fredo Durand, and Saman Amarasinghe. Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. _Acm Sigplan Notices_, 48(6):519-530, 2013.
*   [71] Raghu Ramakrishnan, Johannes Gehrke, and Johannes Gehrke. _Database management systems_, volume 3. McGraw-Hill New York, 2003.
*   [72] Benjamin Recht and Christopher Re. Parallel stochastic gradient algorithms for large-scale matrix completion. _Mathematical Programming Computation_, 5(2):201-226, 2013.
*   [73] Hongyu Ren, Hanjun Dai, Zihang Dai, Mengjiao Yang, Jure Leskovec, Dale Schuurmans, and Bo Dai. Combiner: Full attention transformer with sparse computation cost. _Advances in Neural Information Processing Systems_, 34, 2021.
*   [74] Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. _Transactions of the Association for Computational Linguistics_, 9:53-68, 2021.
*   [75] Amit Sabne. XLA: Compiling machine learning for peak performance. 2020.
*   [76] Victor Sanh, Thomas Wolf, and Alexander M Rush. Movement pruning: Adaptive sparsity by fine-tuning. _arXiv preprint arXiv:2005.07683_, 2020.
*   [77] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_, 2019.
*   [78] Vikas Sindhwani, Tara Sainath, and Sanjiv Kumar. Structured transforms for small-footprint deep learning. In _Advances in Neural Information Processing Systems_, pages 3088-3096, 2015.
*   [79] Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin. Adaptive attention span in transformers. In _Proceedings of the Annual Meeting of the Association for Computational Linguistics_, 2019.
*   [80] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers. In _International Conference on Learning Representations_, 2020.
*   [81] Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey. _arXiv preprint arXiv:2009.06732_, 2020.
*   [82] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_, 30, 2017.
*   [83] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers. _arXiv preprint arXiv:2203.00555_, 2022.
*   [84] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. _arXiv preprint arXiv:2006.04768_, 2020.
*   [85] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insightful visual performance model for multicore architectures. _Communications of the ACM_, 52(4):65-76, 2009.
*   [86] Michael E Wolf and Monica S Lam. A data locality optimizing algorithm. In _Proceedings of the ACM SIGPLAN 1991 conference on Programming language design and implementation_, pages 30-44, 1991

===== 第 16 页 =====

*   [87] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_, pages 38-45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.emnlp-demos.6.
*   [88] David P Woodruff. Optimal space lower bounds for all frequency moments. In _SODA_, volume 4, pages 167-175. Citeseer, 2004.
*   [89] Felix Wu, Angela Fan, Alexei Baevski, Yann N Dauphin, and Michael Auli. Pay less attention with lightweight and dynamic convolutions. In _The International Conference on Learning Representations (ICLR)_, 2019.
*   [90] Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nystromformer: A nystom-based algorithm for approximating self-attention. In _Proceedings of the AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence_, volume 35, page 14138, 2021.
*   [91] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 558-567, 2021.
*   [92] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. _Advances in Neural Information Processing Systems_, 33, 2020.
*   [93] Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, and Josh Susskind. An attention free transformer. _arXiv preprint arXiv:2105.14103_, 2021.
*   [94] Chen Zhu, Wei Ping, Chaowei Xiao, Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and Bryan Catanzaro. Long-short transformer: Efficient transformers for language and vision. _Advances in Neural Information Processing Systems_, 34, 2021.

===== 第 17 页 =====

A 相关工作

**IO 感知运行时优化。** 优化快慢内存读写的广泛概念在计算机科学中有着悠久的历史，并以许多名称而闻名。我们在这项工作中与分析 I/O 复杂度的文献 [1] 建立了最直接的联系，但内存层次结构的概念是基础性的，并以多种形式出现，从工作集模型 [21]，到数据局部性 [86]，到算术强度的 Roofline 模型 [85]，到可扩展性分析 [59]，到计算机体系结构的标准教科书处理 [40]。我们希望这项工作能鼓励社区在深度学习的更多部分采用这些思想。

**具有结构化矩阵的高效 ML 模型。** 矩阵乘法是大多数机器学习模型的核心计算瓶颈。为了降低计算复杂度，有许多方法可以在更高效的矩阵集合上学习。这些矩阵称为 **结构化矩阵**，它们具有次二次方（对于 \(n\times n\) 维度为 \(\rho(n^{2})\)）的参数数量和运行时间。最常见的结构化矩阵示例是稀疏和低秩矩阵，以及信号处理中常见的快速变换（傅里叶、切比雪夫、正弦/余弦、正交多项式）。在机器学习中已经提出了几种更通用的结构化矩阵类别：Toeplitz-like [78]、低位移秩 [49]、拟可分 [25]）。我们用于块稀疏注意力的蝴蝶模式是基于以下事实：蝴蝶矩阵 [15, 64] 及其乘积已被证明能够以几乎最优的运行时间和参数数量表示任何结构化矩阵 [16, 20]。然而，尽管结构化矩阵在理论上是高效的，但它们尚未得到广泛采用，因为很难将其效率转化为实际时钟速度的提升，因为密集无约束矩阵乘法具有高度优化的实现，这种现象称为硬件彩票 [41]。蝴蝶矩阵的扩展 [17, 18] 旨在使蝴蝶矩阵对硬件更友好。

**稀疏训练。** 我们的块稀疏 FlashAttention 可以被视为使稀疏模型训练更高效的一步。稀疏模型在通过稀疏化权重矩阵来压缩模型以进行推理（剪枝）方面取得了成功 [23, 38, 39, 55, 76]。对于模型训练，彩票假设 [28, 29, 30] 表明存在一组从更大的密集网络衍生的小子网络，其性能与原始密集网络一样好。我们的块稀疏 FlashAttention 也可以被视为注意力上下文中的固定彩票：我们在训练期间将稀疏模式固定为蝴蝶模式，并观察到它在长距离竞技场任务上的表现几乎与（密集的）FlashAttention 一样好。

**高效 Transformer。** 基于 Transformer 的模型已成为自然语言处理 [22] 和计算机视觉 [24, 91] 中最广泛使用的架构。然而，它们的计算瓶颈之一是它们的时间和内存随序列长度呈二次方缩放。有许多方法可以克服这个瓶颈，包括使用哈希（即稀疏）进行近似，例如 Reformer [51] 和 Smyrf [19]，以及使用低秩近似，例如 Performer [12, 54]。甚至可以结合稀疏和低秩近似以获得更好的准确性（例如，Longformer [3]、BigBird [92]、Scatterbrain [9]、Long-short transformer [94]、Combiner [73]）。其他方法包括沿序列维度压缩以一次关注多个令牌 [52, 57, 79, 89]。还可以关注来自先前序列的状态以帮助延长上下文（例如，Transformer-XL [14] 和 Compressive Transformer [69]）。我们推荐综述 [81] 以获取更多细节。

有几项工作致力于开发其他模块来代替注意力来建模更长的上下文。HiPPO [35] 及其扩展，最著名的是 S4 [31, 36, 37] 将历史投影到多项式基上，允许通过状态空间模型准确重建历史。它们结合了 CNN（高效训练）、RNN（高效推理）和连续模型（对采样率变化鲁棒）的优点。LambdaNetworks [2]、AFT [93] 和 FLASH [42] 是在图像分类和语言建模上下文中替换注意力的其他尝试。

## 附录 B 算法细节

我们首先推导注意力的前向和反向传播，并表明它们可以以内存高效的方式计算（需要线性而非序列长度二次方的额外内存）。尽管它们减少了所需的额外内存量，但朴素地实现仍然会产生二次 HBM 访问，导致执行速度较慢。我们描述了 FlashAttention 算法，以在 GPU 上实现前向

===== 第 18 页 =====

和反向传播，减少 HBM 访问，从而带来更快的运行时间和更小的内存占用。

### B.1 内存高效的前向传播

使注意力内存高效的主要挑战是 softmax 耦合了 \(\mathbf{K}\) 的列（以及 \(\mathbf{V}\) 的列）。我们的方法是分别计算 softmax 归一化常数以解耦列。这种技术 [60] 已在文献 [51, 66] 中使用，以表明注意力计算不需要二次方的 **额外** 内存（尽管 HBM 访问次数仍然是二次方，导致运行速度慢）。

为简单起见，我们在此省略了 softmax 过程中的最大值偏移步骤。附录 B.3 中的完整算法包含所有步骤。

回想一下，给定输入序列 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\)，我们想要计算注意力输出 \(\mathbf{O}\in\mathbb{R}^{N\times d}\)：

\[\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad \mathbf{P}=\text{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad \mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d}.\]

我们有 \(S_{ij}=q^{T}_{i} k_{j}\)，其中 \(q_{i}\) 和 \(k_{j}\) 分别是 \(\mathbf{Q}\) 和 \(\mathbf{K}\) 的第 \(i\) 和第 \(j\) 列。定义 softmax 的归一化常数：

\[L_{i}=\sum_{j}e^{q^{T}_{i} k_{j}}。\tag{1}\]

令 \(v_{j}\) 为 \(\mathbf{V}\) 的第 \(j\) 列，则输出的第 \(i\) 列为

\[o_{i}=P_{i}.\mathbf{V}=\sum_{j}P_{ij}v_{j}=\sum_{j}\frac{e^{q^{T}_{i} k_{j}}}{L_{i}}v_{j}。\tag{2}\]

我们看到，一旦计算出 \(L_{i}\)，我们就可以通过重复求和 \(e^{q^{T}_{i} k_{j}}_{L_{i}}v_{j}\) 来计算 \(o_{i}\) 而无需额外内存。因此，前向传播可以用 \(O(n)\) 的额外内存计算：

1.  根据方程 (1) 计算所有 \(i\) 的 \(L_{i}\)，这需要 \(O(n)\) 的额外内存。
2.  根据方程 (2) 计算所有 \(i\) 的 \(o_{i}\)，这需要 \(O(d)\) 的额外内存。

### B.2 内存高效的反向传播

我们推导注意力的反向传播，并表明它也可以用线性内存计算。Rabe 和 Staats [66] 建议通过对内存高效的前向传播应用梯度检查点来完成反向传播，而无需二次额外内存。我们反而显式地推导反向传播，并展示如何以内存高效的方式计算它。

假设有一个标量损失函数 \(\phi\)，并令输出梯度为 \(\mathbf{dO}\in\mathbb{R}^{n\times d}\)（其中 \(\mathbf{dO}\) 表示 \(\frac{\partial\phi}{\partial\mathbf{O}}\)）。我们想要计算输入梯度 \(\mathbf{dQ},\mathbf{dK},\mathbf{dV}\in\mathbb{R}^{n\times d}\)（其中 \(\mathbf{dQ},\mathbf{dK},\mathbf{dV}\) 分别表示 \(\frac{\partial\phi}{\partial\mathbf{Q}},\frac{\partial\phi}{\partial\mathbf{K}},\frac{\partial\phi}{\partial\mathbf{V}}\)）。

梯度 \(\mathbf{dV}\) 很容易看出。手动应用反向模式自动微分（即链式法则），我们得到（矩阵表示法）\(\mathbf{dV}=\mathbf{P}^{T}\mathbf{dO}\)。因此：

\[dv_{j}=\sum_{i}P_{ij}do_{i}=\sum_{i}\frac{e^{q^{T}_{i} k_{j}}}{L_{i}} do_{i}。\tag{3}\]

由于我们已经计算了 \(L_{i}\)，\(dv_{j}\) 可以通过重复求和计算，无需额外内存。

梯度 \(\mathbf{dQ}\) 和 \(\mathbf{dK}\) 稍微复杂一些。我们首先讨论梯度 \(\mathbf{dP}\) 和 \(\mathbf{dS}\)。从方程 (2)，我们有 \(\mathbf{dP}=\mathbf{dOV}^{T}\)，所以：

\[dP_{ij}=do^{T}_{i}v_{j}.\]

回想 \(P_{ii}=\text{softmax}(S_{ii})\)。利用 \(\mathbf{y}=\text{softmax}(x)\) 的雅可比矩阵是 \(\text{diag}(\mathbf{y})-\mathbf{y}\mathbf{y}^{T}\) 这一事实，我们有

\[dS_{ii}=(\text{diag}(P_{ii})-P_{ii}.p^{T}_{ii})dP_{ii}=P_{ii}\circ dP_{ii}-(P^{T}_{ii}.dP_{ii})P_{ii},\]

===== 第 19 页 =====

其中 \(\circ\) 表示逐点乘法。

定义

\[D_{i}=P_{i,:}^{T}dP_{i,:}=\sum_{j}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}d\sigma_{i}^{T}v_{j}=d\sigma_{i}^{T}\sum_{j}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}v_{j}=d\sigma_{i}^{T}\sigma_{i}, \tag{4}\]

那么

\[dS_{i,:}=P_{i,:}\circ dP_{i,:}-D_{i}P_{i,:}.\]

因此

\[dS_{ij}=P_{ij}dP_{ij}-D_{i}P_{ij}=P_{ij}(dP_{ij}-D_{i}).\]

现在我们可以得到梯度 **dQ** 和 **dK**。回想 \(S_{ij}=q_{i}^{T} k_{j}\)，所以

\[dq_{i}=\sum_{j}dS_{ij}k_{j}=\sum_{j}P_{ij}(dP_{ij}-D_{i})k_{j}=\sum_{j}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}(d\sigma_{i}^{T}v_{j}-D_{i})k_{j}。\tag{5}\]

类似地，

\[dk_{j}=\sum_{i}dS_{ij}q_{i}=\sum_{i}P_{ij}(dP_{ij}-D_{i})q_{i}=\sum_{i}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}(d\sigma_{i}^{T}v_{j}-D_{i})q_{i}。\tag{6}\]

因此，反向传播也可以用 \(O(n)\) 的额外内存计算：

1.  根据方程 (3) 计算所有 \(j\) 的 \(dv_{j}\)，这需要 \(O(d)\) 的额外内存。
2.  根据方程 (4) 计算所有 \(i\) 的 \(D_{i}\)，这需要 \(O(n)\) 的额外内存。
3.  根据方程 (5) 计算所有 \(i\) 的 \(dq_{i}\)，这需要 \(O(d)\) 的额外内存。
4.  根据方程 (6) 计算所有 \(j\) 的 \(dk_{j}\)，这需要 \(O(d)\) 的额外内存。

### B.3 FlashAttention：前向传播

我们描述 FlashAttention 前向传播的完整细节。给定输入序列 \({\bf Q},{\bf K},{\bf V}\in\mathbb{R}^{N\times d}\)，我们想要计算注意力输出 \({\bf O}\in\mathbb{R}^{N\times d}\)：

\[\begin{array}{ll}{\bf S}={\tau}{\bf Q}{\bf K}^{\top}\in\mathbb{R}^{N\times N},\quad {\bf S}^{\rm masked}={\rm Mask}({\bf S})\in\mathbb{R}^{N\times N},& {\bf P}={\rm softmax}({\bf S}^{\rm masked})\in\mathbb{R}^{N\times N},\\ {\bf p}^{\rm dropped}={\rm dropout}({\bf P},p_{\rm drop}),& {\bf O}={\bf p}^{\rm dropped}{\bf v}\in\mathbb{R}^{N\times d},\end{array}\]

其中 \(\tau\in\mathbb{R}\) 是某个 softmax 缩放因子（通常是 \(\frac{1}{\sqrt{d}}\)），mask 是某个掩码函数，它将输入的一些条目设置为 \(-\infty\) 并保持其他条目不变（例如，当批次中的序列长度不同并被填充时的键填充掩码），并且 \({\rm dropout}(x,p)\) 将 dropout 逐元素应用于 \(x\)（即，对于每个元素 \(x\)，以概率 \(1-p\) 输出 \(\frac{x}{1-p}\)，以概率 \(p\) 输出 \(0\)）。

完整算法在算法 2 中。我们保存输出 \({\bf O}\)、softmax 统计量 \(\ell\) 和 \(m\)，以及用于反向传播的伪随机数生成器状态 \(\mathcal{R}\)。

===== 第 20 页 =====

.

### B.4 FlashAttention：反向传播

我们描述 FlashAttention 反向传播的完整细节。给定输入序列 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\)，输出 \(\mathbf{O}\in\mathbb{R}^{N\times d}\)，以及输出梯度 \(\mathbf{dO}\)，我们想要计算输入梯度 \(\mathbf{dQ},\mathbf{dK},\mathbf{dV}\in\mathbb{R}^{N\times d}\)。

我们首先在算法 3 中描述标准注意力反向传播以供完整。

[htbp] 标准注意力反向传播
**要求：** 矩阵 \(\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{dO}\in\mathbb{R}^{N\times d}\)，\(\mathbf{P}\in\mathbb{R}^{N\times N}\) 在 HBM 中。
1.  从 HBM 按块加载 \(\mathbf{P},\mathbf{dO}\)，计算 \(\mathbf{dV}=\mathbf{P}^{\top}\mathbf{dO}\in\mathbb{R}^{N\times d}\)，将 \(\mathbf{dV}\) 写入 HBM。
2.  从 HBM 按块加载 \(\mathbf{dO},\mathbf{V}\)，计算 \(\mathbf{dP}=\mathbf{dOV}^{\top}\in\mathbb{R}^{N\times N}\)，将 \(\mathbf{dP}\) 写入 HBM。
3.  从 HBM 读取 \(\mathbf{P},\mathbf{dP}\)，计算 \(\mathbf{dS}\in\mathbb{R}^{N\times N}\)，其中 \(dS_{ij}=P_{ij}(dP_{ij}-\sum_{l}P_{il}dP_{il})\)，将 \(\mathbf{dS}\) 写入 HBM。
4.  从 HBM 按块加载 \(\mathbf{dS}\) 和 \(\mathbf{K}\)，计算 \(\mathbf{dQ}=\mathbf{dSK}\)，将 \(\mathbf{dQ}\) 写入 HBM。
5.  从 HBM 按块加载 \(\mathbf{dS}\) 和 \(\mathbf{Q}\)，计算 \(\mathbf{dK}=\mathbf{dS}^{\top}\mathbf{Q}\)，将 \(\mathbf{dK}\) 写入 HBM。
6.  返回 \(\mathbf{dQ},\mathbf{dK},\mathbf{dV}\)。

我们现在对 FlashAttention 反向传播做两个观察：

1.  我们不需要存储来自前向传播的大小为 \(O(N^{2})\) 的 dropout 掩码。相反，我们可以保存来自前向传播的伪随机数生成器状态，并在反向传播中重新生成 dropout 掩码。这使我们仅使用 \(O(N)\) 的额外内存。
2.  当计算 softmax 梯度时，我们使用方程 (4) 计算 \(D_{i}=P_{i}^{\top}dP_{i}\)：无需归约大小为 \(N\) 的 \(P_{i}:\) 和 \(dP_{i}:\)（它们可能无法放入 SRAM）。相反，我们可以重写 \(D_{i}=d\sigma_{i}^{\top}\sigma_{i}\) 并计算大小为 \(d\) 的向量之间的点积。

===== 第 21 页 =====

完整的 FlashAttention 反向传播算法在算法 4 中。概念上它只是附录 B.2 中推导的块版本。

[htbp] FlashAttention 反向传播
**要求：** 矩阵 \(\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},\mathbf{d}\mathbf{O}\in\mathbb{R}^{N\times d}\) 在 HBM 中，向量 \(\ell,m\in\mathbb{R}^{N}\) 在 HBM 中，片上 SRAM 大小为 \(M\)，softmax 缩放常数 \(\tau\in\mathbb{R}\)，掩码函数 mask，dropout 概率 \(p_{\text{drop}}\)，来自前向传播的伪随机数生成器状态 \(\mathcal{R}\)。
1.  将伪随机数生成器状态设置为 \(\mathcal{R}\)。
2.  设置块大小 \(B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)\)。
3.  将 \(\mathbf{Q}\) 分成 \(T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil\) 个块 \(\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}\)，每个块大小为 \(B_{r}\times d\)，并将 \(\mathbf{K},\mathbf{V}\) 分成 \(T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil\) 个块 \(\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}\) 和 \(\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}\)，每个块大小为 \(B_{c}\times d\)。
4.  将 \(\mathbf{O}\) 分成 \(T_{r}\) 个块 \(\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}\)，每个块大小为 \(B_{r}\times d\)，将 \(\mathbf{dO}\) 分成 \(T_{r}\) 个块 \(\mathbf{dO}_{i},\ldots,\mathbf{dO}_{T_{r}}\)，每个块大小为 \(B_{r}\times d\)，将 \(\ell\) 分成 \(T_{r}\) 个块 \(\ell_{i},\ldots,\ell_{T_{r}}\)，每个块大小为 \(B_{r}\)，将 \(m\) 分成 \(T_{r}\) 个块 \(m_{1},\ldots,m_{T_{r}}\)，每个块大小为 \(B_{r}\)。
5.  在 HBM 中初始化 \(\mathbf{dQ}=(0)_{N\times d}\)，并将其分成 \(T_{r}\) 个块 \(\mathbf{dQ}_{1},\ldots,\mathbf{dQ}_{T_{r}}\)，每个块大小为 \(B_{r}\times d\)。
6.  在 HBM 中初始化 \(\mathbf{dK}=(0)_{N\times d},\mathbf{dV}=(0)_{N\times d}\)，并将 \(\mathbf{dK},\mathbf{dV}\) 分成 \(T_{c}\) 个块 \(\mathbf{dK}_{1},\ldots,\mathbf{dK}_{T_{c}}\) 和 \(\mathbf{dV}_{1},\ldots,\mathbf{dV}_{T_{c}}\)，每个块大小为 \(B_{c}\times d\)。
7.  **for** \(1\leq j\leq T_{c}\) **do**
8.      从 HBM 加载 \(\mathbf{K}_{j}\)，\(\mathbf{V}_{j}\) 到片上 SRAM。
9.      在 SRAM 上初始化 \(\mathbf{dK}_{j}=(0)_{B_{c}\times d},\mathbf{dV}_{j}=(0)_{B_{c}\times d}\)。
10.     **for** \(1\leq i\leq T_{r}\) **do**
11.         从 HBM 加载 \(\mathbf{Q}_{i},\mathbf{O}_{i},\mathbf{dO}_{i},\mathbf{dQ}_{i},\ell_{i},m_{i}\) 到片上 SRAM。
12.         在芯片上，计算 \(\mathbf{S}_{ij}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^{T}\in\mathbb{R}^{B_{r}\times B_{c}}\)。
13.         在芯片上，计算 \(\mathbb{S}_{\text{masked}}^{\text{masked}}=\textsc{mask}(\mathbf{S}_{ij})\)。
14.         在芯片上，计算 \(\mathbf{P}_{ij}^{\text{1}}=\text{diag}(l_{i})^{-1}\exp(\mathbb{S}_{\text{masked}}^{\text{masked}}-m_{i})\in\mathbb{R}^{B_{r}\times B_{c}}\)。
15.         在芯片上，计算 dropout 掩码 \(\mathbf{Z}_{ij}\in\mathbb{R}^{B_{r}\times B_{c}}\)，其中每个条目以概率 \(1-p_{\text{drop}}\) 具有值 \(\frac{1}{1-p_{\text{drop}}}\)，以概率 \(p_{\text{drop}}\) 具有值 \(0\)。
16.         在芯片上，计算 \(\mathbf{P}_{ij}^{\text{dropped}}=\mathbf{P}_{ij}\circ \mathbf{Z}_{ij}\)（逐点乘法）。
17.         在芯片上，计算 \(\mathbf{dV}_{j}\leftarrow \mathbf{dV}_{j}+(\mathbf{P}_{ij}^{\text{dropped}})^{\tau}\mathbf{dO}_{i}\in\mathbb{R}^{B_{c}\times d}\)。
18.         在芯片上，计算 \(\mathbf{dP}_{ij}^{\text{dropped}}=\mathbf{dO}_{i}\mathbf{V}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}\)。
19.         在芯片上，计算 \(\mathbf{dP}_{ij}=\mathbf{dP}_{ij}^{\text{dropped}}\circ \mathbf{Z}_{ij}\)（逐点乘法）。
20.         在芯片上，计算 \(D_{i}=\text{rowsum}(\mathbf{dO}_{i}\circ \mathbf{O}_{i})\in\mathbb{R}^{B_{r}}\)。
21.         在芯片上，计算 \(\mathbf{dS}_{ij}=\mathbf{P}_{ij}\circ (\mathbf{dP}_{ij}-D_{i})\in\mathbb{R}^{B_{r}\times B_{c}}\)。
22.         写入 \(\mathbf{dQ}_{i}\leftarrow \mathbf{dQ}_{i}+\tau\mathbf{dS}_{ij}\mathbf{K}_{j}\in\mathbb{R}^{B_{r}\times d}\) 到 HBM。
23.         在芯片上，计算 \(\mathbf{dK}_{j}\leftarrow \mathbf{dK}_{j}+\tau\mathbf{dS}_{ij}^{\top}\mathbf{Q}_{i}\in\mathbb{R}^{B_{c}\times d}\)。
24.     **end for**
25.     写入 \(\mathbf{dK}_{j}\leftarrow \mathbf{dK}_{j},\mathbf{dV}_{j}\leftarrow \mathbf{dV}_{j}\) 到 HBM。
26. **end for**
27. 返回 \(\mathbf{dQ},\mathbf{dK},\mathbf{dV}\)。

我们看到，与前向传播类似，反向传播执行 \(O(N^{2})\) FLOPs，并且除了输入、输出、输出梯度和输入梯度之外，仅需要 \(O(N)\) 的额外内存。

我们分析了反向传播的 IO 复杂度，类似于前向传播（定理 3.1）。

**定理 5.** _设 \(N\) 为序列长度，\(d\) 为头维度，\(M\) 为 SRAM 大小，满足 \(d\leq M\leq Nd\)。标准注意力（算法 4）反向传播需要 \(\Theta(Nd+N^{2})\) 次 HBM 访问，而 FlashAttention 反向传播（算法 4）需要 \(\Theta(N^{2}d^{2}M^{-1})\) 次 HBM 访问。_

证明在附录 C 中。

===== 第 22 页 =====

B.5 与 Rabe 和 Staats [66] 的比较

我们在这里描述我们的 FlashAttention 算法与 Rabe 和 Staats [66] 的算法之间的一些相似之处和差异。

概念上，FlashAttention 和 Rabe 和 Staats [66] 都使用成熟的平铺（或 softmax 缩放）技术 [51, 60] 在注意力矩阵的块上操作。为了减少内存占用，两种方法都避免在前向传播中存储大的注意力矩阵，并在反向传播中重新计算它。

第一个主要区别是 Rabe 和 Staats [66] 专注于减少总内存占用（所需的 GPU 内存最大量），而 FlashAttention 专注于减少内存访问（内存读写的次数）。如第 2 节所述，内存访问量是运行时间的主要决定因素。减少内存访问也必然减少了所需的总内存量（例如，如果一个操作产生 \(A\) 次内存访问，那么其总内存需求最多为 \(A\)）。因此，FlashAttention 比标准注意力更快（\(2\)-\(4\times\)），而 Rabe 和 Staats [66] 的速度与标准注意力大致相同或稍慢。在总内存需求方面，两种方法都提供了显著的内存节省。

两种方法之间的第二个区别是从每个块汇总信息以传递给下一个块的方式。Rabe 和 Staats [66] 用其临时输出以及 softmax 归一化统计量来总结每个块。在前向传播结束时，所有块的临时输出使用统计量组合以产生最终输出。FlashAttention 相反在处理每个块之后增量更新输出（算法 1 第 12 行），因此只需要一个输出副本（而不是 \(K\) 个块对应 \(K\) 个副本）。这意味着 FlashAttention 与 Rabe 和 Staats [66] 相比具有更小的总内存需求。

最后一个主要区别是反向传播的计算方式。Rabe 和 Staats [66] 使用梯度检查点来重新计算注意力矩阵和每个块的临时输出。FlashAttention 相反通过分析简化了反向传播（附录 B.2 和 B.4）。它只重新计算注意力矩阵，不重新计算每个块的临时输出。这减少了反向传播的内存需求并带来了加速。

## 附录 C 证明

定理 1 的证明。我们首先计算 FLOPs 的数量和所需的额外内存。

主要的 FLOPs 来自矩阵乘法。在内层循环中（算法 1 第 9 行），我们计算 \(\mathbf{Q}_{i}\mathbf{K}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}\)，其中 \(\mathbf{Q}_{i}\in\mathbb{R}^{B_{r}\times d}\) 和 \(\mathbf{K}_{j}\in\mathbb{R}^{B_{c}\times d}\)，这需要 \(O(B_{r}B_{c}d)\) FLOPs。我们还计算（算法 1 第 12 行）\(\tilde{\mathbf{P}}_{ij}\mathbf{V}_{j}\in\mathbb{R}^{B_{r}\times d}\)，其中 \(\tilde{\mathbf{P}}_{ij}\in\mathbb{R}^{B_{r}\times B_{c}}\) 和 \(\mathbf{V}_{j}\in\mathbb{R}^{B_{c}\times d}\)，这需要 \(O(B_{r}B_{c}d)\) FLOPs。我们执行内层循环 \(T_{c}T_{r}=\left\lceil\frac{N}{B_{c}}\right\rceil\left\lceil\frac{N}{B_{r}}\right\rceil\) 次。因此总的 FLOPs 数为

\[O\left(\frac{N^{2}}{B_{c}B_{r}}B_{r}B_{c} d\right)=O(N^{2}d).\]

在额外内存需求方面，我们看到我们需要 \(O(N)\) 内存来存储统计量 (\(\ell,m\))。

我们现在通过归纳法证明算法的正确性，对 \(j\) 从 \(0\) 到 \(T_{c}\) 进行归纳。令 \(\mathbf{K}_{:j}\in\mathbb{R}^{jB_{c}\times d}\) 为 \(\mathbf{K}\) 的前 \(jB_{c}\) 行，类似地 \(\mathbf{V}_{:j}\in\mathbb{R}^{jB_{c}\times d}\) 为 \(\mathbf{V}\) 的前 \(jB_{c}\) 行。令 \(\mathbf{S}_{:,:j}=\mathbf{Q}\mathbf{K}_{:j}^{\top}\in\mathbb{R}^{N\times jB_{c}}\)，且 \(\mathbf{P}_{:,:j}=\text{softmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N\times jB_{c}}\)（softmax 按行应用）。令 \(m^{j},\ell^{(j)},\mathbf{O}^{(j)}\) 为外层循环第 \(j\) 次迭代后（算法 1 第 5 行）HBM 中 \(m,\ell,\mathbf{O}\) 的值。（注意这些 \(m,\ell,\mathbf{O}\) 的值在每次外层循环迭代后更新。）我们想证明在外层循环第 \(j\) 次迭代之后，我们在 HBM 中计算了：

\[m^{(j)}=\text{rowmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N},\quad \ell^{(j)}=\text{rowsum}(\exp(\mathbf{S}_{:,:j}-m^{(j)}))\in\mathbb{R}^{N},\quad \mathbf{O}^{(j)}=\mathbf{P}_{:,:j}\mathbf{V}_{:j}\in\mathbb{R}^{N\times d}.\]

基于我们的初始化（算法 1 第 2 行），这个主张对于 \(j=0\)（即，在外层循环任何迭代执行之前）成立。假设该主张对某个 \(j=0,\ldots,T_{c}-1\) 成立。我们想证明该主张对 \(j+1\) 也成立。确实，当我们在外层循环第 \((j+1)\) 次迭代的内层循环中更新统计量时（算法 1 第 10 行）

===== 第 23 页 =====

，我们更新 \(m^{(j+1)}=\max(m^{(j)},\tilde{m})\)，其中 \(\tilde{m}\in\mathbb{R}^{N}\) 是 \(\mathbf{S}_{:,j:j+1}\) 的行最大值，即 \(\mathbf{S}\) 从第 \(jB_{c}\) 列到第 \((j+1)B_{c}-1\) 列的切片。这意味着

\[m^{(j+1)}=\text{rowmax}(\mathbf{S}_{:,:j+1})\in\mathbb{R}^{N}.\]

类似地，我们更新

\[\ell^{(j+1)}=e^{m^{(j)}-m^{(j+1)}}\ell^{(j)}+e^{\tilde{m}-m^{(j+1)}}\tilde{\ell},\]

其中 \(\tilde{\ell}=\text{rowsum}(\exp(\mathbf{S}_{:,j:j+1}-\tilde{m}))\in\mathbb{R}^{N}\)。通过第 3.1 节中相同的代数操作，我们得到：

\[\ell^{(j+1)}=\text{rowsum}(\exp(\mathbf{S}_{:,:j+1}-m^{(j+1)}))\in\mathbb{R}^{N}.\]

令 \(\mathbf{V}_{j:j+1}\) 为 \(\mathbf{V}\) 从第 \(jB_{c}\) 列到第 \((j+1)B_{c}-1\) 列的切片，我们还更新：

\[\mathbf{O}^{(j+1)} =\text{diag}(\ell^{(j+1)})^{-1}(\text{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathbf{O}^{(j)}+e^{\tilde{m}-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1}-\tilde{m})\mathbf{V}_{j:j+1})\]
\[=\text{diag}(\ell^{(j+1)})^{-1}(\text{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathbf{P}_{:,:j}\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})\]
\[=\text{diag}(\ell^{(j+1)})^{-1}(\text{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\text{diag}(\ell^{(j)})\exp(\mathbf{S}_{:,:j}-m^{(j)})\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})\]
\[=\text{diag}(\ell^{(j+1)})^{-1}(e^{-m^{(j+1)}}\exp(\mathbf{S}_{:,:j})\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})\]
\[=\text{diag}(\ell^{(j+1)})^{-1}(\exp(\mathbf{S}_{:,:j}-m^{(j+1)})\mathbf{V}_{:j}+\exp(\mathbf{S}_{j:j+1}-m^{(j+1)})\mathbf{V}_{j:j+1})\]
\[=\text{diag}(\ell^{(j+1)})^{-1}\left(\exp\left([\mathbf{S}_{:,:j}\quad \mathbf{S}_{j:j+1}]-m^{(j+1)}\right)\right)\begin{bmatrix}\mathbf{V}_{:j}\\ \mathbf{V}_{j:j+1}\end{bmatrix}\]
\[=\text{softmax}(\mathbf{S}_{:j+1})\mathbf{V}_{:j+1}.\]

然后我们看到该主张对 \(j+1\) 也成立。通过归纳，该主张对所有 \(j=0,\ldots,T_{c}\) 成立。

当 \(j=T_{c}\) 时，我们得出结论，HBM 中 \(\mathbf{O}\) 的最终值是 \(\text{softmax}(\mathbf{S})\mathbf{V}=\text{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}\)。

定理 2 的证明。我们首先分析标准注意力实现的 IO 复杂度。输入 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\) 驻留在 HBM 中，并且在算法结束时输出 \(\mathbf{O}\in\mathbb{R}^{N\times d}\) 被写入 HBM。

在计算矩阵乘法 \(\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\) 的第一步中，输入 \(\mathbf{Q},\mathbf{K}\) 从 HBM 读取，输出 \(\mathbf{S}\in\mathbb{R}^{N\times N}\) 被写入 HBM（算法 4.2 第 4.2 行）。这产生 \(\Theta(Nd+N^{2})\) 次 HBM 访问。

在计算 \(\mathbf{P}=\text{softmax}(\mathbf{S})\) 的第二步中，输入 \(\mathbf{S}\) 从 HBM 读取，输出 \(\mathbf{P}\) 被写入 HBM（算法 4.2 第 4.2 行）。这产生 \(\Theta(N^{2})\) 次 HBM 访问。

在计算 \(\mathbf{O}=\mathbf{P}\mathbf{V}\) 的最后一步中，输入 \(\mathbf{P},\mathbf{V}\) 从全局内存读取，输出 \(\mathbf{O}\) 被写入 HBM（算法 4.2 第 4.2 行）。这产生 \(\Theta(Nd+N^{2})\) 次 HBM 访问。

总体而言，标准注意力实现需要 \(\Theta(Nd+N^{2})\) 次全局内存访问。

我们现在分析流式注意力的 IO 复杂度。

遵循算法 4.2，我们看到 \(\mathbf{K}\) 和 \(\mathbf{V}\) 的每个元素从 HBM 加载一次（算法 4.2 第 4.2 行）。我们对 \(\mathbf{Q}\) 和 \(\mathbf{O}\) 进行 \(T_{c}\) 次遍历，每次遍历加载所有 \(\mathbf{Q}\) 和所有 \(\mathbf{O}\) 到 HBM（算法 4.2 第 4.2 行）。因此 HBM 访问次数为 \(\Theta\left(Nd+NdT_{c}\right)=\Theta(NdT_{c})\)。

我们推导块大小 \(B_{c}\) 和 \(B_{r}\) 的条件。我们需要大小为 \(B_{c}\times d\) 的块 \(\mathbf{K}_{j}\) 和 \(\mathbf{V}_{j}\) 能放入片上内存，这转化为：

\[B_{c}d=O(M)\Leftrightarrow B_{c}=O\left(\frac{M}{d}\right).\]

类似地，我们需要大小为 \(B_{r}\times d\) 的块 \(\mathbf{Q}_{i},\mathbf{O}_{i}\) 能放入片上内存，这转化为：

\[B_{r}d=O(M)\Leftrightarrow B_{r}=O\left(\frac{M}{d}\right).\]

最后，我们需要大小为 \(B_{r}\times B_{c}\) 的块 \(\mathbf{S}_{ij}\) 能放入片上内存，这转化为：

\[B_{r}B_{c}=O(M).\]

===== 第 24 页 =====

因此我们设置：

\[B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\min\left(\frac{M}{d},\frac{M}{B_{c}}\right)\right)=\Theta\left(\min\left(\frac{M}{d},d\right)\right).\]

然后我们有：

\[T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{Nd}{M}\right).\]

结果，HBM 访问次数为：

\[\Theta\left(NdT_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).\]

命题 3 的证明。为了矛盾，假设存在一种算法计算精确注意力，其中对于所有 \(M\in[d,Nd]\)，HBM 访问次数为

\[o\left(\frac{N^{2}d^{2}}{M}\right).\]

在 \(M=\Theta(Nd)\) 的范围内，这导致 HBM 访问次数：

\[o\left(\frac{N^{2}d^{2}}{Nd}\right)=o(Nd).\]

然而，注意力的输入（矩阵 \(\mathbf{Q},\mathbf{K},\mathbf{V}\)）和输出 \(\mathbf{O}\) 的大小为 \(Nd\)，并且它们开始时在 HBM 中，因此如果算法计算精确注意力，它必须产生至少 \(\Omega(Nd)\) 次 HBM 访问。这是一个矛盾。

定理 5 的证明。注意力反向传播的 IO 复杂度与注意力前向传播的 IO 复杂度（定理 2）非常相似。这里我们提供证明的概要。

我们首先分析标准注意力反向传播的 IO 复杂度。输入 \(\mathbf{Q},\mathbf{K},\mathbf{V},d\mathbf{O}\in\mathbb{R}^{N\times d}\) 驻留在 HBM 中，并且在算法结束时输出 \(d\mathbf{Q},d\mathbf{K},d\mathbf{V}\in\mathbb{R}^{N\times d}\) 被写入 HBM。

在标准注意力反向传播的每一步，需要从 HBM 加载大小为 \(Nd\) 或 \(N^{2}\) 的输入，并且需要将大小为 \(N^{2}\) 或 \(Nd\) 的输出写入 HBM。这产生 \(\Theta(Nd+N^{2})\) 次 HBM 访问。

我们现在分析 FlashAttention 反向传播的 IO 复杂度。

类似于定理 2，我们看到 \(\mathbf{K}\) 和 \(\mathbf{V}\) 的每个元素从 HBM 加载一次。\(d\mathbf{K}\) 和 \(d\mathbf{V}\) 的每个元素仅写入 HBM 一次。我们对 \(\mathbf{Q},\mathbf{O},d\mathbf{O}\) 进行 \(T_{c}\) 次遍历，每次遍历加载所有 \(\mathbf{Q},\mathbf{O},d\mathbf{O}\) 到 HBM。我们还对 \(d\mathbf{Q}\) 进行 \(T_{c}\) 次遍历，每次遍历从/向 HBM 读取/写入所有 \(d\mathbf{Q}\)。因此 HBM 访问次数为 \(\Theta\left(Nd+NdT_{c}\right)=\Theta(NdT_{c})\)。

如定理 2 的证明，块大小的约束是：

\[B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\min\left(\frac{M}{d},d\right)\right).\]

然后我们有：

\[T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{Nd}{M}\right).\]

结果，HBM 访问次数为：

\[\Theta\left(NdT_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).\]

===== 第 25 页 =====

## 附录 D 扩展细节

### D.1 块稀疏 FlashAttention

我们在算法 D.1 中描述完整的块稀疏 FlashAttention 算法。该算法与算法 D.2 相同，只是我们跳过零块。

[t] 块稀疏 FlashAttention 前向传播
**要求：** 矩阵 \(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}\) 在 HBM 中，片上 SRAM 大小为 \(M\)，softmax 缩放常数 \(\tau\in\mathbb{R}\)，掩码函数 mask，dropout 概率 \(p_{\text{drop}}\)，块大小 \(B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)\)，块稀疏掩码 \(M\in\{0,1\}^{N/B_{r}\times N/B_{c}}\)。
1.  初始化伪随机数生成器状态 \(\mathcal{R}\) 并保存到 HBM。
2.  在 HBM 中初始化 \(\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}\)。
3.  将 \(\mathbf{Q}\) 分成 \(T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil\) 个块 \(\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}\)，每个块大小为 \(B_{r}\times d\)，并将 \(\mathbf{K},\mathbf{V}\) 分成 \(T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil\) 个块 \(\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}\) 和 \(\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}\)，每个块大小为 \(B_{c}\times d\)。
4.  将 \(\mathbf{O}\) 分成 \(T_{r}\) 个块 \(\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}\)，每个块大小为 \(B_{r}\times d\)，将 \(\ell\) 分成 \(T_{r}\) 个块 \(\ell_{i},\ldots,\ell_{T_{r}}\)，每个块大小为 \(B_{r}\)，将 \(m\) 分成 \(T_{r}\) 个块 \(m_{1},\ldots,m_{T_{r}}\)，每个块大小为 \(B_{r}\)。
5.  **for** \(1\leq j\leq T_{c}\) **do**
6.      从 HBM 加载 \(\mathbf{K}_{j},\mathbf{V}_{j}\) 到片上 SRAM。
7.      **for** \(1\leq i\leq T_{r}\) **do**
8.          **if** \(M_{ij}\neq 0\) **then**
9.              从 HBM 加载 \(\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}\) 到片上 SRAM。
10.             在芯片上，计算 \(\mathbf{S}_{ij}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^{T}\in\mathbb{R}^{B_{r}\times B_{c}}\)。
11.             在芯片上，计算 \(\mathbf{S}_{ij}^{\text{masked}}=\textsc{mask}(\mathbf{S}_{ij})\)。
12.             在芯片上，计算 \(\tilde{m}_{ij}=\text{rowmax}(\mathbf{S}_{ij}^{\text{masked}})\in\mathbb{R}^{B_{r}}\)，\(\tilde{\mathbf{P}}_{ij}=\exp(\mathbf{S}_{ij}^{\text{masked}}-\tilde{m}_{ij})\in\mathbb{R}^{B_{r}\times B_{c}}\)（逐点），\(\tilde{\ell}_{ij}=\text{rowsum}(\tilde{\mathbf{P}}_{ij})\in\mathbb{R}^{B_{r}}\)。
13.             在芯片上，计算 \(m_{i}^{\text{new}}=\text{max}(m_{i},\tilde{m}_{ij})\in\mathbb{R}^{B_{r}}\)，\(\ell_{i}^{\text{new}}=e^{m_{i}-m_{i}^{\text{new}}}\ell_{i}+e^{\tilde{m}_{ij}-m_{i}^{\text{new}}}\tilde{\ell}_{ij}\in\mathbb{R}^{B_{r}}\)。
14.             在芯片上，计算 \(\tilde{\mathbf{P}}_{ij}^{\text{dropped}}=\text{dropout}(\tilde{\mathbf{P}}_{ij}, p_{\text{drop}})\)。
15.             写入 \(\mathbf{O}_{i}\leftarrow \text{diag}(\ell_{i}^{\text{new}})^{-1}(\text{diag}(\ell_{i})e^{m_{i}-m_{i}^{\text{new}}} \mathbf{O}_{i}+e^{\tilde{m}_{ij}-m_{i}^{\text{new}}}\tilde{\mathbf{P}}_{ij}^{\text{dropped}}\mathbf{V}_{j})\) 到 HBM。
16.             写入 \(\ell_{i}\leftarrow \ell_{i}^{\text{new}}\)，\(m_{i}\gets m_{i}^{\text{new}}\) 到 HBM。
17.         **end if**
18.     **end for**
19. **end for**
20. 返回 \(\mathbf{O},\ell,m,\mathcal{R}\)。

命题 4 的证明。证明与定理 2 的证明非常相似。对于块稀疏情况，注意我们只需要加载对应于非零块的块。因此，HBM 访问次数按 \(s\)（块稀疏掩码中非零块的比例）缩放。然而，对于小的 \(s\) 值，我们仍然需要写入结果 \(\mathbf{O}\in\mathbb{R}^{N\times d}\)。因此 HBM 访问次数为

\[\Theta\left(Nd+\frac{N^{2}d^{2}}{M}s\right).\]

### D.2 潜在扩展

我们在这里讨论 IO 感知方法加速深度学习训练的一些潜在扩展。

**多 GPU 注意力。** 大型语言模型在数百或数千个 GPU 上训练，并且通常在同一节点上的 4-8 个 GPU 之间分割注意力计算 [77]。这引入了另一个内存层次级别：除了 GPU SRAM 和 GPU HBM，我们还有其他

===== 第 26 页 =====

GPU 的 HBM。对于非常长的序列，同一节点上的不同 GPU 可以通过考虑不同级别内存层次结构的不对称性来协作计算注意力。

**稀疏 MLP 层。** 典型的密集 MLP 层是计算受限的，而不是内存受限的。为了提高其效率，可以使用具有稀疏权重矩阵的 MLP 层 [17]。然而，许多稀疏 MLP 层反而是内存受限的，并且它们的加速通常与稀疏度不成比例。我们相信 IO 感知实现可以缓解这个问题并实现稀疏性的好处。我们对这个方向的未来工作感到兴奋，以减少大型模型的计算需求并提高其实际时钟运行时间。

**核机器学习。** 我们在 FlashAttention 中的方法依赖于 \(N\times N\) 注意力矩阵是低秩矩阵 \(\mathbf{Q}\mathbf{K}^{\top}\)（秩 \(d\ll N\)）的函数这一事实。因此，我们可以重复加载输入 \(\mathbf{Q},\mathbf{K}\) 并重新计算我们需要的注意力矩阵块，显著减少 HBM 访问。类似的情况发生在核机器学习中：\(N\times N\) 核矩阵 \(\mathbf{K}\) 的每个元素 \(K_{ij}\) 是两个大小为 \(d\ll N\) 的向量的函数，因为它测量两个数据点 \(x_{i}\) 和 \(x_{j}\) 之间的相似性。KeOps 库 [8, 26] 是一个成功的例子，说明减少内存读写可以加速核操作。我们希望这将激励更专注于减少 IO 而不仅仅是 FLOPs 的核方法。

## 附录 E 完整实验结果

### E.1 BERT

我们按照参考 MLPerf 1.1 实现的训练过程和超参数训练 BERT-large。具体来说，我们使用 LAMB 优化器，学习率为 3.75e-3，批量大小为 448，最多训练 7100 步。一旦验证准确率（用于掩码语言建模）达到目标 72.0%，训练停止，并测量实际时钟运行时间。我们使用 Apex AMP（具有 O2 优化级别）进行 FP16 精度训练。

我们将我们的结果与提交给 MLPerf 1.1 的 Nvidia 报告的训练速度进行比较（表 1）。

我们使用 MLPerf 1.1 参考实现提供的相同训练/验证数据分割。具体来说，我们在与 Nvidia 基线相同的 10000 个验证样本上进行评估。

我们在 8 个 A100-80GB GPU 上训练模型。每次训练运行需要 16 到 19 分钟，我们平均了 10 次运行的结果。

### E.2 GPT-2

我们使用来自 Huggingface transformers 库和 Nvidia 的 Megatron-LM 存储库的 GPT-2 [67] 的标准实现。我们遵循 Megatron-LM 存储库的训练配方。

我们使用有效批量大小 512，并使用梯度累积来适应可用的 GPU 内存。我们使用 AdamW 优化器，对于 GPT-2 small 学习率为 6e-4，对于 GPT-2 medium 为 1.5e-4，权重衰减为 0.1。所有模型使用相同的超参数训练 400K 步。我们使用混合精度训练（PyTorch AMP）运行所有实现。

我们使用 Openwebtext 数据集，使用 GPT-2 BPE 分词器。我们随机选择数据集的 0.5% 作为验证集，其余用作训练集。这种验证集的随机选择进行一次，所有模型在相同的验证集上进行评估。

我们在 8 个 A100-40GB GPU 上训练模型，并测量实际时钟训练时间。训练 GPT-2 small 需要 2.7-9.5 天，训练 GPT-2 medium 需要 6.9-21.0 天（表 2）。

在图 4 中，我们绘制了 GPT-2 small/medium 在整个训练过程中使用 HuggingFace 实现或我们的 FlashAttention 实现的验证困惑度。我们看到 FlashAttention 的行为与基线实现相同，并且两种实现的验证困惑度曲线几乎重叠。

**长文档分类。** 对于 MIMIC-III 和 ECtHR，我们遵循 Dai 等人 [13] 的超参数。

===== 第 27 页 =====

### E.3 LRA 细节

我们遵循来自长距离竞技场论文 [80]、长距离竞技场存储库 (https://github.com/google-research/long-range-arena) 和 Nystromformer 复现 [90] 的超参数。为了对基线方法慷慨，如果我们无法复现任何基线在任何五个任务上的性能，我们报告该基线在该任务上来自 Tay 等人 [80] 或 Xiong 等人 [90] 的更好性能。

经过超参数调整后，几乎所有的注意力方法在所有五个 LRA 任务上都达到了相似的准确率。

我们使用混合精度训练运行所有方法，除了 Performer（混合精度不稳定）和 Local Attention（实现不支持 FP16）。

为了计算总体实际时间加速，我们取每个五个任务的实际时间加速的几何平均值。

**Path-X** 对于 Path-X 和 Path-256，我们遵循来自长距离竞技场论文 [80] 的 PathFinder-32 实验的超参数。对于两者，我们首先在 Path-64 上预训练一个模型。我们在 200 个周期后获取检查点，对其位置嵌入进行上采样（我们在空间中网格状复制位置嵌入），并在下游任务上微调它 200 个周期，其中一个周期的线性预热和学习率的余弦衰减。对于 Path-X，我们取最佳性能检查点（根据验证准确率），并额外使用相同的预热和学习率微调 200 个周期（这为 Path-X 的 FlashAttention 增加了大约 4 个点的准确率，但之后模型开始过拟合）。

### E.4 与 Apex FMHA 的比较

我们将我们的方法/实现与 Apex FMHA (https://github.com/NVIDIA/apex/tree/master/Apex/contrib/csrc/fmha) 进行比较。

当我们开始这个项目时，Apex FMHA 是（我们所知的）最快的注意力实现，针对长度最多为 512 的短序列量身定制。事实上，截至 MLPerf 1.1 [58]，几乎所有在 Nvidia GPU 上运行的 BERT 训练基准的 MLPerf 提交都使用 FMHA 作为其模型代码。自从

图 4：使用两种实现的 GPT-2 small/medium 的验证困惑度。我们确认 FlashAttention 产生与来自 HuggingFace 的基线实现相同的验证曲线。

===== 第 28 页 =====

FMHA 针对 BERT 模型，它仅支持头维度 64，并且仅在 A100 GPU 上运行。FMHA 将注意力计算 dropout(softmax(Mask(\(\mathbf{QK}^{\mathsf{T}}\)))\(\mathbf{V}\) 融合到一个 CUDA 核中。在前向传播中，它将注意力矩阵 softmax(Mask(\(\mathbf{QK}^{T}\))) 存储到 HBM 以用于梯度计算。因此，它不提供显著的内存节省（尽管对于较短的序列，内存占用通常不是主要问题）。

我们使用 FMHA 代码作为起点，并应用两种成熟的技术（平铺和重新计算）来处理长序列并节省内存，如第 3 节所述。因此，我们可以支持更长的序列（例如，最多 64K 长度）。我们还支持更多的头维度（16、32、64、128）和更广泛的 GPU 类型（在撰写本文时包括所有 Turing 和 Ampere GPU）。

在表 7 中，我们比较了 FlashAttention 和 Apex FMHA 对于短序列的性能（因为 FMHA 仅支持序列长度最多 512）。通常 FlashAttention 在前向传播中比 FMHA 稍快，在反向传播中比 FMHA 稍慢。这是因为我们在前向传播中不存储注意力矩阵并在反向传播中重新计算它。与 FMHA 相比，FlashAttention 的总运行时间对于序列长度 128 大约慢 4%，对于序列长度 256 快 8%，对于序列长度 512 快 5%。

### E.5 在不同硬件和配置上的加速

加速在不同类型和世代的 GPU 之间有所不同，具体取决于 HBM 带宽和 SRAM 大小。在本节中，我们在不同 GPU 和配置上分析 FlashAttention 的加速。

**A100** 图 5 显示了在 A100 GPU 上，批量大小 8，头维度 64，12 个注意力头，跨不同序列长度的加速。我们通常看到 2-4 倍加速，并且当使用 dropout 和掩码时由于核融合我们看到更多加速。

表 7：FlashAttention 与 FMHA 按序列长度的运行时间（ms），带有掩码和 dropout，在 A100-SXM4-40GB GPU 上测量。批量大小 64，16 个头，头维度 64（即 BERT-large 大小）。

图 5：在不同序列长度上相对于标准 PyTorch 注意力的加速，在 A100 上。

===== 第 29 页 =====

**A100，头维度 128** 当我们增加头维度时，加速也会改变。每个块需要更多内存，因此我们需要使用更小的块大小以适应 SRAM。图 6 显示了在 A100 上头维度 128 的加速（批量大小 16，12 个头）。我们看到整体加速较少——但我们仍然可以看到显著的加速（高达 3 倍）与因果掩码，其中一半的块被掩码掉。

**RTX 3090** 图 7 显示了在 RTX 3090 GPU 上的加速。这里，我们使用批量大小 12 和 12 个注意力头。我们在 RTX 3090 上观察到稍高的加速（介于 2.5-4.5 倍之间），因为 RTX 3090 上的内存带宽低于 A100（大约 900 GB/s 对比 1.5 TB/s）。

**T4** 图 8 显示了在 T4 GPU 上的加速。T4 SRAM 比 A100 小，因此我们需要在 FlashAttention 中使用更小的块大小。因此，我们在 T4 上观察到较少的加速，这与第 3.2 节中的 IO 复杂度分析一致。T4 GPU 通常用于推理，因此我们还报告了仅前向传播的加速。

图 6：在不同序列长度上相对于标准 PyTorch 注意力的加速，在 A100 上，头维度 128。

图 7：在不同序列长度上相对于标准 PyTorch 注意力的加速，在 RTX 3090 上。

===== 第 30 页 =====

### E.6 完整基准测试结果

我们报告在 A100 上的完整基准测试结果和实验细节。

**基线** 我们与来自 PyTorch/HuggingFace 和 Megatron 的精确注意力参考实现、近似注意力和稀疏注意力进行比较。对于近似注意力，我们与 Reformer [51]、Local Attention [68]、Linformer Attention [84]、Smyrf [19] 和 LongShortFormer (LSPormer) [94] 的参考实现进行比较。对于稀疏注意力，我们与来自 OpenAI [11]、Longformer[3] 和 BigBird Attention [92] 的块稀疏注意力的参考实现进行比较。对于近似和稀疏注意力，我们使用 1/8 的压缩比，或压缩序列长度为 256，以较小者为准。

**设置** 我们测量具有 8 个头、维度 64、批量大小 16 的注意力计算的运行时间和内存使用情况，在一台具有一个 40 GB GPU HBM 的 A100 GPU 的机器上。我们在实验中改变序列长度。我们在随机向量上计算 \(\mathbf{Q}\)、\(\mathbf{K}\) 和 \(\mathbf{V}\) 的注意力（我们不测量从隐藏层的投影）。对于 dropout，我们使用 dropout 0.1；对于掩码，我们使用一个填充掩码，其掩码长度在总序列长度和总序列长度减 20 之间均匀随机。为了测量运行时间，我们取 100 次注意力调用测量的平均值。我们只测量一次内存占用，因为它在不同运行之间不变化。

图 8：在不同序列长度上相对于标准 PyTorch 注意力的加速，在 T4 上。**顶部：** 组合前向传播 + 反向传播。**底部：** 仅前向传播。

===== 第 31 页 =====

我们报告前向传播、反向传播以及组合前向 + 反向传播的计时结果。我们测量每种方法在有和没有 dropout、掩码或两者的情况下——除了 Block Sparse、Longformer 和 BigBird。这些方法由于外部库中的错误未能成功运行带有掩码的反向传播，因此我们测量它们没有掩码以表示慷慨。我们对所有测量使用 FP16，除了 Local Attention，其实现仅支持 FP32。

对于每个基线，我们增加序列长度直到它在 GPU 上内存不足，除了以下例外：Megatron 实现不支持超过 2048 的序列长度。Block-Sparse (OpenAI) 不支持超过 4096 的序列长度。Longformer 和 BigBird 不支持超过 8092 的序列长度。

我们在组合前向 + 反向传播上测量内存使用情况，没有 dropout 或掩码。

**结果** 表 8 总结了所有实验配置并包含指向结果表的指针。

表 8：指向结果表的指针。

表 9：各种精确/近似/稀疏注意力机制按序列长度的前向传播运行时间（ms），**带有 dropout 和掩码**。最佳以 **粗体** 显示，次佳加下划线。

===== 第 32 页 =====

表 10：各种精确/近似/稀疏注意力机制按序列长度的反向传播运行时间（ms），带有 dropout 和掩码。最佳以 **粗体** 显示，次佳加下划线。

表 11：各种精确/近似/稀疏注意力机制按序列长度的前向传播 + 反向传播运行时间（ms），带有 dropout 和掩码。最佳以 **粗体** 显示，次佳加下划线。

表 12：各种精确/近似/稀疏注意力机制按序列长度的前向传播运行时间（ms），带有掩码。最佳以 **粗体** 显示，次佳加下划线。

===== 第 33 页 =====

表 15：各种精确/近似/稀疏注意力机制按序列长度的前向传播运行时间（ms），带有 dropout。最佳以 **粗体** 显示，次佳加下划线。

表 14：各种精确/近似/稀疏注意力机制按序列长度的前向传播 + 反向传播运行时间（ms），带有掩码。最佳以 **粗体** 显示，次佳加下划线。

表 13：各种精确/近似/稀疏注意力机制按序列长度的反向传播运行时间（ms），带有掩码。最佳以 **粗体** 显示，次佳加下划线。

表 16：各种精确/近似/稀疏注意力机制按序列长度的反向传播运行时间（ms），带有 dropout。最佳以 **粗体** 显示，次佳加下划线。

表 17：各种精确/近似/稀疏注意力机制按序列长度的前向传播 + 反向传播运行时间（ms），带有 dropout。最佳以 **粗体** 显示，次佳加下划线。

===== 第 34 页 =====

表 19：各种精确/近似/稀疏注意力机制按序列长度的反向传播运行时间（ms）。最佳以 **粗体** 显示，次佳加下划线。

表 21：各种精确/近似/稀疏注意力机制按序列长度的内存使用（MB）。最佳以 **粗体** 显示，次佳加下划线。

表 18：各种精确/近似/稀疏注意力机制按序列长度的前向传播运行时间（ms）。最佳以 **粗体** 显示，次佳加下划线。

表 20：各种精确/近似/稀疏注意力机制按序列长度的前向传播 + 反向传播运行时间（ms）。最佳以 **粗体** 显示，次佳加下划线。
