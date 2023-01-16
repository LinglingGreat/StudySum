["Denoising Diffusion Probabalistic Models"](https://arxiv.org/abs/2006.11239)

一个崩塌(corruption)的过程：在每个时间步(timestep)添加少量噪声。给定某个时间步的 $x_{t-1}$ ，可以通过下面的公式得到它的下一个版本（比之前多一点噪声） $x_t$ ：

$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad$

$q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$

也就是说， $x_{t-1}$乘以系数 $\sqrt{1 - \beta_t}$ ，加上带有系数 $\beta_t$的噪声.  $\beta$ 是根据schedule设定随着t的变化而变化的，并且决定了每个时间步要添加多少噪声。

我们不想执行这个操作500次才能得到 $x_{500}$ ，所以有另一个公式可以根据$x_0$计算出任意时刻t的 $x_t$ : 
$$\begin{aligned}

q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, {(1 - \bar{\alpha}_t)} \mathbf{I})

\end{aligned}$$ where $\bar{\alpha}_t = \prod_{i=1}^T \alpha_i$ and $\alpha_i = 1-\beta_i$


我们可以画出 $\sqrt{\bar{\alpha}_t}$ (labelled as `sqrt_alpha_prod`) 和$\sqrt{(1 - \bar{\alpha}_t)}$ (labelled as `sqrt_one_minus_alpha_prod`) 来看看输入(x)和噪声是如何在不同的时间步中量化和叠加的（scaled and mixed）

```python
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")

plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")

plt.legend(fontsize="x-large");
```

![](img/Pasted%20image%2020230115213343.png)

可以探索一下使用不同的 beta_start , beta_end 与 beta_schedule时曲线是如何变化的

```python
# One with too little noise added:

# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.004)

# The 'cosine' schedule, which may be better for small image sizes:

# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
```

可以使用`noise_scheduler.add_noise`功能来添加不同程度的噪声

```python
timesteps = torch.linspace(0, 999, 8).long().to(device)

# xb: an image array, torch.Size([img_count, 3, 32, 32])
noise = torch.randn_like(xb)

noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)

print("Noisy X shape", noisy_xb.shape)

show_images(noisy_xb).resize((8 * 64, 64), resample=Image.NEAREST)
```

