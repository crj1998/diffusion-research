import matplotlib.pyplot as plt
plt.style.use("ggplot")
import torch
from diffusers import DDPMScheduler

def prepare_scheduler_for_custom_training(noise_scheduler: DDPMScheduler, device):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    all_snr = alphas_cumprod / (1 - alphas_cumprod)

    noise_scheduler.all_snr = all_snr.to(device)

def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler: DDPMScheduler):
    # fix beta: zero terminal SNR

    def enforce_zero_terminal_snr(betas: torch.Tensor):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod

def visualize_zero_snr():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    axes[0].set_yscale('log')
    axes[2].set_yscale('log')
    axes[0].set_title(r'$\log \beta$')
    axes[1].set_title(r'$\bar{\alpha}$')
    axes[2].set_title(r'$\log \mathrm{SNR}$')

    axes[0].set_xlabel("T")
    axes[1].set_xlabel("T")
    axes[2].set_xlabel("T")
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False #, prediction_type="v_prediction"
    )
    prepare_scheduler_for_custom_training(noise_scheduler, "cpu")

    axes[0].plot(noise_scheduler.betas.cpu().numpy(), label="vanilla")
    axes[1].plot(noise_scheduler.alphas_cumprod.cpu().numpy())
    axes[2].plot(noise_scheduler.all_snr.cpu().numpy())

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False #, prediction_type="v_prediction"
    )
    fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
    prepare_scheduler_for_custom_training(noise_scheduler, "cpu")

    axes[0].plot(noise_scheduler.betas.cpu().numpy(), label="enforce zero")
    axes[1].plot(noise_scheduler.alphas_cumprod.cpu().numpy())
    axes[2].plot(noise_scheduler.all_snr.cpu().numpy())
    axes[2].hlines(5.0, xmin=0, xmax=999, color="grey", linestyles="dashed")
    axes[2].hlines(0.001, xmin=0, xmax=999, color="grey", linestyles="dashed")


    fig.legend(loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1.0, 1.0))
    plt.savefig("scheduler.jpg", dpi=160)


def visualize_weighting():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False #, prediction_type="v_prediction"
    )
    prepare_scheduler_for_custom_training(noise_scheduler, "cpu")

    baseline_weight = (1.0 - noise_scheduler.betas) * (1.0 - noise_scheduler.alphas_cumprod) / noise_scheduler.betas
    axes[0][0].plot(noise_scheduler.all_snr.cpu().numpy(), baseline_weight.cpu().numpy(), label="baseline")
    axes[0][1].plot(baseline_weight.cpu().numpy())
    normalize_weight = baseline_weight / baseline_weight.sum()
    axes[1][0].plot(noise_scheduler.all_snr.cpu().numpy(), normalize_weight.cpu().numpy())
    axes[1][1].plot(normalize_weight.cpu().numpy())

    gamma05_weight = baseline_weight / (1+noise_scheduler.all_snr)**0.5
    axes[0][0].plot(noise_scheduler.all_snr.cpu().numpy(), gamma05_weight.cpu().numpy(), label="gamma-0.5")
    axes[0][1].plot(gamma05_weight.cpu().numpy())
    normalize_weight = gamma05_weight / gamma05_weight.sum()
    axes[1][0].plot(noise_scheduler.all_snr.cpu().numpy(), normalize_weight.cpu().numpy())
    axes[1][1].plot(normalize_weight.cpu().numpy())


    gamma10_weight = baseline_weight / (1+noise_scheduler.all_snr)**1.0
    axes[0][0].plot(noise_scheduler.all_snr.cpu().numpy(), gamma10_weight.cpu().numpy(), label="gamma-1.0")
    axes[0][1].plot(gamma10_weight.cpu().numpy())
    normalize_weight = gamma10_weight / gamma10_weight.sum()
    axes[1][0].plot(noise_scheduler.all_snr.cpu().numpy(), normalize_weight.cpu().numpy())
    axes[1][1].plot(normalize_weight.cpu().numpy())

    minsnr_weight = baseline_weight * torch.minimum(noise_scheduler.all_snr, 5 * torch.ones_like(noise_scheduler.all_snr)) / noise_scheduler.all_snr
    axes[0][0].plot(noise_scheduler.all_snr.cpu().numpy(), minsnr_weight.cpu().numpy(), label="min-snr-5.0")
    axes[0][1].plot(minsnr_weight.cpu().numpy())
    normalize_weight = minsnr_weight / minsnr_weight.sum()
    axes[1][0].plot(noise_scheduler.all_snr.cpu().numpy(), normalize_weight.cpu().numpy())
    axes[1][1].plot(normalize_weight.cpu().numpy())

    axes[0][0].set_xscale('log')
    axes[0][0].minorticks_off()
    axes[1][0].set_xscale('log')
    axes[1][0].minorticks_off()

    axes[0][0].set_ylabel("un-normalized")
    axes[1][0].set_ylabel("normalized")
    axes[1][0].set_xlabel("SNR")
    axes[1][1].set_xlabel("timestep")

    fig.legend(loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.04, 1.0, 1.0))
    plt.savefig("demo.png")


if __name__ == "__main__":
    visualize_zero_snr()
    visualize_weighting()