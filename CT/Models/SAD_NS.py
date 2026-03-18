import torch.nn as nn
import torch
from tqdm import tqdm
import torch
        
class SAD_NS(nn.Module):
    def __init__(self,config,model,silence=True):
        super().__init__()
        self.config=config
        self.diffusion_steps = self.config["diffusion_steps"]
        self.silence=silence

        # NEW: sampling bias in [0,1]
        self.sampling_bias = float(self.config.get("sampling_bias", 0.0))

        if self.config.get("noise_schedule", "Cosine") == "Cosine":
            betas=self._cosine_variance_schedule(self.diffusion_steps)
        elif self.config.get("noise_schedule") == "Linear":
            beta_start = self.config.get("beta_start", 1e-4 * (500/self.diffusion_steps))
            beta_end   = self.config.get("beta_end", 2e-2 * (500/self.diffusion_steps))
            betas = self._linear_variance_schedule(self.diffusion_steps, beta_start=beta_start, beta_end=beta_end)
        else:
            raise ValueError("Noise Schedule not recognized.")
        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)
        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))
        self.model=model

    # NEW: biased timestep sampler
    def _sample_timesteps(self, batch_size: int, device):
        """
        sampling_bias:
          0.0 -> uniform over {0,...,T-1}
          1.0 -> always 0
          between -> more weight near 0
        """
        T = self.diffusion_steps
        b = float(self.sampling_bias)

        if b <= 0.0:
            return torch.randint(0, T, (batch_size,), device=device)

        if b >= 1.0:
            return torch.zeros((batch_size,), dtype=torch.long, device=device)

        # geometric-like weights: w_k = r^k with r in (0,1)
        r = 1.0 - b
        ks = torch.arange(T, device=device, dtype=torch.float32)
        weights = torch.pow(torch.tensor(r, device=device), ks)  # shape [T]
        return torch.multinomial(weights, num_samples=batch_size, replacement=True).long()
        
    def forward(self, x, noise):
        x_t = x[:,-1:]
        noise_t = noise[:,-1:]
        x_short_lag = x[:,-2:-1]
        noise_short_lag = noise[:,-2:-1]

        # CHANGED: use biased sampler instead of uniform randint
        s_t = self._sample_timesteps(x.shape[0], x.device)
        s_short_lag = self._sample_timesteps(x.shape[0], x.device)
        
        x_short_lag_noised = self._forward_diffusion(x_short_lag, s_short_lag, noise_short_lag)
        x_t_noised = self._forward_diffusion(x_t, s_t, noise_t)
        
        x_in = torch.cat((x_short_lag_noised, x_t_noised), dim=1)
        pred_noise, pred_s_short_lag, pred_s_t = self.model(x_in, None, None, None, True)
        
        return pred_noise, pred_s_short_lag, pred_s_t, s_short_lag, s_t

    @torch.no_grad()
    def sampling_with_fix_short_lag(self, x_short_lag):
        x_t = torch.randn_like(x_short_lag,device = x_short_lag.device)
        x_in = torch.cat((x_short_lag, x_t), dim = 1)
        
        for i in tqdm(range(self.diffusion_steps-1,-1,-1),desc="Denoising",disable=self.silence):
            s = torch.full((x_short_lag.shape[0],), i, dtype=torch.int64, device=x_short_lag.device)
            noise=torch.randn_like(x_in[:,0:1],device=x_short_lag.device)
            x_in = self._reverse_diffusion_fix_short_lag(x_in, s, noise)

        return x_in[:,-1:]

    @torch.no_grad()
    def sampling_with_refinement(self, x_short_lag):
        x_t = torch.randn_like(x_short_lag,device=x_short_lag.device)
        x_in = torch.cat((x_short_lag, x_t), dim = 1)
        pred_s_short_lag = self.model.ratio_class(x_in, head = 1)
        
        for i in tqdm(range(self.diffusion_steps-1,pred_s_short_lag-1,-1),desc="Denoising",disable=self.silence):
            s = torch.full((x_short_lag.shape[0],), i, dtype=torch.int64, device=x_short_lag.device)
            noise=torch.randn_like(x_in,device=x_short_lag.device)
            x_in = self._reverse_diffusion_with_guidence(x_in, s, noise)
        for i in tqdm(range(pred_s_short_lag-1,-1,-1),desc="Denoising",disable=self.silence):
            s = torch.full((x_short_lag.shape[0],), i, dtype=torch.int64, device=x_short_lag.device)
            noise=torch.randn_like(x_in,device=x_short_lag.device)
            x_in = self._reverse_diffusion_both(x_in, s, noise)
        return x_in

    @torch.no_grad()
    def denoising_with_refinement(self, x, forward_diff = True):
        pred_s = self.model.ratio_class_both(x)
        pred_s_t = pred_s[:,-1]
        pred_s_short_lag = pred_s[:,-2]
        x_t = x[:,-1:]
        x_short_lag = x[:,-2:-1]
        if forward_diff:
            x_t = self._forward_diffusion(x_t, pred_s_t)
            x_short_lag = self._forward_diffusion(x_short_lag, pred_s_short_lag)
            x_in = torch.cat((x_short_lag , x_t), dim = 1)
        else:
            x_in = x

        for i in tqdm(range(pred_s_t-1,pred_s_short_lag-1,-1),desc="Denoising",disable=self.silence):
            s = torch.full((x_short_lag.shape[0],), i, dtype=torch.int64, device=x.device)
            noise=torch.randn_like(x_in,device=x.device)
            x_in = self._reverse_diffusion_with_guidence(x_in, s, pred_s_short_lag, noise)
        for i in tqdm(range(pred_s_short_lag-1,-1,-1),desc="Denoising",disable=self.silence):
            s = torch.full((x_short_lag.shape[0],), i, dtype=torch.int64, device=x.device)
            noise=torch.randn_like(x_in,device=x.device)
            x_in = self._reverse_diffusion_both(x_in, s, noise)
        return x_in
        
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*torch.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas

    def _linear_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=2e-2):
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        return betas
        
    def _forward_diffusion(self,x_0,t,noise=None):
        if type(t)==int:
            t=t*torch.ones(len(x_0),device=x_0.device,dtype=torch.int64)

        if noise==None:
            noise=torch.randn_like(x_0).to(x_0.device)

        assert x_0.shape==noise.shape
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise

    def _forward_diffusion_from_to(self, x_from, t_from, t_to, noise=None):
        """
        Sample x_{t_to} given x_{t_from} for the DDPM forward process,
        assuming t_to >= t_from (more noise).
        Implements:
          x_{t_to} = sqrt(a_bar_to / a_bar_from) * x_{t_from}
                   + sqrt(1 - a_bar_to / a_bar_from) * eps
        """
        # Ensure tensors [B] int64 on correct device
        if isinstance(t_from, int):
            t_from = torch.full((x_from.shape[0],), t_from, device=x_from.device, dtype=torch.int64)
        if isinstance(t_to, int):
            t_to = torch.full((x_from.shape[0],), t_to, device=x_from.device, dtype=torch.int64)

        assert t_from.shape == t_to.shape
        assert t_from.ndim == 1 and t_to.ndim == 1

        # This helper is for adding noise only
        if not torch.all(t_to >= t_from):
            raise ValueError("Expected t_to >= t_from for forward noising jump.")

        if noise is None:
            noise = torch.randn_like(x_from)

        a_bar_from = self.alphas_cumprod.gather(-1, t_from).reshape(x_from.shape[0], 1, 1, 1)
        a_bar_to   = self.alphas_cumprod.gather(-1, t_to).reshape(x_from.shape[0], 1, 1, 1)

        ratio = a_bar_to / a_bar_from  # in (0,1]
        return torch.sqrt(ratio) * x_from + torch.sqrt(1.0 - ratio) * noise

    @torch.no_grad()
    def _reverse_diffusion_fix_short_lag(self,x_conditional, s, noise):
        x_t = x_conditional[:,-1:]
        x_short_lag = x_conditional[:,-2:-1]
        pred_x_t_noise = self.model(x_conditional, None, None, None, False)[:,-1:]
    
        alpha_s = self.alphas.gather(-1,s).reshape(x_t.shape[0],1,1,1)
        alpha_s_cumprod = self.alphas_cumprod.gather(-1,s).reshape(x_t.shape[0],1,1,1)
        beta_s = self.betas.gather(-1,s).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_s = self.sqrt_one_minus_alphas_cumprod.gather(-1,s).reshape(x_t.shape[0],1,1,1)
        mean_t = (1./torch.sqrt(alpha_s))*(x_t-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred_x_t_noise)
    
        if s.min()>0:
            alpha_s_cumprod_prev = self.alphas_cumprod.gather(-1,s-1).reshape(x_t.shape[0],1,1,1)
            std = torch.sqrt(beta_s*(1.-alpha_s_cumprod_prev)/(1.-alpha_s_cumprod))
        else:
            std = 0.0
        x_t = mean_t + std * noise[:,-1:]
        x_conditional = torch.cat((x_short_lag, x_t), dim = 1)
        return x_conditional
        
    @torch.no_grad()
    def _reverse_diffusion_with_guidence(self,x_conditional, s, s_short, noise):
        x_t_noised = x_conditional[:,-1:]
        x_short_lag = x_conditional[:,-2:-1]
        eps_short = noise[:,0:1]
        x_short_lag_noised = self._forward_diffusion_from_to(
            x_from=x_short_lag,
            t_from=s_short.view(-1),   # [B]
            t_to=s.view(-1),           # [B]
            noise=eps_short
        )

        # Replace conditional short-lag channel with aligned-noise-level version
        x_conditional = x_conditional.clone()
        x_conditional[:, 0:1] = x_short_lag_noised
        pred_x_t_noise = self.model(x_conditional, None, None, None, False)[:,-1:]
    
        alpha_s = self.alphas.gather(-1,s).reshape(x_t_noised.shape[0],1,1,1)
        alpha_s_cumprod = self.alphas_cumprod.gather(-1,s).reshape(x_t_noised.shape[0],1,1,1)
        beta_s = self.betas.gather(-1,s).reshape(x_t_noised.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_s = self.sqrt_one_minus_alphas_cumprod.gather(-1,s).reshape(x_t_noised.shape[0],1,1,1)
        mean_t = (1./torch.sqrt(alpha_s))*(x_t_noised-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred_x_t_noise)
    
        if s.min()>0:
            alpha_s_cumprod_prev = self.alphas_cumprod.gather(-1,s-1).reshape(x_t_noised.shape[0],1,1,1)
            std = torch.sqrt(beta_s*(1.-alpha_s_cumprod_prev)/(1.-alpha_s_cumprod))
        else:
            std = 0.0
        x_t = mean_t + std * noise[:,-1:]
        x_conditional = torch.cat((x_short_lag, x_t), dim = 1)
        return x_conditional

    @torch.no_grad()
    def _reverse_diffusion_both(self,x_conditional, s, noise):
        pred_noise = self.model(x_conditional, None, None, None, False)
    
        alpha_s = self.alphas.gather(-1,s).reshape(x_conditional.shape[0],1,1,1)
        alpha_s_cumprod = self.alphas_cumprod.gather(-1,s).reshape(x_conditional.shape[0],1,1,1)
        beta_s = self.betas.gather(-1,s).reshape(x_conditional.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_s = self.sqrt_one_minus_alphas_cumprod.gather(-1,s).reshape(x_conditional.shape[0],1,1,1)
        mean_t = (1./torch.sqrt(alpha_s))*(x_conditional-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred_noise)
    
        if s.min()>0:
            alpha_s_cumprod_prev = self.alphas_cumprod.gather(-1,s-1).reshape(x_conditional.shape[0],1,1,1)
            std = torch.sqrt(beta_s*(1.-alpha_s_cumprod_prev)/(1.-alpha_s_cumprod))
        else:
            std = 0.0
        x_conditional = mean_t + std * noise
        return x_conditional
