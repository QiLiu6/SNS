import torch.nn as nn
import torch
from tqdm import tqdm
import torch


class LSDM_QG(nn.Module):
    def __init__(self, config, model, silence=True):
        super().__init__()
        self.config = config
        self.diffusion_steps = self.config["diffusion_steps"]
        self.silence = silence
        if self.config.get("noise_schedule", "Cosine") == "Cosine":
            betas = self._cosine_variance_schedule(self.diffusion_steps)
        elif self.config.get("noise_schedule") == "Linear":
            beta_start = self.config.get("beta_start", 1e-4)
            beta_end = self.config.get("beta_end", 2e-2)
            betas = self._linear_variance_schedule(
                self.diffusion_steps, beta_start=beta_start, beta_end=beta_end
            )
        else:
            raise ValueError("Noise Schedule not recognized.")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.model = model

    def forward(self, x, noise, short_lag, long_lag):
        x_t = x[:, -1]
        noise_t = noise[:, -2:]
        x_short = x[:, -short_lag - 1]
        noise_short = noise[:, -4:-2]
        x_long = x[:, -long_lag - 1]
        s_short = torch.randint(0, self.diffusion_steps, (x.shape[0],), device=x.device)
        s = s_short
        if self.config.get("noise_short_lag", False):
            if self.config.get("different_noise", False):
                if self.config.get("noise_short_lag_more", False):
                    x_short = self._forward_diffusion(x_short, s_short, noise_short)
                    s = torch.where(
                        s_short > 0,
                        torch.randint_like(s_short, low=0, high=1),  # placeholder
                        torch.zeros_like(s_short),
                    )
                    u = torch.rand_like(s_short.float())
                    s = torch.where(
                        s_short > 0,
                        torch.floor(u * s_short.float()).long(),
                        torch.zeros_like(s_short),
                    )
                else:
                    x_short = self._forward_diffusion(x_short, s_short, noise_short)
                    s = torch.randint(0, self.diffusion_steps, (x.shape[0],), device=x.device)
            else:
                x_short = self._forward_diffusion(x_short, s_short, noise_t)
        x_t_noised = self._forward_diffusion(x_t, s, noise_t)
        long_lag_emb = torch.full((x.shape[0],), long_lag, dtype=torch.int64, device=x.device)
        x_in = torch.cat((x_long, x_short, x_t_noised), dim=1)
        pred_noise, pred_s_short, pred_s_t = self.model(x_in, None, None, long_lag_emb, True)
        return pred_noise, pred_s_short, pred_s_t, s_short, s
    # Add these TWO methods directly into LSDM_QG (no helper functions used).
    # They assume x is shaped like your other denoisers: [B, 3, 2, H, W]
    # where x[:,0]=long(2ch), x[:,1]=short(2ch), x[:,2]=t(2ch).
    
    @torch.no_grad()
    def denoising_down_left(
        self,
        x,                  # [B, 3, 2, H, W] : (long, short, t)
        long_lag,            # [B]
        pred_s,              # [B, >=2], pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,
    ):
        device = x.device
    
        pred_s_t = pred_s[:, -1].long().to(device)        # [B]
        pred_s_short = pred_s[:, -2].long().to(device)    # [B]
    
        x_t = x[:, -1]           # [B,2,H,W]
        x_short = x[:, -2]       # [B,2,H,W]
        x_long = x[:, -3]        # [B,2,H,W]
    
        if forward_diff:
            x_t = self._forward_diffusion(x_t, pred_s_t)
            x_short = self._forward_diffusion(x_short, pred_s_short)
            x_in = torch.cat((x_long, x_short, x_t), dim=1)  # [B,6,H,W]
        else:
            # if user passes already-clean frames, still pack to [B,6,H,W]
            x_in = torch.cat((x_long, x_short, x_t), dim=1)
    
        # ---- phase 1: denoise t first (fix short) ----
        start_s_t = int(pred_s_t.max().item())
        for i in tqdm(range(start_s_t - 1, stop, -1), desc="Denoising t", disable=self.silence):
            active = pred_s_t >= i
            if not torch.any(active):
                continue
    
            x_active = x_in[active]
            long_active = long_lag[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            noise = torch.randn_like(x_active, device=device)
    
            x_in[active] = self._reverse_diffusion_fix_short_lag(x_active, s_active, long_active, noise)
    
        # ---- phase 2: denoise short (fix t) ----
        start_s_short = int(pred_s_short.max().item())
        for i in tqdm(range(start_s_short - 1, stop, -1), desc="Denoising short", disable=self.silence):
            active = pred_s_short >= i
            if not torch.any(active):
                continue
    
            x_active = x_in[active]
            long_active = long_lag[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            noise = torch.randn_like(x_active, device=device)
    
            x_in[active] = self._reverse_diffusion_fix_long_lag(x_active, s_active, long_active, noise)
    
        # return grouped
        return x_in.reshape(x_in.shape[0], 3, 2, x_in.shape[-2], x_in.shape[-1])
    
    
    @torch.no_grad()
    def denoising_left_down(
        self,
        x,                  # [B, 3, 2, H, W] : (long, short, t)
        long_lag,            # [B]
        pred_s,              # [B, >=2], pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,
    ):
        device = x.device
    
        pred_s_t = pred_s[:, -1].long().to(device)        # [B]
        pred_s_short = pred_s[:, -2].long().to(device)    # [B]
    
        x_t = x[:, -1]           # [B,2,H,W]
        x_short = x[:, -2]       # [B,2,H,W]
        x_long = x[:, -3]        # [B,2,H,W]
    
        if forward_diff:
            x_t = self._forward_diffusion(x_t, pred_s_t)
            x_short = self._forward_diffusion(x_short, pred_s_short)
            x_in = torch.cat((x_long, x_short, x_t), dim=1)  # [B,6,H,W]
        else:
            x_in = torch.cat((x_long, x_short, x_t), dim=1)
    
        # ---- phase 1: denoise short first (fix t) ----
        start_s_short = int(pred_s_short.max().item())
        for i in tqdm(range(start_s_short - 1, stop, -1), desc="Denoising short", disable=self.silence):
            active = pred_s_short >= i
            if not torch.any(active):
                continue
    
            x_active = x_in[active]
            long_active = long_lag[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            noise = torch.randn_like(x_active, device=device)
    
            x_in[active] = self._reverse_diffusion_fix_long_lag(x_active, s_active, long_active, noise)
    
        # ---- phase 2: denoise t (fix short) ----
        start_s_t = int(pred_s_t.max().item())
        for i in tqdm(range(start_s_t - 1, stop, -1), desc="Denoising t", disable=self.silence):
            active = pred_s_t >= i
            if not torch.any(active):
                continue
    
            x_active = x_in[active]
            long_active = long_lag[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            noise = torch.randn_like(x_active, device=device)
    
            x_in[active] = self._reverse_diffusion_fix_short_lag(x_active, s_active, long_active, noise)
    
        return x_in.reshape(x_in.shape[0], 3, 2, x_in.shape[-2], x_in.shape[-1])

    
    @torch.no_grad()
    def denoising_SNS(
        self,
        x,                  # [B, 3, ...] : (long, short, t) in dim=1
        long_lag,
        pred_s,             # [B, >=2] where pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,
    ):
        device = x.device
        pred_s_t = pred_s[:, -1].long().to(device)        # [B]
        pred_s_short = pred_s[:, -2].long().to(device)    # [B]
    
        x_t = x[:, -1]
        x_short = x[:, -2]
        x_long = x[:, -3]
    
        if forward_diff:
            x_t = self._forward_diffusion(x_t, pred_s_t)
            x_short = self._forward_diffusion(x_short, pred_s_short)
            x_in = torch.cat((x_long, x_short, x_t), dim=1)
        else:
            x_in = torch.cat((x_long, x_short , x_t), dim = 1)
    
        # need to start from the max level anyone needs (either t or short)
        start_step = int(max(pred_s_t.max().item(), pred_s_short.max().item()))
    
        for i in tqdm(range(start_step - 1, stop, -1), desc="Denoising", disable=self.silence):
            # hi/lo per sample
            hi = torch.maximum(pred_s_t, pred_s_short)
            lo = torch.minimum(pred_s_t, pred_s_short)
    
            # active if this sample still has any work at step i
            active = hi >= i
            if not torch.any(active):
                continue
    
            x_active = x_in[active]
            long_active = long_lag[active]
            s_t_active = pred_s_t[active]
            s_short_active = pred_s_short[active]
    
            hi_a = torch.maximum(s_t_active, s_short_active)
            lo_a = torch.minimum(s_t_active, s_short_active)
    
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            x_active_out = x_active
    
            # region where only the higher one denoises down toward the lower:
            #   lo < i <= hi
            mask_only_region = (i > lo_a) & (i <= hi_a)
    
            # decide which one is higher in that region
            mask_only_t = mask_only_region & (s_t_active > s_short_active)
            mask_only_short = mask_only_region & (s_short_active > s_t_active)
    
            # once aligned (i <= lo), denoise both together
            mask_both = (i <= lo_a)
    
            # --- only t (bring t down to short level) ---
            if torch.any(mask_only_t):
                x_g = x_active[mask_only_t]
                s_g = s_active[mask_only_t]
                long_g = long_lag[mask_only_t]
                noise_g = torch.randn_like(x_g, device=device)
    
                x_active_out[mask_only_t] = self._reverse_diffusion_fix_short_lag(
                    x_g, s_g, long_g, noise_g
                )
    
            # --- only short (bring short down to t level) ---
            if torch.any(mask_only_short):
                x_s = x_active[mask_only_short]
                s_s = s_active[mask_only_short]
                long_s = long_lag[mask_only_short]
                noise_s = torch.randn_like(x_s, device=device)
    
                x_active_out[mask_only_short] = self._reverse_diffusion_fix_long_lag(
                    x_s, s_s, long_s, noise_s
                )
    
            # --- both (now aligned) ---
            if torch.any(mask_both):
                x_b = x_active[mask_both]
                s_b = s_active[mask_both]
                long_b = long_lag[mask_both]
                noise_b = torch.randn_like(x_b, device=device)
    
                x_active_out[mask_both] = self._reverse_diffusion_both(
                    x_b, s_b, long_b, noise_b
                )
    
            x_in[active] = x_active_out
    
        return x_in.reshape(x_in.shape[0], 3, 2, 64, 64)
    
    @torch.no_grad()
    def denoising_fixed_short_k_tweedie(
        self,
        x,                   # [B, 3, 2, H, W] or [B, 3*2, H, W] depending on forward_diff
        long_lag,            # [B] (your embedding / condition index)
        pred_s,              # [B, >=2], pred_s[:,-1]=t-level, pred_s[:,-2]=short-level (kept for API symmetry)
        k: int,              # number of reverse steps to run with fix_short
        forward_diff: bool = True,
        stop: int = -1,      # keep same semantics as your other denoisers; usually -1
        silence: bool = None,
    ):
        """
        Runs k steps of _reverse_diffusion_fix_short_lag, then uses Tweedie's formula
        to estimate x0 for the t-frame and returns (x_long, x_short, x0_hat).

        - "fix_short" means: only denoise x_t (last 2 channels), keep x_short fixed.
        - Tweedie completion is deterministic (posterior mean estimate), not ancestral sampling.
        """
        if silence is None:
            silence = self.silence
        device = x.device

        # Accept either [B,3,2,H,W] or already-concatenated [B,6,H,W]
        if x.ndim == 5:
            # [B,3,2,H,W] -> [B,6,H,W] with ordering (long, short, t)
            x_long = x[:, 0]
            x_short = x[:, 1]
            x_t = x[:, 2]
            x_in = torch.cat((x_long, x_short, x_t), dim=1)
        elif x.ndim == 4:
            # assume [B,6,H,W] already
            x_in = x
        else:
            raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")

        pred_s_t = pred_s[:, -1].long().to(device)       # [B]
        # pred_s_short = pred_s[:, -2].long().to(device) # unused here (short is fixed during reverse)

        # If requested, forward-noise x_t up to pred_s_t before denoising.
        # (We keep x_short and x_long as provided, matching your "fixed short" idea.)
        if forward_diff:
            x_long = x_in[:, -6:-4]
            x_short = x_in[:, -4:-2]
            x_t = x_in[:, -2:]
            x_t = self._forward_diffusion(x_t, pred_s_t)  # add noise to t only
            x_in = torch.cat((x_long, x_short, x_t), dim=1)

        # Per-sample target "current step" after k reverse steps
        # s_cur[b] = max(pred_s_t[b] - k, 0)
        k = int(k)
        if k < 0:
            raise ValueError("k must be nonnegative.")
        s_cur = torch.clamp(pred_s_t - k, min=0)  # [B]

        # Global loop bounds
        start_step = int(pred_s_t.max().item())
        # we will iterate i = start_step-1 ... down to stop+1 (inclusive-ish), but only apply to samples that need it.
        for i in tqdm(range(start_step - 1, stop, -1), desc="Denoising (k steps)", disable=silence):
            # A sample needs to execute reverse at step i iff:
            #   s_cur[b] < i+1 <= pred_s_t[b]
            # equivalently:
            #   i >= s_cur[b]  AND  i < pred_s_t[b]
            # but note your reverse uses "s" meaning current level; in your other denoisers you pass s=i.
            need = (i >= s_cur) & (i < pred_s_t)
            if not torch.any(need):
                continue

            x_need = x_in[need]
            long_need = long_lag[need]
            s_need = torch.full((x_need.shape[0],), i, dtype=torch.int64, device=device)
            noise_need = torch.randn_like(x_need, device=device)

            x_in[need] = self._reverse_diffusion_fix_short_lag(
                x_need, s_need, long_need, noise_need
            )

        # --- Tweedie completion ---
        # We are now at noise level s_cur for each sample (possibly 0).
        # Compute eps_theta at that level and map x_s -> x0_hat deterministically for the t-slice only.
        x_long = x_in[:, -6:-4]
        x_short = x_in[:, -4:-2]
        x_t = x_in[:, -2:]  # this is x_{s_cur} after k reverse steps

        # Predict eps for x_t at the *current* s_cur state
        eps_pred_t = self.model(torch.cat((x_long, x_short, x_t), dim=1), None, None, long_lag, False)[:, -2:]

        a_bar = self.alphas_cumprod.gather(-1, s_cur).reshape(x_in.shape[0], 1, 1, 1)
        sqrt_a_bar = torch.sqrt(a_bar)
        sqrt_one_minus_a_bar = torch.sqrt(1.0 - a_bar)

        # Tweedie / posterior mean estimate of x0
        x0_hat_t = (x_t - sqrt_one_minus_a_bar * eps_pred_t) / (sqrt_a_bar + 1e-12)

        x_out = torch.cat((x_long, x_short, x0_hat_t), dim=1)

        # Return in the same 3x2 grouped format you use elsewhere
        # (assumes channel-pair size is 2 and spatial is 64x64, but works for general H,W too)
        B, C, H, W = x_out.shape
        if C != 6:
            raise ValueError(f"Expected 6 channels (3*2), got {C}.")
        return x_out.view(B, 3, 2, H, W)
    
    @torch.no_grad()
    def denoising_ACDM(
        self,
        x,                  # [B, 3, ...] : (long, short, t) in dim=1
        long_lag,            # [B, ...] or [B, 1, ...] condition, must be batch-indexable
        pred_s,              # [B, >=2] where pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,
    ):
        device = x.device
    
        pred_s_t = pred_s[:, -1].long().to(device)       
        pred_s_short = pred_s[:, -2].long().to(device)   
    
        x_t = x[:, -1]            
        x_short = x[:, -2]      
        x_long = x[:, -3]       
        if forward_diff:
            x_t = self._forward_diffusion(x_t, pred_s_t)
            x_short = self._forward_diffusion(x_short, pred_s_short)
            x_in = torch.cat((x_long, x_short , x_t), dim = 1)
        else:
            x_in = x
    
        start_step = int(pred_s_t.max().item())
    
        for i in tqdm(range(start_step - 1, stop, -1), desc="Denoising", disable=self.silence):
            active = pred_s_t >= i
            if not torch.any(active):
                continue
            x_active = x_in[active]
            long_active = long_lag[active]
            short_active = pred_s_short[active]

            mask_guid = (short_active < i)
            mask_both = ~mask_guid
    
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)

            x_active_out = x_active

            if torch.any(mask_guid):
                x_g = x_active[mask_guid]
                s_g = s_active[mask_guid]
                long_g = long_active[mask_guid]
                short_g = short_active[mask_guid]
                noise_g = torch.randn_like(x_g, device=device)
    
                x_active_out[mask_guid] = self._reverse_diffusion_with_guidence(
                    x_g, s_g, short_g, long_g, noise_g
                )
    
            if torch.any(mask_both):
                x_b = x_active[mask_both]
                s_b = s_active[mask_both]
                long_b = long_active[mask_both]
                noise_b = torch.randn_like(x_b, device=device)  # adapt if needed
    
                x_active_out[mask_both] = self._reverse_diffusion_both(
                    x_b, s_b, long_b, noise_b
                )
            x_in[active] = x_active_out
        return x_in

    @torch.no_grad()
    def denoising_with_fixed_short(
        self,
        x,                  # [B, 3, ...] : (long, short, t) in dim=1
        long_lag,            # [B, ...] or [B, 1, ...] condition, must be batch-indexable
        pred_s,              # [B, >=2] where pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,
    ):
        device = x.device
    
        pred_s_t = pred_s[:, -1].long().to(device)       
        pred_s_short = pred_s[:, -2].long().to(device)   
    
        x_t = x[:, -1]            
        x_short = x[:, -2]      
        x_long = x[:, -3]       
        if forward_diff:
            x_t = self._forward_diffusion(x_t, pred_s_t)
            x_short = self._forward_diffusion(x_short, pred_s_short)
            x_in = torch.cat((x_long, x_short , x_t), dim = 1)
        else:
            x_in = x
    
        start_step = int(pred_s_t.max().item())
    
        for i in tqdm(range(start_step - 1, stop, -1), desc="Denoising", disable=self.silence):
            active = pred_s_t >= i
            if not torch.any(active):
                continue
            x_active = x_in[active]
            long_active = long_lag[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            x_active_out = x_active
            x_g = x_active
            s_g = s_active
            long_g = long_active
            noise_g = torch.randn_like(x_g, device=device)
            x_active_out = self._reverse_diffusion_fix_short_lag(
                x_g, s_g, long_g, noise_g
            )
            x_in[active] = x_active_out
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
    def _reverse_diffusion_fix_short_lag(self,x_conditional, s, long_lag, noise):
        x_t = x_conditional[:,-2:]
        x_short_lag = x_conditional[:,-4:-2]
        x_long = x_conditional[:,-6:-4]
        pred_x_t_noise = self.model(x_conditional, None, None, long_lag, False)[:,-2:]
    
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
        x_t = mean_t + std * noise[:,-2:]
        x_conditional = torch.cat((x_long, x_short_lag, x_t), dim = 1)
        return x_conditional
    
    @torch.no_grad()
    def _reverse_diffusion_fix_long_lag(self,x_conditional, s, long_lag, noise):
        x_t = x_conditional[:,-2:]
        x_short_lag = x_conditional[:,-4:-2]
        x_long = x_conditional[:,-6:-4]
        pred_x_short_noise = self.model(x_conditional, None, None, long_lag, False)[:,-2:]
    
        alpha_s = self.alphas.gather(-1,s).reshape(x_t.shape[0],1,1,1)
        alpha_s_cumprod = self.alphas_cumprod.gather(-1,s).reshape(x_t.shape[0],1,1,1)
        beta_s = self.betas.gather(-1,s).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_s = self.sqrt_one_minus_alphas_cumprod.gather(-1,s).reshape(x_t.shape[0],1,1,1)
        mean_short_lag = (1./torch.sqrt(alpha_s))*(x_short_lag-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred_x_short_noise)
    
        if s.min()>0:
            alpha_s_cumprod_prev = self.alphas_cumprod.gather(-1,s-1).reshape(x_t.shape[0],1,1,1)
            std = torch.sqrt(beta_s*(1.-alpha_s_cumprod_prev)/(1.-alpha_s_cumprod))
        else:
            std = 0.0
        x_short_lag = mean_short_lag + std * noise[:,-4:-2]
        x_conditional = torch.cat((x_long, x_short_lag, x_t), dim = 1)
        return x_conditional
        
    @torch.no_grad()
    def _reverse_diffusion_with_guidence(self,x_conditional, s_t, s_short, long_lag, noise):
        x_t_noised = x_conditional[:,-2:]
        x_short_lag = x_conditional[:,-4:-2]
        x_long_lag = x_conditional[:,-6:-4]

        eps_short = noise[:,-4:-2]
        x_short_lag_noised = self._forward_diffusion_from_to(
            x_from=x_short_lag,
            t_from=s_short.view(-1),   # [B]
            t_to=s_t.view(-1),           # [B]
            noise=eps_short
        )
        x_conditional = x_conditional.clone()
        x_conditional[:, -4:-2] = x_short_lag_noised
        pred_x_t_noise = self.model(x_conditional, None, None, long_lag, False)[:,-2:]
    
        alpha_s = self.alphas.gather(-1,s_t).reshape(x_t_noised.shape[0],1,1,1)
        alpha_s_cumprod = self.alphas_cumprod.gather(-1,s_t).reshape(x_t_noised.shape[0],1,1,1)
        beta_s = self.betas.gather(-1,s_t).reshape(x_t_noised.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_s = self.sqrt_one_minus_alphas_cumprod.gather(-1,s_t).reshape(x_t_noised.shape[0],1,1,1)
        mean_t = (1./torch.sqrt(alpha_s))*(x_t_noised-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred_x_t_noise)
    
        if s_t.min()>0:
            alpha_s_cumprod_prev = self.alphas_cumprod.gather(-1,s_t-1).reshape(x_t_noised.shape[0],1,1,1)
            std = torch.sqrt(beta_s*(1.-alpha_s_cumprod_prev)/(1.-alpha_s_cumprod))
        else:
            std = 0.0
        x_t = mean_t + std * noise[:,-2:]
        x_conditional = torch.cat((x_long_lag, x_short_lag, x_t), dim = 1)
        return x_conditional

    @torch.no_grad()
    def _reverse_diffusion_both(self,x_conditional, s, long_lag, noise):
        noised = x_conditional[:,-4:].clone()
        pred_noise = self.model(x_conditional, None, None, long_lag, False)
    
        alpha_s = self.alphas.gather(-1,s).reshape(x_conditional.shape[0],1,1,1)
        alpha_s_cumprod = self.alphas_cumprod.gather(-1,s).reshape(x_conditional.shape[0],1,1,1)
        beta_s = self.betas.gather(-1,s).reshape(x_conditional.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_s = self.sqrt_one_minus_alphas_cumprod.gather(-1,s).reshape(x_conditional.shape[0],1,1,1)
        mean_t = (1./torch.sqrt(alpha_s))*(noised-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred_noise)
    
        if s.min()>0:
            alpha_s_cumprod_prev = self.alphas_cumprod.gather(-1,s-1).reshape(x_conditional.shape[0],1,1,1)
            std = torch.sqrt(beta_s*(1.-alpha_s_cumprod_prev)/(1.-alpha_s_cumprod))
        else:
            std = 0.0
        x_conditional[:,-4:] = mean_t + std * noise[:,-4:]
        return x_conditional

