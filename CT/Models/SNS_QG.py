import torch.nn as nn
import torch
from tqdm import tqdm
import torch
        
class SNS_QG(nn.Module):
    def __init__(self,config,model,silence=True):
        super().__init__()
        self.config=config
        self.diffusion_steps = self.config["diffusion_steps"]
        self.silence=silence
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
        
    def forward(self, x, noise):
        x_t = x[:,-1]
        noise_t = noise[:,-2:]
        x_short_lag = x[:,-2]
        noise_short_lag = noise[:,-4:-2]
        x_long_lag = x[:, -3]
        s_t = torch.randint(0, self.diffusion_steps, (x.shape[0],), device=x.device)
        s_short_lag = torch.randint(0, self.diffusion_steps, (x.shape[0],), device=x.device)
        
        x_short_lag_noised = self._forward_diffusion(x_short_lag, s_short_lag, noise_short_lag)
        x_t_noised = self._forward_diffusion(x_t, s_t, noise_t)
        
        x_in = torch.cat((x_long_lag, x_short_lag_noised, x_t_noised), dim=1)
        pred_noise, pred_s_short_lag, pred_s_t = self.model(x_in, None, None, None, True)
        
        return pred_noise, pred_s_short_lag, pred_s_t, s_short_lag, s_t

    @torch.no_grad()
    def denoising_down_left(
        self,
        x,                  # [B, 3, ...] : (long, short, t) in dim=1
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
            x_in = torch.cat((x_long, x_short , x_t), dim = 1)
    
        start_s_t = int(pred_s_t.max().item())
    
        for i in tqdm(range(start_s_t - 1, stop, -1), desc="Denoising", disable=self.silence):
            active = pred_s_t >= i
            if not torch.any(active):
                continue
            x_active = x_in[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            x_active_out = x_active
            x_g = x_active
            s_g = s_active
            noise_g = torch.randn_like(x_g, device=device)
            x_active_out = self._reverse_diffusion_fix_short_lag(
                x_g, s_g, noise_g
            )
            x_in[active] = x_active_out

        start_s_short = int(pred_s_short.max().item())

        for i in tqdm(range(start_s_short - 1, stop, -1), desc="Denoising", disable=self.silence):
            active = pred_s_short >= i
            if not torch.any(active):
                continue
            x_active = x_in[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            x_active_out = x_active
            x_g = x_active
            s_g = s_active
            noise_g = torch.randn_like(x_g, device=device)
            x_active_out = self._reverse_diffusion_fix_long_lag(
                x_g, s_g, noise_g
            )
            x_in[active] = x_active_out
        return x_in.reshape(x_in.shape[0], 3, 2, 64, 64)

    @torch.no_grad()
    def denoising_left_down(
        self,
        x,                  # [B, 3, ...] : (long, short, t) in dim=1
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
            x_in = torch.cat((x_long, x_short , x_t), dim = 1)

        start_s_short = int(pred_s_short.max().item())

        for i in tqdm(range(start_s_short - 1, stop, -1), desc="Denoising", disable=self.silence):
            active = pred_s_short >= i
            if not torch.any(active):
                continue
            x_active = x_in[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)  
            x_active_out = x_active
            x_g = x_active
            s_g = s_active
            noise_g = torch.randn_like(x_g, device=device)
            x_active_out = self._reverse_diffusion_fix_long_lag(
                x_g, s_g, noise_g
            )
            x_in[active] = x_active_out
    
        start_s_t = int(pred_s_t.max().item())
    
        for i in tqdm(range(start_s_t - 1, stop, -1), desc="Denoising", disable=self.silence):
            active = pred_s_t >= i
            if not torch.any(active):
                continue
            x_active = x_in[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            x_active_out = x_active
            x_g = x_active
            s_g = s_active
            noise_g = torch.randn_like(x_g, device=device)
            x_active_out = self._reverse_diffusion_fix_short_lag(
                x_g, s_g, noise_g
            )
            x_in[active] = x_active_out
        
        return x_in.reshape(x_in.shape[0], 3, 2, 64, 64)
        
    @torch.no_grad()
    def denoising_with_fixed_short(
        self,
        x,                  # [B, 3, ...] : (long, short, t) in dim=1
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
            x_in = torch.cat((x_long, x_short , x_t), dim = 1)
    
        start_step = int(pred_s_t.max().item())
    
        for i in tqdm(range(start_step - 1, stop, -1), desc="Denoising", disable=self.silence):
            active = pred_s_t >= i
            if not torch.any(active):
                continue
            x_active = x_in[active]
            s_active = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
            x_active_out = x_active
            x_g = x_active
            s_g = s_active
            noise_g = torch.randn_like(x_g, device=device)
            x_active_out = self._reverse_diffusion_fix_short_lag(
                x_g, s_g, noise_g
            )
            x_in[active] = x_active_out
        return x_in.reshape(x_in.shape[0], 3, 2, 64, 64)
    @torch.no_grad()
    def denoising_SNS(
        self,
        x,                  # [B, 3, ...] : (long, short, t) in dim=1
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
                noise_g = torch.randn_like(x_g, device=device)
    
                x_active_out[mask_only_t] = self._reverse_diffusion_fix_short_lag(
                    x_g, s_g, noise_g
                )
    
            # --- only short (bring short down to t level) ---
            if torch.any(mask_only_short):
                x_s = x_active[mask_only_short]
                s_s = s_active[mask_only_short]
                noise_s = torch.randn_like(x_s, device=device)
    
                x_active_out[mask_only_short] = self._reverse_diffusion_fix_long_lag(
                    x_s, s_s, noise_s
                )
    
            # --- both (now aligned) ---
            if torch.any(mask_both):
                x_b = x_active[mask_both]
                s_b = s_active[mask_both]
                noise_b = torch.randn_like(x_b, device=device)
    
                x_active_out[mask_both] = self._reverse_diffusion_both(
                    x_b, s_b, noise_b
                )
    
            x_in[active] = x_active_out
    
        return x_in.reshape(x_in.shape[0], 3, 2, 64, 64)

    @torch.no_grad()
    def denoising_with_fixed_short(
        self,
        x,                  # [B, 3, 2, H, W] : (long, short, t) in dim=1
        pred_s,             # [B, >=2] where pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,            # stop=-1 means go down to s=0 inclusive
    ):
        device = x.device
        pred_s_t = pred_s[:, -1].long().to(device)        # [B]
        pred_s_short = pred_s[:, -2].long().to(device)    # [B]

        x_long  = x[:, 0]   # [B,2,H,W]
        x_short = x[:, 1]
        x_t     = x[:, 2]

        if forward_diff:
            # under YOUR convention, this is noisy even when pred_s_* == 0
            x_short = self._forward_diffusion(x_short, pred_s_short)
            x_t     = self._forward_diffusion(x_t, pred_s_t)

        x_in = torch.cat((x_long, x_short, x_t), dim=1)   # [B,6,H,W]

        # We want to apply reverse at s = pred_s_t, pred_s_t-1, ..., 0
        # Loop s from max down to 0, with active = pred_s_t >= s.
        start_s = int(pred_s_t.max().item())

        # If stop=-1, include s=0. If stop>=0, include s=stop+1..0? (keep same style as yours)
        # We'll interpret "stop" like your code: range(start, stop, -1) includes 0 when stop=-1.
        for s in tqdm(range(start_s, stop, -1), desc="Denoising (t only)", disable=self.silence):
            active = (pred_s_t >= s)
            if not torch.any(active):
                continue

            x_a = x_in[active]
            s_a = torch.full((x_a.shape[0],), s, dtype=torch.int64, device=device)
            noise = torch.randn_like(x_a, device=device)

            x_in[active] = self._reverse_diffusion_fix_short_lag(x_a, s_a, noise)

        B, C, H, W = x_in.shape
        return x_in.view(B, 3, 2, H, W)

    @torch.no_grad()
    def denoising(
        self,
        x,                  # [B, 3, 2, H, W] : (long, short, t) in dim=1
        pred_s,             # [B, >=2] where pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,            # stop=-1 means go down to s=0 inclusive
    ):
        """
        Coupled SNS denoising consistent with YOUR indexing:
          - forward(t=0) is noisy
          - reverse should include s=0 update (deterministic) to reach the clean estimate
        Strategy:
          loop s from max(pred_s_t, pred_s_short) down to 0
          for each sample:
            if s <= min(st, ss): denoise BOTH
            elif s <= st and s > ss: denoise t only (fix_short)
            elif s <= ss and s > st: denoise short only (fix_long)
        """
        device = x.device
        pred_s_t = pred_s[:, -1].long().to(device)
        pred_s_short = pred_s[:, -2].long().to(device)

        x_long  = x[:, 0]
        x_short = x[:, 1]
        x_t     = x[:, 2]

        if forward_diff:
            x_short = self._forward_diffusion(x_short, pred_s_short)
            x_t     = self._forward_diffusion(x_t, pred_s_t)

        x_in = torch.cat((x_long, x_short, x_t), dim=1)   # [B,6,H,W]

        hi = torch.maximum(pred_s_t, pred_s_short)        # [B]
        lo = torch.minimum(pred_s_t, pred_s_short)        # [B]
        start_s = int(hi.max().item())

        for s in tqdm(range(start_s, stop, -1), desc="Denoising (SNS)", disable=self.silence):
            # any work at this s?
            active = (hi >= s)
            if not torch.any(active):
                continue

            x_a = x_in[active]
            st_a = pred_s_t[active]
            ss_a = pred_s_short[active]
            lo_a = torch.minimum(st_a, ss_a)

            s_a = torch.full((x_a.shape[0],), s, dtype=torch.int64, device=device)
            noise = torch.randn_like(x_a, device=device)

            # region split (inclusive, consistent with "run at s=0"):
            # - both if s <= lo
            # - only t if ss < s <= st
            # - only short if st < s <= ss
            mask_both = (s_a <= lo_a)

            mask_only_t = (~mask_both) & (st_a >= s_a) & (ss_a < s_a)
            mask_only_short = (~mask_both) & (ss_a >= s_a) & (st_a < s_a)

            x_out = x_a

            if torch.any(mask_only_t):
                x_out[mask_only_t] = self._reverse_diffusion_fix_short_lag(
                    x_a[mask_only_t], s_a[mask_only_t], noise[mask_only_t]
                )

            if torch.any(mask_only_short):
                x_out[mask_only_short] = self._reverse_diffusion_fix_long_lag(
                    x_a[mask_only_short], s_a[mask_only_short], noise[mask_only_short]
                )

            if torch.any(mask_both):
                x_out[mask_both] = self._reverse_diffusion_both(
                    x_a[mask_both], s_a[mask_both], noise[mask_both]
                )

            x_in[active] = x_out

        B, C, H, W = x_in.shape
        return x_in.view(B, 3, 2, H, W)
        
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
        x_t = x_conditional[:,-2:]
        x_short_lag = x_conditional[:,-4:-2]
        x_long = x_conditional[:,-6:-4]
        pred_x_t_noise = self.model(x_conditional, None, None, None, False)[:,-2:]
    
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
    def _reverse_diffusion_fix_long_lag(self,x_conditional, s, noise):
        x_t = x_conditional[:,-2:]
        x_short_lag = x_conditional[:,-4:-2]
        x_long = x_conditional[:,-6:-4]
        pred_x_short_noise = self.model(x_conditional, None, None, None, False)[:,-4:-2]
    
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
    def _reverse_diffusion_with_guidence(self,x_conditional, s_t, s_short, noise):
        x_t_noised = x_conditional[:,-1:]
        x_short_lag = x_conditional[:,-2:-1]
        x_long_lag = x_conditional[:,-3:-2]

        eps_short = noise[:,-2:-1]
        x_short_lag_noised = self._forward_diffusion_from_to(
            x_from=x_short_lag,
            t_from=s_short.view(-1),   # [B]
            t_to=s_t.view(-1),           # [B]
            noise=eps_short
        )
        x_conditional = x_conditional.clone()
        x_conditional[:, -2:-1] = x_short_lag_noised
        pred_x_t_noise = self.model(x_conditional, None, None, None, False)[:,-1:]
    
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
        x_t = mean_t + std * noise[:,-1:]
        x_conditional = torch.cat((x_long_lag, x_short_lag, x_t), dim = 1)
        return x_conditional

    @torch.no_grad()
    def _reverse_diffusion_both(self,x_conditional, s, noise):
        noised = x_conditional[:,-4:].clone()
        pred_noise = self.model(x_conditional, None, None, None, False)
    
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