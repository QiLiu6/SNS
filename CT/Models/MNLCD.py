import torch.nn as nn
import torch
from tqdm import tqdm
import torch
        
class MNLCD(nn.Module):
    def __init__(self,config,model,silence=True):
        super().__init__()
        self.config=config
        
        self.diffusion_steps = self.config["diffusion_steps"]
            
        self.in_channels = self.config["input_channels"]
        self.image_width = self.config["image_width"]
        self.image_height = self.config["image_height"]

        self.silence=silence
        
        betas=self._cosine_variance_schedule(self.diffusion_steps)
        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)
        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))
        
        self.model=model

    def forward(self, x, noise):
        x_t = x[:,-1:]
        x_short = x[:,-2:-1]
        s_t = torch.randint(0,self.diffusion_steps,(x.shape[0],)).to(x.device)
        s_short = torch.randint(0,self.diffusion_steps,(x.shape[0],)).to(x.device)
        x_t_noised = self._forward_diffusion(x_t, s_t, noise[:,-1:])
        x_short_noised = self._forward_diffusion(x_short, s_short, noise[:,-2:-1])

        x_t_conditionals = torch.cat((x_short_noised, x_t_noised), dim=1)
        pred_noise = self.model(x_t_conditionals, s_short, s_t, None)
            
        return pred_noise
            
    @torch.no_grad()
    def denoising(
        self,
        x,                  # [B, 2, ...] : (short, t) in dim=1
        pred_s,             # [B, >=2] where pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,
    ):
        device = x.device
    
        # levels per sample
        pred_s_t = pred_s[:, -1].long().to(device)        # [B]
        pred_s_short = pred_s[:, -2].long().to(device)    # [B]
    
        x_t = x[:, -1:]
        x_short = x[:, -2:-1]   
    
        if forward_diff:
            noise_t = torch.randn_like(x_t, device=device)
            noise_short = torch.randn_like(x_short, device=device)
    
            x_t = self._forward_diffusion(x_t, pred_s_t, noise_t)
            x_short = self._forward_diffusion(x_short, pred_s_short, noise_short)
            x_in = torch.cat((x_short, x_t), dim=1)       # [B,2,...]
        else:
            x_in = x
    
        # start from the max level anyone needs (either t or short)
        start_step = int(max(pred_s_t.max().item(), pred_s_short.max().item()))
    
        for i in tqdm(range(start_step - 1, stop, -1), desc="Denoising", disable=self.silence):
            # hi/lo per sample
            hi = torch.maximum(pred_s_t, pred_s_short)    # [B]
            lo = torch.minimum(pred_s_t, pred_s_short)    # [B]
    
            # active if this sample still has any work at step i
            active = hi >= i
            if not torch.any(active):
                continue
    
            x_active = x_in[active]                       # [Ba,2,...]
            s_t_active = pred_s_t[active]                 # [Ba]
            s_short_active = pred_s_short[active]         # [Ba]
    
            hi_a = torch.maximum(s_t_active, s_short_active)
            lo_a = torch.minimum(s_t_active, s_short_active)
    
            # scalar step i replicated per active sample
            s_i = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)  # [Ba]
            x_active_out = x_active.clone()
    
            # region where only the higher one denoises down toward the lower: lo < i <= hi
            mask_only_region = (i > lo_a) & (i <= hi_a)
    
            # decide which one is higher in that region
            mask_only_t = mask_only_region & (s_t_active > s_short_active)
            mask_only_short = mask_only_region & (s_short_active > s_t_active)
    
            # once aligned (i <= lo), denoise both together
            mask_both = (i <= lo_a)
    
            # --- only t (bring t down to short level) ---
            if torch.any(mask_only_t):
                x_g = x_active[mask_only_t]
                s_pair = torch.stack((s_i[mask_only_t], s_i[mask_only_t]), dim=1)  # [Bg,2]
                noise_g = torch.randn_like(x_g, device=device)
                x_active_out[mask_only_t] = self._reverse_diffusion_fix_short_lag(x_g, s_pair, noise_g)
    
            # --- only short (bring short down to t level) ---
            if torch.any(mask_only_short):
                x_s = x_active[mask_only_short]
                s_pair = torch.stack((s_i[mask_only_short], s_i[mask_only_short]), dim=1)  # [Bs,2]
                noise_s = torch.randn_like(x_s, device=device)
                x_active_out[mask_only_short] = self._reverse_diffusion_fix_long_lag(x_s, s_pair, noise_s)
    
            # --- both (now aligned) ---
            if torch.any(mask_both):
                x_b = x_active[mask_both]
                s_pair = torch.stack((s_i[mask_both], s_i[mask_both]), dim=1)  # [Bb,2]
                noise_b = torch.randn_like(x_b, device=device)
                x_active_out[mask_both] = self._reverse_diffusion_both(x_b, s_pair, noise_b)
    
            x_in[active] = x_active_out

        # keep your original behavior: return only the predicted x_t
        return x_in

    @torch.no_grad()
    def denoising_t_only(
        self,
        x,                  # [B, 2, H, W] : (short, t)
        pred_s,             # [B, >=2] where pred_s[:,-1]=t-level, pred_s[:,-2]=short-level
        forward_diff=True,
        stop=-1,
    ):
        device = x.device
    
        pred_s_t = pred_s[:, -1].long().to(device)        # [B]
        pred_s_short = pred_s[:, -2].long().to(device)    # [B] (passed to model, short stays fixed)
    
        x_short = x[:, -2:-1]   # [B,1,H,W]
        x_t = x[:, -1:]         # [B,1,H,W]
    
        # Optionally forward diffuse ONLY x_t to its predicted level
        if forward_diff:
            noise_t = torch.randn_like(x_t, device=device)
            noise_short = torch.randn_like(x_short, device=device)
    
            x_t = self._forward_diffusion(x_t, pred_s_t, noise_t)
            x_short = self._forward_diffusion(x_short, pred_s_short, noise_short)
            x_in = torch.cat((x_short, x_t), dim=1)       # [B,2,...]
    
        start_step = int(pred_s_t.max().item())
    
        for i in tqdm(range(start_step - 1, stop, -1), desc="Denoising t-only", disable=self.silence):
            active = pred_s_t >= i
            if not torch.any(active):
                continue
    
            x_active = x_in[active]  # [Ba,2,H,W]
            s_i = torch.full((x_active.shape[0],), i, dtype=torch.int64, device=device)
    
            s_short_active = pred_s_short[active]
            s_pair = torch.stack((s_short_active, s_i), dim=1)  # [Ba,2]
    
            noise = torch.randn_like(x_active, device=device)
    
            # Updates ONLY x_t; short is carried through unchanged
            x_active_out = self._reverse_diffusion_fix_short_lag(x_active, s_pair, noise)
    
            x_in[active] = x_active_out
    
        # Return full (short, t) or just t depending on your preference:
        # return x_in[:, -1:]
        return x_in

        
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*torch.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise=None):
        if type(t)==int:
            t=t*torch.ones(len(x_0),device=x_0.device,dtype=torch.int64)
        if noise==None:
            noise=torch.randn_like(x_0).to(x_0.device)
        assert x_0.shape==noise.shape
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise

    @torch.no_grad()
    def _reverse_diffusion_both(self,x_conditional, s, noise):
        noised = x_conditional[:,-2:].clone()
        s_t = s[:,-1]
        s_short = s[:,-2]
        pred_noise = self.model(x_conditional, s_short, s_t, None)
    
        alpha_s_t = self.alphas.gather(-1,s_t).reshape(x_conditional.shape[0],1,1,1)
        alpha_s_t_cumprod = self.alphas_cumprod.gather(-1,s_t).reshape(x_conditional.shape[0],1,1,1)
        beta_s_t = self.betas.gather(-1,s_t).reshape(x_conditional.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_s_t = self.sqrt_one_minus_alphas_cumprod.gather(-1,s_t).reshape(x_conditional.shape[0],1,1,1)
        mean = (1./torch.sqrt(alpha_s_t))*(noised-((1.0-alpha_s_t)/sqrt_one_minus_alpha_cumprod_s_t)*pred_noise)
    
        if s.min()>0:
            alpha_s_t_cumprod_prev = self.alphas_cumprod.gather(-1,s_t-1).reshape(x_conditional.shape[0],1,1,1)
            std = torch.sqrt(beta_s_t*(1.-alpha_s_t_cumprod_prev)/(1.-alpha_s_t_cumprod))
        else:
            std = 0.0
        x_conditional[:,-2:] = mean + std * noise[:,-2:]
        return x_conditional
        
    @torch.no_grad()
    def _reverse_diffusion_fix_short_lag(self,x_conditional, s, noise):
        x_t = x_conditional[:,-1:]
        x_short = x_conditional[:,-2:-1]
        s_t = s[:,-1]
        s_short = s[:,-2]
        pred_x_t_noise = self.model(x_conditional, s_short, s_t, None)[:,-1:]
    
        alpha_s_t = self.alphas.gather(-1,s_t).reshape(x_t.shape[0],1,1,1)
        alpha_s_t_cumprod = self.alphas_cumprod.gather(-1,s_t).reshape(x_t.shape[0],1,1,1)
        beta_s_t = self.betas.gather(-1,s_t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_s_t = self.sqrt_one_minus_alphas_cumprod.gather(-1,s_t).reshape(x_t.shape[0],1,1,1)
        mean_t = (1./torch.sqrt(alpha_s_t))*(x_t-((1.0-alpha_s_t)/sqrt_one_minus_alpha_cumprod_s_t)*pred_x_t_noise)
    
        if s.min()>0:
            alpha_s_t_cumprod_prev = self.alphas_cumprod.gather(-1,s_t-1).reshape(x_t.shape[0],1,1,1)
            std = torch.sqrt(beta_s_t*(1.-alpha_s_t_cumprod_prev)/(1.-alpha_s_t_cumprod))
        else:
            std = 0.0
        x_t = mean_t + std * noise[:,-1:]
        x_conditional = torch.cat((x_short, x_t), dim = 1)
        return x_conditional

    @torch.no_grad()
    def _reverse_diffusion_fix_long_lag(self, x_conditional, s, noise):
        x_t = x_conditional[:, -1:]
        x_short = x_conditional[:, -2:-1]
    
        s_t = s[:, -1]
        s_short = s[:, -2]
    
        pred_x_short_noise = self.model(x_conditional, s_short, s_t, None)[:, -2:-1]
    
        alpha_s_short = self.alphas.gather(-1, s_short).reshape(x_short.shape[0], 1, 1, 1)
        alpha_s_short_cumprod = self.alphas_cumprod.gather(-1, s_short).reshape(x_short.shape[0], 1, 1, 1)
        beta_s_short = self.betas.gather(-1, s_short).reshape(x_short.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_s_short = self.sqrt_one_minus_alphas_cumprod.gather(-1, s_short).reshape(x_short.shape[0], 1, 1, 1)
        
        mean_short = (1.0 / torch.sqrt(alpha_s_short)) *(x_short - ((1.0 - alpha_s_short) / sqrt_one_minus_alpha_cumprod_s_short) * pred_x_short_noise)
    
        if s_short.min() > 0:
            alpha_s_short_cumprod_prev = self.alphas_cumprod.gather(-1, s_short - 1).reshape(x_short.shape[0], 1, 1, 1)
            std = torch.sqrt(beta_s_short * (1.0 - alpha_s_short_cumprod_prev) / (1.0 - alpha_s_short_cumprod))
        else:
            std = 0.0
    
        x_short = mean_short + std * noise[:, -2:-1]
        x_conditional = torch.cat((x_short, x_t), dim=1)
        return x_conditional