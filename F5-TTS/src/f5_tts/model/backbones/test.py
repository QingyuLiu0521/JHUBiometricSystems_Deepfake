import torch
from f5_tts.model.backbones import dit as dit_new
from f5_tts.model.backbones import dit_ori as dit_old
from f5_tts.model.utils import lens_to_mask

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

B = 3
seq_lens = torch.tensor([80, 90, 100], device=device)
T = int(seq_lens.max().item())
nt = 40
mel_dim = 100
text_num_embeds = 512
dim = 1024

# 输入
text = torch.randint(low=-1, high=text_num_embeds, size=(B, nt), device=device)
x = torch.randn(B, T, mel_dim, device=device, dtype=dtype)
cond = torch.randn(B, T, mel_dim, device=device, dtype=dtype)
audio_mask = lens_to_mask(seq_lens, length=T).to(device)

# 两个模型，参数完全一致
m_new = dit_new.DiT(
    dim=dim, mel_dim=mel_dim, text_num_embeds=text_num_embeds,
    conv_layers=2, text_embedding_average_upsampling=True
).to(device).eval()

m_old = dit_old.DiT(
    dim=dim, mel_dim=mel_dim, text_num_embeds=text_num_embeds,
    conv_layers=2, text_embedding_average_upsampling=True
).to(device).eval()

m_old.load_state_dict(m_new.state_dict(), strict=True)

with torch.no_grad():
    # 只比 get_input_embed（你改动核心）
    y_new = m_new.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=False, audio_mask=audio_mask)
    y_old = m_old.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=False, audio_mask=audio_mask)

diff = (y_new - y_old).abs()
print("shape:", y_new.shape, y_old.shape)
print("max_abs:", diff.max().item())
print("mean_abs:", diff.mean().item())
print("allclose(atol=1e-5, rtol=1e-5):", torch.allclose(y_new, y_old, atol=1e-5, rtol=1e-5))