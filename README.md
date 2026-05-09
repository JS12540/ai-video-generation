# ai-video-generation
An effort to use open source AI video generation for free for content creation.

Now I have everything I need. Here's a comprehensive breakdown:Click any card above to get the full Colab code for that model. Here's the full breakdown of what I found:

---

**Tier 1 — Confirmed working on free T4:**

**CogVideoX-2B** is officially confirmed to run on a free T4 Colab using torchao INT8 quantization on the text encoder, transformer, and VAE — fitting in as little as 2.5GB VRAM with CPU offload. This is the most reliable option and has official Colab notebooks.

**Wan2GP** works on free T4, but you must stick to the Wan 2.2 TextImage2Video 5B FastWan model at 480p — larger checkpoints exceed the 15GB budget and will OOM. It produces roughly a 5-second clip in ~8 minutes and supports Google Drive persistence so you don't re-download 20GB each session.

---

**Tier 2 — Possible with heavy tricks:**

**CogVideoX-5B** also has official free T4 Colab notebooks using INT8 quantization, though inference takes about 30 minutes per run.

**LTX-Video 0.9.7 (2B)** is the fastest open-source video model per frame, but even with GGUF quantization, 97 frames is the sweet spot and 32GB+ RAM is recommended since models occupy significant RAM during load/unload operations even with GGUF.

Check this also:
```
Open-Sora 1.3 (1B)
hpcaitech / PKU — Apache 2.0
Tight fit on T4
Params 1B
VRAM need ~10–14GB
Output 240p–480p · 4–8s
Smallest model ✅
CPU offload ✅
BF16 / FP16
```
---

**Tier 3 — Skip on free Colab:**

HunyuanVideo (13B) can run on GPUs with 14GB of VRAM when using offloading — but that's the absolute minimum and the free T4 isn't reliable enough. Wan 2.2 A14B and anything 14B+ needs a paid A100.

---
