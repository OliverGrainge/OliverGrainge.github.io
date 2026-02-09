---
permalink: /
title: "Oliver Grainge"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

<div class="home-hero">
  <h2>AI Engineer &amp; Researcher</h2>
  <p class="hero-tagline">
    Specializing in production ML optimization and deployment. I take models from research to production with measurable performance improvements across edge and cloud platforms&mdash;through model compression, custom kernel development, and hardware-software co-design.
  </p>
  <div class="stat-badges">
    <span class="stat-badge"><i class="fas fa-file-alt"></i> 4 publications (IEEE RAL, AAAI)</span>
    <span class="stat-badge"><i class="fas fa-graduation-cap"></i> PhD candidate, Southampton</span>
    <span class="stat-badge"><i class="fas fa-microchip"></i> CUDA / Triton / ARM NEON</span>
    <span class="stat-badge"><i class="fab fa-github"></i> Open source contributor</span>
  </div>
</div>

<h2 class="home-section-title"><i class="fas fa-briefcase"></i> Experience</h2>

<ul class="exp-list">
  <li class="exp-item">
    <div class="exp-item-header">
      <span class="exp-role">Contract Researcher &mdash; Performance Engineering</span>
      <span class="exp-date">Feb 2025 &ndash; Present</span>
    </div>
    <div class="exp-org">Arm &middot; Remote</div>
    <div class="exp-desc">Architected 6 hands-on tutorials for the Arm Total Performance toolkit covering memory optimization, library acceleration (APL, KleidiAI), and automated porting for AWS Graviton instances.</div>
  </li>
  <li class="exp-item">
    <div class="exp-item-header">
      <span class="exp-role">Research Assistant</span>
      <span class="exp-date">Jun 2025 &ndash; Nov 2025</span>
    </div>
    <div class="exp-org">University College London &middot; London, UK</div>
    <div class="exp-desc">Engineered 1.58-bit precision pipeline for Stable Diffusion (4x memory reduction, 95% quality retention). Designed custom CUDA and Triton kernels for bit-packed tensor operations, delivering 30% speedup over PyTorch baseline.</div>
  </li>
  <li class="exp-item">
    <div class="exp-item-header">
      <span class="exp-role">Visiting Researcher</span>
      <span class="exp-date">Aug 2024 &ndash; Jan 2025</span>
    </div>
    <div class="exp-org">Queensland University of Technology &middot; Brisbane, Australia</div>
    <div class="exp-desc">Engineered speculative decoding for vision-language transformers achieving 2.5x inference speedup for sub-100ms robotic navigation. Implemented training data filtering methods achieving equivalent accuracy with 38% less data.</div>
  </li>
  <li class="exp-item">
    <div class="exp-item-header">
      <span class="exp-role">Research Fellow</span>
      <span class="exp-date">Jan 2024 &ndash; Jan 2025</span>
    </div>
    <div class="exp-org">AI Security Institute &middot; Remote</div>
    <div class="exp-desc">Built automated benchmarking framework evaluating 25+ VLMs across 26k geo-tagged images with 99.9% reliability over 500k+ API calls. Developed privacy-preserving techniques reducing geolocation accuracy by 40%, with interactive demo attracting 5k+ users.</div>
  </li>
  <li class="exp-item">
    <div class="exp-item-header">
      <span class="exp-role">Contract Researcher &mdash; AI Inference Optimization</span>
      <span class="exp-date">Nov 2024 &ndash; Jan 2025</span>
    </div>
    <div class="exp-org">Arm &middot; Remote</div>
    <div class="exp-desc">Engineered demonstrations achieving 40% latency reduction via SIMD/INT8 on mobile and 2.1x throughput on cloud instances. Built Hyperopt-based per-layer precision optimizer demonstrating 22% memory reduction on GPT models.</div>
  </li>
</ul>

<h2 class="home-section-title"><i class="fas fa-file-alt"></i> Selected Publications</h2>

<ul class="pub-list">
  <li class="pub-item">
    <div class="pub-title"><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Pg7KOd4AAAAJ&citation_for_view=Pg7KOd4AAAAJ:qjMakFHDy7sC">Assessing the Geolocation Capabilities, Limitations and Societal Risks of Generative Vision-Language Models</a></div>
    <div class="pub-venue"><span class="venue-name">AAAI 2025</span> &middot; First comprehensive benchmark of VLM geolocation capabilities across 4 datasets</div>
  </li>
  <li class="pub-item">
    <div class="pub-title"><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Pg7KOd4AAAAJ&citation_for_view=Pg7KOd4AAAAJ:9yKSN-GCB0IC">TeTRA-VPR: A Ternary Transformer for Compact Visual Place Recognition</a></div>
    <div class="pub-venue"><span class="venue-name">IEEE RAL</span> &middot; 65% memory reduction and 35% latency reduction for VPR transformers</div>
  </li>
  <li class="pub-item">
    <div class="pub-title"><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Pg7KOd4AAAAJ&citation_for_view=Pg7KOd4AAAAJ:d1gkVwhDpl0C">Design Space Exploration of Low-Bit Quantized Visual Place Recognition</a></div>
    <div class="pub-venue"><span class="venue-name">IEEE RAL</span> &middot; Deployment guidelines for extreme quantization on embedded devices</div>
  </li>
  <li class="pub-item">
    <div class="pub-title"><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Pg7KOd4AAAAJ&citation_for_view=Pg7KOd4AAAAJ:u-x6o8ySG0sC">Structured Pruning for Efficient Visual Place Recognition</a></div>
    <div class="pub-venue"><span class="venue-name">IEEE RAL</span> &middot; Channel pruning achieving 21% latency and 16% memory reduction with &lt;1% accuracy loss</div>
  </li>
</ul>

<h2 class="home-section-title"><i class="fas fa-code"></i> Open Source Projects</h2>

<ul class="project-grid">
  <li class="project-card">
    <div class="project-card-title"><a href="https://github.com/OliverGrainge/BitOps">BitOps</a></div>
    <div class="project-card-tech">C++ &middot; CUDA &middot; ARM NEON &middot; AVX2</div>
    <div class="project-card-desc">High-performance ternary matrix multiplication library with multi-backend support. 16x memory reduction via 2-bit weight packing with kernels outperforming PyTorch FP32 on edge devices.</div>
  </li>
  <li class="project-card">
    <div class="project-card-title"><a href="https://github.com/OliverGrainge/BitCore">BitCore</a></div>
    <div class="project-card-tech">PyTorch</div>
    <div class="project-card-desc">Quantization-aware training toolkit with drop-in BitLinear layers (BitNet, TWN, ParetoQ 1.58-bit). Train-to-deploy workflow with optional BitOps acceleration for 8x inference memory savings.</div>
  </li>
  <li class="project-card">
    <div class="project-card-title"><a href="https://github.com/OliverGrainge/BitCore-Demos">BitNet Chat Interface</a></div>
    <div class="project-card-tech">Python &middot; Gradio &middot; CUDA</div>
    <div class="project-card-desc">Interactive chat with 1.58-bit BitNet models. 24x speedup and 80% memory reduction on ARM M4 vs PyTorch FP32, with backend switching and streaming responses.</div>
  </li>
  <li class="project-card">
    <div class="project-card-title"><a href="https://github.com/OliverGrainge/VSLAM">VSLAM</a></div>
    <div class="project-card-tech">Python &middot; OpenCV &middot; NumPy</div>
    <div class="project-card-desc">Pure-Python stereo SLAM pipeline (KITTI-compatible) with feature tracking, stereo matching, PnP/ICP motion estimation, and bundle adjustment.</div>
  </li>
</ul>

<h2 class="home-section-title"><i class="fas fa-cogs"></i> Technical Skills</h2>

<div class="skills-group">
  <div class="skills-label">Languages</div>
  <div class="skill-pills">
    <span class="skill-pill">Python</span>
    <span class="skill-pill">C/C++</span>
    <span class="skill-pill">CUDA</span>
    <span class="skill-pill">Triton</span>
    <span class="skill-pill">Bash</span>
    <span class="skill-pill">SQL</span>
  </div>
</div>

<div class="skills-group">
  <div class="skills-label">ML Frameworks</div>
  <div class="skill-pills">
    <span class="skill-pill">PyTorch</span>
    <span class="skill-pill">TensorFlow</span>
    <span class="skill-pill">Hugging Face</span>
    <span class="skill-pill">vLLM</span>
    <span class="skill-pill">llama.cpp</span>
    <span class="skill-pill">ONNX Runtime</span>
    <span class="skill-pill">TensorRT</span>
    <span class="skill-pill">OpenCV</span>
  </div>
</div>

<div class="skills-group">
  <div class="skills-label">Model Optimization</div>
  <div class="skill-pills">
    <span class="skill-pill">Quantization (INT8/INT4/ternary)</span>
    <span class="skill-pill">QAT / PTQ / GPTQ / AWQ</span>
    <span class="skill-pill">Pruning</span>
    <span class="skill-pill">Knowledge Distillation</span>
    <span class="skill-pill">LoRA / QLoRA</span>
    <span class="skill-pill">Custom CUDA Kernels</span>
    <span class="skill-pill">FlashAttention</span>
    <span class="skill-pill">SIMD (NEON, AVX2)</span>
  </div>
</div>

<div class="skills-group">
  <div class="skills-label">Infrastructure &amp; MLOps</div>
  <div class="skill-pills">
    <span class="skill-pill">AWS (EC2, Graviton)</span>
    <span class="skill-pill">Docker</span>
    <span class="skill-pill">Kubernetes</span>
    <span class="skill-pill">SLURM</span>
    <span class="skill-pill">Ray</span>
    <span class="skill-pill">MLflow</span>
    <span class="skill-pill">W&B</span>
    <span class="skill-pill">Triton Inference Server</span>
    <span class="skill-pill">FastAPI</span>
  </div>
</div>

<h2 class="home-section-title"><i class="fas fa-graduation-cap"></i> Education</h2>

<div class="edu-item">
  <div class="edu-degree">PhD (iPhD) in Machine Intelligence</div>
  <div class="edu-school">University of Southampton &middot; Oct 2022 &ndash; Present</div>
  <div class="edu-school">Thesis: Efficient Resource-Constrained Visual Place Recognition</div>
</div>
<div class="edu-item">
  <div class="edu-degree">BEng Electronics and Electrical Engineering &mdash; First Class Honours (83%)</div>
  <div class="edu-school">University of Southampton &middot; Sept 2019 &ndash; Jul 2022</div>
</div>

<div class="home-cta">
  <p><strong>Interested in collaborating on efficient ML research or production AI deployment?</strong></p>
  <div class="cta-links">
    <a href="mailto:oliver@grainge.me" class="cta-link-primary"><i class="fas fa-envelope"></i> Get in touch</a>
    <a href="https://github.com/OliverGrainge" class="cta-link-secondary"><i class="fab fa-github"></i> GitHub</a>
    <a href="https://www.linkedin.com/in/oliver-grainge-a6b45a132/" class="cta-link-secondary"><i class="fab fa-linkedin"></i> LinkedIn</a>
    <a href="https://scholar.google.com/citations?user=Pg7KOd4AAAAJ&hl=en" class="cta-link-secondary"><i class="fas fa-graduation-cap"></i> Scholar</a>
  </div>
</div>
