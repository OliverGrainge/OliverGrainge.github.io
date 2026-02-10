---
title: "PRIV-LOC Demo: Play GeoGuessr vs Vision-Language Models"
excerpt: "Test your place-recognition skills against state-of-the-art VLMs in this interactive GeoGuessr-style game. Built on Hugging Face Spaces."
collection: portfolio
header:
  teaser: REPLACE_ME.png
tags:
  - Demo
  - AI Safety
  - VLMs
---

**Can you out-guess today's Vision-Language Models?** The *PRIV-LOC Demo* drops you into a random street-level scene and challenges both you and a suite of VLMs—GPT-4o, Gemini 2.0, LLaMA-Scout 17B and more—to pinpoint its location. Click on the map, lock in your guess, then watch how close the models land.

The game runs entirely in your browser; no GPS metadata is present in the images, so any accurate localisation reflects the latent capabilities of the models themselves. If the embed below does not load (e.g. due to content-security restrictions), use the "Launch in new tab" button.

<p>
  <a href="https://huggingface.co/spaces/oeg1n18/priv-loc-demo"
     target="_blank" rel="noopener noreferrer"
     style="padding:0.6em 1em;background:#0052CC;color:#fff;border-radius:6px;text-decoration:none;">
    Launch PRIV-LOC Demo
  </a>
</p>

<iframe
  src="https://oeg1n18-priv-loc-demo.hf.space"
  width="100%"
  height="680"
  style="border:none;max-width:100%;"
  loading="lazy"
  title="PRIV-LOC GeoGuessr Demo"
  allow="clipboard-write; fullscreen">
</iframe>
