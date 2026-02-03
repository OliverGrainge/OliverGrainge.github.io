---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

You can download my full CV as a PDF [here](/files/resume.pdf).

## Education

* **Ph.D in Machine Intelligence**, University of Southampton, 2025 (expected)
* *Thesis focus: Neural network quantization and compression for efficient visual place recognition*

## Publications

  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

## Selected Projects

  <ul>{% for post in site.portfolio reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
