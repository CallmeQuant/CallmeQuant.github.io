# callmequant.github.io

Source of Binh Ho's personal website, built with Jekyll on GitHub Pages.

## Layout

- `index.html`, `papers.md`, `notes/index.html` -- the three main pages.
- `_data/papers.yml` -- list of papers (edit here to add one).
- `_data/topics.yml` -- topics used to group the notes; every post sets `topic:` to one of them.
- `_posts/` -- notes, written in kramdown Markdown with MathJax 3. Inline math is written `$$...$$`; display math is a paragraph of its own between `$$` lines.
- `assets/` -- stylesheet, photo, favicon and the figures shown in the notes.
- `figures/<slug>/` -- code for every figure in the note with that slug, in three versions:
  `fig.py` (matplotlib; writes the SVG/PNG used on the site to `assets/figures/<slug>/`),
  `fig.R` (ggplot2) and `fig.jl` (CairoMakie), which write to `figures/<slug>/output/` (not tracked).
  Shared colours and themes are in `figures/_style/`.

## Preview locally

```
bundle install
bundle exec jekyll serve
```

## Regenerate figures

```
python figures/<slug>/fig.py
Rscript figures/<slug>/fig.R
julia figures/<slug>/fig.jl
```

Python needs numpy, scipy and matplotlib; R needs ggplot2, patchwork and scales; Julia needs CairoMakie and the packages listed at the top of each script.
