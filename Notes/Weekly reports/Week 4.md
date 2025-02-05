## Dato: 24. jan

# Plan for uken:
- hvem er friske vs syke i ADNI datasettet?
  - [X] Alle nedlasta filer er fra friske pasienter.
- lage en metric så vi kan sammenligne modeller
  - [X] Ligger under Metrics
- pca plot av latent dimensjoner i 2d og se om vi finner clustering
  - [X] Ingen god clustering når vi brukte AE, VAE skal i teorien være bedre.
- finne god grunn (feks ikke observert clustering eller for blurry bilder) til å bruke annen modell (eg diffusion eller en type GAN: cycle, pix2pix, wasserstein...)
  - [X] Blurry bilder, ikke observert clustering. Men kan bruke AE som base for GAN elns.

# Spørsmål og svar:
- Målgruppe for rapporten?
  - [X] Ketil. Anta at backpropagation osv er kjent.