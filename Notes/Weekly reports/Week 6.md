### Dato: 5. feb

# Forrige uke oppsummert:
- loss invers prop med størrelsen på latent dim 
- å ta vekk gjennomsnittet så ikke ut t å gjør encoding mer effektiv
- pca for å komprimere bildene uten tap av spatial informasjon (last inn data på 0.1s i stede for 1 min)
- ae rekonstruere isje små feil me legge inn i bilde
- laget metrics for å sammenligne bilde og rekonstruert uavhengig av lysstyrke. har også sett på SSIM osv.
- fant ut mesteparten av infoen om hjernen er lagret i kantene dvs de små forskjellene i størrelse på hjernedeler. og åpenbart om hjernemassen er der eller ikke.

- sleit me å få gan t å fungere. koss ska gan komprimere bildene t latent space?

# Plan for uken:
- gan modell med diffusjon via prompter modell?
  - en "prompter" modell lager en rekke tall fra et bilde (som en encoder)
  - en "generator" modell lager et bilde som svarer til prompten (encoded bilde)
  - [X] ______
- se på clustering med bruk av VAE
  - [X] mye bedre enn AE. må bruke varierende reg styrke basert på hvor mange latent dims du har. ref [test_vae](../../test_vae.ipynb)

- sammenligne reconstruction loss for studiegrupper og folkegrupper og lag histogram
- finne forskjell mellom de som får ad og de som forblir mci
- se på gjennomsnittsforskjeller mellom ad og mci
- sammenligne pca og gjennomsnitts-diff. vis effekt av pca i frekvensdomenet
- se om høyfrekvens signal kan gi utslag i reconstruction loss: siden AE har en lowpass-effekt, vil bilder med mer støy se "sykere" ut enn bilder med lite støy. Henger sammen med at $\epsilon-blur(\epsilon) \approx \epsilon$.

# Spørsmål og svar:
- er klassiferere utenfor scopen?
  - [X] i første omgang skal klassifisering ikke være med, men kan være kjekt å ta med for å vise at encodingen fungerer og finne merkverdige forskjeller i latent space mellom grupper.
- husk referanser!!

Neste møte: onsdag kl 9