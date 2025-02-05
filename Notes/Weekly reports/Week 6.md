Dato: 5. feb
-------------

Hva vi har gjort:
- loss invers prop med størrelsen på latent dim 
- å ta vekk gjennomsnittet så ikke ut t å gjør encoding mer effektiv
- pca for å komprimere bildene uten tap av spatial informasjon (last inn data på 0.1s i stede for 1 min)
- ae rekonstruere isje små feil me legge inn i bilde
- laget metrics for å sammenligne bilde og rekonstruert uavhengig av lysstyrke. har også sett på SSIM osv.

- sleit me å få gan t å fungere. koss ska gan komprimere bildene t latent space?


Hva vi skal gjøre:
- gan modell med diffusjon via prompter modell?
  - en "prompter" modell lager en rekke tall fra et bilde (som en encoder)
  - en "generator" modell lager et bilde som svarer til prompten (encoded bilde)

Spørsmål:
- 