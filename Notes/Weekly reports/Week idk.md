# Plan framover:
1. april: alt skrevet inn i rapporten (ca 100 sider med tekst).
14. april: ferdig å kutte ned/skrive om rapporten.
30. april: levere rapporten.

# Spørsmål til møtet:
- [ ] Hvor mye kan vi skrive med chatgpt?
- [ ] Hvilke vanlige feil bør vi unngå?
- [ ] Kan vi få tilgang til tidligere vurderinger av andres rapporter?
- [ ] Skal vi ta med forsøk som ikke fungerte i tillegg til dem som fungerte? Ja

Neste møte: 26. mars

# Notater fra møtet:
gjøre et bidrag noe nytt
gjøre noe vanskelig (få d t)
sydd sammen rapport med fin innledning, begrunne arbeid, forklare d nye, vise resultatene, diskutere ka som gjekk bra/dårlig og neste steg

ha figurer der de skal brukes, og ha med forklaring.
foretrekk lite tekst framfor mye vas.

ligge på rundt 50 sider

ta med plott av ReLU etc (?)

arbeidet skal kunne bli gjort på nytt med samme resultat

prøve å forklare latent space av MRI data med flere variabler

del 1: anomaly detection og forbedring av forrige metoder
del 2: anvendelse (clustering) og forklaring
del 3: statistisk forklaring (p-verdi for clustering etc)


et problem me å prøve å forklare clustering me så mange variabler e at sannsynligvis vil me få false positives. for å minske sjansen for dette kan vi bruke flere modeller sine latent space og se om forklaringene er statistisk significant, men vil fremdeles ha stor sjanse for false positives pga samme data.

kjøre på nytt med "signifikante" kliniske variabler. begrunne hvorfor vi må begrense scopet.

argumentere med klinisk årsak/effekt kausalitet. lage en hypotese og test (eller bruke andre sine).

"explorative analysis" gir deg mer frihet til å teste ting uten å legge det frem som en "hypotese"

explainable ai: kan dele hjernen inn i deler (bruke biologisk) og se på reconstruction loss for hver hjernedel
- teste om hjernedeler som er kjent for å være forskjellige for AD pasienter faktisk har høy recon loss.

trene opp en predictor for biomarkører basert på latent space og se hvilke som har høy reconstruction loss? er det relatert til reconstruction loss i MRI?