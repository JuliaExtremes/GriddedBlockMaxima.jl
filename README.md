# GriddedBlockMaxima

[![Build Status](https://github.com/JuliaExtremes/GriddedBlockMaxima.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaExtremes/GriddedBlockMaxima.jl/actions/workflows/CI.yml?query=branch%3Amain)

GriddedBlockMaxima.jl permet de simuler des maxima par blocs sur une grille régulière et d’ajuster un modèle spatial bayésien à ces observations. 

Conditionnellement aux paramètres, les maxima de la cellule $j$ suivent la loi GEV de localisation $\mu_j$, d’échelle $\sigma_j = \exp(\phi_j)$ et de forme $\xi$. Les paramètres sont considérés stationnaires dans le temps. La localisation et le logarithme de l’échelle varient d’une cellule à l’autre selon deux champs de Markov gaussiens intrinsèques (iGMRF), de précisions respectives $\kappa_1$ et $\kappa_2$. Les iGMRF peuvent être d'ordre 1 ou 2. Le paramètre de forme $\xi$ est commun à toute la grille.

## Préparer l’environnement

Depuis le dossier où exécuter l’exemple, démarrer Julia avec l’environnement du dossier et les threads disponibles. La paralélisation avec `threads` peut améliorer la performance lorsque le nombre de bloc maxima est grand, mais la parallélisation n'est pas essentielle.

```sh
julia --project=. --threads=auto
```

Pour installer GMRF.jl et GriddedBlockMaxima.jl depuis GitHub, exécuter ces commandes une fois dans Julia. Gadfly sert aux graphiques de l’exemple.

```julia
import Pkg
Pkg.add(url="https://github.com/jojal5/GMRF.jl")
Pkg.add(url="https://github.com/JuliaExtremes/GriddedBlockMaxima.jl")
Pkg.add("Gadfly")
```

Une fois l’environnement prêt, les blocs ci-dessous peuvent être réunis dans `script.jl` et exécutés avec `julia --project=. --threads=auto script.jl`.

## Simuler des maxima sur une grille

L’exemple simule 300 maxima par cellule sur une grille de 30 × 30. On fixe les deux précisions et la forme, puis on tire les champs de localisation et de log-échelle. On conserve les paramètres générateurs pour les comparer ensuite aux estimations.

```julia
import GriddedBlockMaxima as GBM
import GMRF
using Random

Random.seed!(2026)
m₁, m₂ = 30, 30
n_blocs = 300
F_unit = GMRF.iGMRF(m₁, m₂)

true_parameters = GBM.generate_gridded_parameters(F_unit;
    ξ = 0.1,
    location_precision = 10.0,
    logscale_precision = 5.0,
)
y = GBM.generate_gridded_blockmaxima(true_parameters, n_blocs)
```

`y[j]` contient les 300 maxima de la cellule `j`. Pour ajuster le modèle à des données observées, on peut passer directement à l’étape suivante avec un vecteur de séries `y` dont les cellules suivent l’ordre de la grille `F_unit`.

## Estimer les paramètres par MCMC

On initialise l’état à partir des observations, puis on construit l’échantillonneur de Metropolis–Hastings. L’ajustement reçoit les données et la structure de grille. Les valeurs génératrices seront utilisées seulement lors de la comparaison.

```julia
niter = 1000
warmup = 500

θ = GBM.compute_initial_values(y, F_unit)
sampler = GBM.initialize_sampler(F_unit)
chain = GBM.run_mcmc!(θ, sampler, y, F_unit; niter=niter, warmup=warmup)
```

`run_mcmc!` met à jour `θ` et `sampler` sur place et renvoie les états successifs dans `chain`. Pour la localisation et la log-échelle, les cellules d’un même sous-ensemble conditionnellement indépendant peuvent être mises à jour en parallèle.

| Champ de `chain` | Paramètre | Accès à l’itération `i` |
| --- | --- | --- |
| `μ` | Localisation de la cellule `j` | `chain.μ[j, i]` |
| `ϕ` | Logarithme de l’échelle de la cellule `j` | `chain.ϕ[j, i]` |
| `ξ` | Forme commune | `chain.ξ[i]` |
| `κ₁` | Précision du champ de localisation | `chain.κ₁[i]` |
| `κ₂` | Précision du champ de log-échelle | `chain.κ₂[i]` |

Dans cet exemple, `chain.μ` et `chain.ϕ` sont des matrices de 900 × 1 000. Les trois autres champs sont des vecteurs de longueur 1 000.

## Comparer les estimations aux valeurs génératrices

La colonne 1 contient l’état initial. Les 500 transitions d’adaptation produisent les colonnes 2 à 501. On utilise donc les 499 tirages des colonnes 502 à 1 000 pour calculer la médiane a posteriori et l’intervalle de crédibilité central à 95 %. L’exemple compare les deux paramètres locaux à la cellule 95 et les trois paramètres communs à la grille.

```julia
using Statistics

j = 95
retenus = (warmup + 2):niter
parametres = [
    ("μ[$j]", chain.μ[j, :], true_parameters[:μ][j]),
    ("ϕ[$j]", chain.ϕ[j, :], true_parameters[:ϕ][j]),
    ("ξ",     chain.ξ,       true_parameters[:ξ]),
    ("κ₁",    chain.κ₁,      true_parameters[:κ₁]),
    ("κ₂",    chain.κ₂,      true_parameters[:κ₂]),
]

for (nom, tirages, vraie) in parametres
    x = tirages[retenus]
    println((parametre=nom, vraie=vraie, mediane=median(x),
             intervalle95=quantile(x, [0.025, 0.975])))
end
```

Pour travailler directement sur l’échelle des précipitations, `exp.(chain.ϕ[j, retenus])` donne les tirages de $\sigma_j$.

### Tracer les chaînes

Ces graphiques montrent toutes les itérations, adaptation comprise. La ligne rouge indique la valeur utilisée pour simuler les données.

```julia
using Gadfly

for (nom, tirages, vraie) in parametres
    p = plot(
        layer(Geom.line; x=eachindex(tirages), y=tirages),
        layer(Geom.hline(color="red"); yintercept=[vraie]),
        Guide.xlabel("Itération"),
        Guide.ylabel(nom),
    )
    display(p)
end
```

