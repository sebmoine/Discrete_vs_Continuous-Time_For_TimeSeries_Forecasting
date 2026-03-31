# TO-DO (31-03-2026)
## PatchTST :
- Corriger les PDF de PatchTST avec métriques et mettre en .png ("scores" --> "figures")
- Corriger sorties de results.txt et test_results.txt dans "scores"
- Corriger les sorties des "prédictions" en ajoutant les dates et save en .parquet

## LSTM :
- Rajouter les train log ?
- Rajouter un modèle faisant la prédiction sur les 24 dernières heures (params de config ?)
- Recharger le lstm_scores.parquet et ajouter data si pas dejà fait

## XGB:
- Modifier la méthode de fenêtre sur le Univariate (first method)
- Corriger tout le code avec celui du notebook (notamment les validation etc...)
- Faire les checkpoints
- Corriger plot xgb_UNIVARIATE_preds_last_24h.png
- Faire les "prédictions"
- Comparer/Ajouter le Univariate avec la méthode papier (fenetre deja ajustée)

## AUTRES
- Faire un .parquet du DataFrame des scores, unifié
- Faire un plot des prédictions sur les 24 dernières heures des méthodes
- Faire en plus des "scores" dans un format lisibles direct type .txt ou Markdown
- Faire les "times" et arrondir les secondes dans la méthode fcts_times
- [LAST] Implem LNN