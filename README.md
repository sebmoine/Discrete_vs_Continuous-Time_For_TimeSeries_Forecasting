# TO-DO (31-03-2026)
## PatchTST :
- Corriger les sorties des "prédictions" en ajoutant les dates et save en .parquet (line 325)
- Adapter les params pour prédire sur 24h (ajout de VALSIZE et TESTSIZE)

## Linear :
- Ne conserver que les preds du meilleur modèle OU du moins mettre une balise type "[BEST]"

## LSTM :
- Vérifier/Refaire les fenêtres

## XGB:
- Modifier la méthode de fenêtre sur le Univariate (first method)
- Corriger tout le code avec celui du notebook (notamment les validation etc...)
- Faire les checkpoints
- Corriger plot xgb_UNIVARIATE_preds_last_24h.png
- Faire les "prédictions"
- Comparer/Ajouter le Univariate avec la méthode papier (fenetre deja ajustée)
- Explorer https://github.com/truefit-ai/m5

## AUTRES
- Faire un .parquet du DataFrame des scores, unifié
- Faire un plot des prédictions sur les 24 dernières heures des méthodes
- Faire en plus des "scores" dans un format lisibles direct type .txt ou Markdown
- Faire les "times" et arrondir les secondes dans la méthode fcts_times
- [LAST] Implem LNN
- Add model_name = "all"