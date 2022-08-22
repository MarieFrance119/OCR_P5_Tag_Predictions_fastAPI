# OCR_P5_Tag_Predictions_fastAPI
## Categorize questions
Contient le code à déployer pour une mise en production sur une API.

A partir d'un texte, l'API prédit les tags associés à ce texte.

Le dossier Code contient 
- les fichiers python à exécuter :
  - un fichier main.py avec le code pour l'API
  - un fichier functions.py avec les fonctions appelées dans le fichier main.py
- un dossier models où se trouvent les éléments enregistrés au format joblib nécessaires à l'exécution des fonctions (pipeline du modèle final, encodage des tags).

L'API utilisée est FastApi (site web de FastApi https://fastapi.tiangolo.com/#run-it).

Via le terminal (en se plaçant dans le répertoire contenant les fichiers python et le dossier models), on lance la commande suivante :

uvicorn main:app --reload

L'API est ensuite disponible sur http://127.0.0.1:8000/docs#/

### Autre API
Une autre API (déployée sur le web) pour tester la prédiction de tag est disponible [ici](https://share.streamlit.io/mariefrance119/ocr_p5_tag_predictions_api/main/main.py) avec le [repository associé](https://github.com/MarieFrance119/OCR_P5_Tag_Predictions_API).
