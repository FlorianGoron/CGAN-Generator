# Générateur de Séquences de Match avec CGAN

Ce code est un générateur de séquences de match basé sur les architectures GAN (Generative Adversarial Networks) et en particulier, Conditional GAN (CGAN). L'objectif est de générer des séquences représentant différentes actions d'un match tel que l'assemblage de ces séquences constitue un match cohérent.

## Fonctionnalités principales

`generate_match(self, nombre_de_matchs=1, nombre_de_minutes=15, style_de_jeu='Equilibré', model_path=None)`: Cette fonction génère des séquences de match en fonction du style de jeu choisi. Vous pouvez spécifier le nombre de matchs, la durée du match et le style de jeu (Offensif, Defensif, Equilibré).

Pour récupérer ce code sur GitHub et exécuter la ligne souhaitée en une commande, vous devriez suivre ces étapes :

1. **Récupération du code depuis GitHub**

    ```bash
    git clone https://github.com/FlorianGoron/CGAN-Generator.git
    cd CGAN-Generator
    ```

2. **Configuration de l'environnement virtuel**

    Il est recommandé d'utiliser un environnement virtuel pour gérer les dépendances. Après avoir cloné le dépôt et avant de lancer le script, installez et activez un environnement virtuel :

    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows, utilisez : venv\Scripts\activate
    pip install -r requirements.txt  
    ```

3. **Exécution depuis le terminal en une ligne**:
    
    Il vous suffit d'utiliser le code suivant en définissant la valeur des paramètres souhaités :

   ```bash
   python nom_du_fichier.py --nombre_de_matchs=3 --style_de_jeu='Offensif'
   ```

Exemple : Ceci exécutera le code avec `nombre_de_matchs` défini sur 3 et `style_de_jeu` sur 'Offensif'. Les autres paramètres prendront leurs valeurs par défaut. 

## Dépendances

- Voir le fichier requirements.txt

## Notes
- Le modèle est entraîné sur les données du match_1 sur 30000 epochs et sauvegardé en tant que `generator_model.h5`. 
- Le dictionnaire de mappage contient la correspondance entre les noms d'action et leur index.
- Les données d'entraînement et la façon dont elles sont prétraitées sont incluses dans le cas où vous souhaiteriez entrainer le modele.












