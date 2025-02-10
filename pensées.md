# Alife RL pipeline
Le but est de faire une all-in-one pipeline pour appliquer RLHF pour entraîner un modèle (appelé Generator) qui trouve des paramètres de simulations interessants dans le cadre de la vie artificielle.  

## La théorie
Voici la pipeline théorique:
- Le Generator génère un batch de paramètres.  
- Un humain fait un classement de simulations  
- Le classement est utilisé pour entraîner un reward model (Rewardor)  
- Le modèle de génération s'entraîne grâce au Rewardor  

Et on peut loop le processsus jusqu'à ce que les paramètres générés soient tous intéressants.  


## L'application
Grossièrement, ce serait de faire un script qui permet d'acceder à tous les steps et de pouvoir les faire run en parallèle, et que ce soit étendable à n'importe quel type de simulations.  
Aussi d'avoir plusieurs profils de runs, pour pouvoir tester différents modèles sans rien perdre.

Je ferais:
- Un `main`, qui sert d'index et de launcher pour les autres parties. Au lancement, on sélectionne un profil, ou on en crée un, puis on sélectionne si on veut lancer un labeling, ou un training.
- Une partie `Labeling`, où on code la simulation et la génération des vidéos ou images ou n'importe. La pipeline s'occupe ensuite de prendre les data générées, les présente des paires à l'utilisateur qui sélectionne la meilleure, jusqu'à ce qu'il ne veule plus ou que le classement soit établi. Les résultats sont enregistrés dans le profil de la run.  
- Une partie `Training`, qui comporte 2 parties: le training du Rewardor, et le training du Generator.
- Une partie `Benchmarking`, qui permet de juste regarder des simulations générées par le Generator d'un profil et de sauvegarder celles qu'on veut

Structure:   
```{text}
*main.py
    *simulations
        lenia
            simulate.py
            generate_vids.py
        particle_life
            ...
    *generators
        lenia_filter
            generator.py
        lenia_new_gen
            ...
    *rewardors
        common
            super_rewardor_that_works_for_everything.py

    .profiles
        lenia_filterG_and_commonR
            labeled_data.dat
            vids.dat
            generator.dat
            rewardor.dat
            saved_runs
                ...
        particle_life_newG_and_commonR
            ...
```   
*Stars indicate compulsory files/dir*