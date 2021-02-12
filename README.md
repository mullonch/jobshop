# Projet AP/OC

Développements faits dans le cadre du projet de l'ensignement de métaheuristique du MS Valdom.

# Documentation

Objets et méthodes utilisables :


## JobShop

##### créer un probleme de JobShop :
| |Arguments|Résultat|
|-|-|-|
|constructeur|Lien vers le fichier de données du JobShop |Objet JobShop|

***Exemple :***
```
 js = JobShop("instances/ft06")
```

## Solvers

##### différents types de solver :
|classe|Heuristique utilisée|
|-|-|
|GreedySolver|méthodes gloutonnes|
|DescenteSolver|Méthodes de descente|
|RandomSolver|Solution réalisable aléatoire|
|MultipleDescenteSolver|Méthode de descentes multiples|
|TabooSolver|Méthode tabou|

##### instancier un solver :
|classe|Paramètres|
|-|-|
|GreedySolver|strategy : Stratégie de recherche (SPT, LPT, SRPT... default = EST_LRPT)<br/> p_rand : Probabilité de faire un choix aléatoire à chaque selection (default : 0)|
|DescenteSolver|-|
|RandomSolver|-|
|MultipleDescente|nb_starters : nombre d'essais (default = 10)<br/>starter_strategy : stratégie pour le tirage des starters (default = random)<br/>starter_randomisation : pourcentage des choix des starters qui seront randomisés (default = 100)|
|TabooSolver|max_iter : nombre d'itération<br/>timeout : temps d'execution limite<br/>return_time=temps d'attente avant retour d'une solution explorée en recherche|


***Exemples :***
```
 js = JobShop("instances/ft06")
 rs = RandomSolver()
 gs = GreedySolver(strategy="LRPT", p_rand=0.3)

 solution1 = rs.solve(js) #Solution aléatoire
 solution2 = gs.solve(rs) #Solution greedy

 solution3 = DescenteSolver().solve(js, start=solution1)

 solution4 = MultipleDescenteSolver().solve(js)
 #...
```
