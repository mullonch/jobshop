# Projet AP/OC "Méthodes Approchées pour la Résolution de Problèmes d’Ordonnancement"

Développements faits dans le cadre du projet de l'ensignement de métaheuristique du MS Valdom et retravaillé par la suite

## Documentation

Objets et méthodes utilisables :


### Classe `JobShop`

Classe permettant de modéliser un problème d'ordonnancement.
La façon la plus simple d'instancier un problème est d'utiliser la méthode statique `from_instance_file` prenant en paramètre un chemin vers un fichier d'instance et renvoyant un objet `Jobshop` correspondant au problème donné.

L'exemple ci-dessous crée une instance de `Jobshop` à partir du fichier `abz5` et affiches des bornes inférieures et supérieurs naïves.
```
js = JobShop.from_instance_file(filename="./instances/abz5")
print(js.naive_bounds) #(859, 7773)
```

### Classes `Solver`

Diverses classe permettant de créer des solver pour problèmes de Jobshop.

Chaque classe dispose d'une methode `__init__` permettant de définir les paramètres propres au solver et d'une methode `solve` prennant en paramètre un objet `JobShop` et renvoyant un objet `Solution`.

### - `RandomSolver` :

Effectue un ordonnancement aléatoire (mais valide) des taches :

```
js = JobShop.from_instance_file(filename="./instances/abz5")
print(js.naive_bounds) #(859, 7773)
solver = RandomSolver()
solution = solver.solve(js)
print(solution.duration)
```

### - `GreedySolver` :

Utilise un algorithme glouton pour effectuer l'ordonnancement.
Le constructeur prend deux paramètres en entrée :
 - `strategy (str)` : methode utilisée pour choisir les taches de manières gloutonne, par défaut à "EST_SPT", Valeurs possibles : 
    - `"SPT"` : (Shortest Processing Time) : donne priorité à la tâche la plus courte
    - `"LPT"` : (Longest Processing Time) : donne priorité à la tâche la plus longue
    - `"SRPT"` : (Shortest Remaining Processing Time) : donne la priorité à la tâche appartenant au job ayant la plus petite durée restante
    - `"LRPT"` : (Longest Remaining Processing Time) : donne la priorité à la tâche appartenant au job ayant la plus grande durée
    - `"EST_SPT"`, `"EST_LPT"`, `"EST_LRPT"` ou `"EST_SRPT"` : L'algorithme commençe par filtrer les taches faisables en limitant le choix de la prochaine tâche à celles pouvant commencer au plus tôt, puis applique la stratégie donnée.
    - `"random"` : choisit les taches aléatoirement (strictmeent équivalent à utiliser un RandomSolver)
  - `p_random (float)` : Probabilité de choisir une action aléatoire à chaque prise de décision plutot que de suivre la stratégie définie. par édfaut à 0.

```
js = JobShop.from_instance_file(filename="./instances/abz5")
print(js.naive_bounds) #(859, 7773)
solver = GreedySolver("EST_SPT")
solution = solver.solve(js)
print(solution.duration) #1352
```

### - `DescenteSolver` :
