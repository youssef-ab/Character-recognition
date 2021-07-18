# Coding assignment - Label refinement
## Problématique
On a une dataset d'images chacune contient des mots localisés et annotés. L'objectif est de développer un algorithme permettant la reconnaissance des caractères de chaque mot et l'extraction de leurs localisations i.e. créer un jeu de données OCR au niveau des symboles à partir d'un jeu de données OCR au niveau des mots. Ceci va nous aider à économiser les ressources financières et humaines nécessaires pour ce travail.


# Méthodes implémentées
## Méthode 1
### Algorithme
Dans cette méthode, on ne fait pas de la reconnaissance. En effet, on se base sur l'ordre des contours détectés dans l'image de chaque mot par un traitement d'image en utilisant la bibliothèque **OpenCV**, et aussi sur la valeur du mot donnée en entrée de la fonction dans la liste des annotations.  
Le traitement d'image fait sur chaque mot est le suivant :

- Conversion en image en grayscale.
- Seuillage de l'image pour la rendre binaire (caractères en noir et arrière-plan en blanc). Ceci permet de faciliter la détection des contours.
- Détection des contours, puis on les classifie sur l'axe x.

On se basant sur ce classement et sur l'ordre des caractères dans le mot, on peut assigner à chaque contour un caractère.
### Avantages et limitations
Cette méthode a des limitations qui sont liées au dataset, au traitement d'image effectué et aussi à l'algorithme elle-même (reconnaissance par ordre). Voici l'explication de chaque problème et son effet sur la sortie :

  
- La dataset peut contenir certains mots dont les caractères sont très serrés, par conséquence le nombre de contours détectés va diminuer et on aura un chevauchement des caractères. Lorsque ce problème est rencontré au début du mot, les caractères assignés aux contours qui viennent après vont être erronés.  
- Parfois, le seuillage effectué n'arrive pas à coloriser les caractères en noir (existence des petites taches blanches sur les caractères) et donc avoir une image binaire, par conséquent, on aura une détection de plusieurs contours qui correspondent au même caractère.  
- à cause de ces problèmes expliqués précédemment, la détection par ordre des contours n'est pas valable tout le temps et dépend totalement du résultat de la fonction de détection de contours d'**OpenCV**. Si le nombre de contours détectés est différent du nombre de caractères dans le mot (supérieur ou inférieur) les affectations effectuées seront erronées.
- Dans cette méthode, il n'existe pas une métrique pour mesurer la précision, car on ne peut pas savoir si le caractère affecté à un contour est le vrai caractère.
Le seul avantage de cette méthode est le temps d'exécution qui est très inférieur au temps d'exécution de la deuxième méthode qu'on va voir après.
## Méthode 2
### Algorithme
Comme la première méthode, on effectue le même traitement pour isoler chaque contour (caractère), puis on effectue une reconnaissance de chaque caractère à l'aide de la bibliothèque **Pytesseract**. Ceci nous permet d'éviter les problèmes générés par la première méthode basée sur l'ordre du caractère dans le mot. Elle nous permet aussi de mesurer la précision de détection à l'aide de deux métriques implémentées :

- Mesure du pourcentage des caractères reconnus par rapport à tous les caractères dans l'image d'entrée. Pour cela on vérifie l'existence de chaque caractère détecte par **Pytesseract** dans le mot et aussi son indice. Ceci garantit que le caractère reconnu correspond à celui existant dans le mot dont l'indice est celui de son contour (sur l'axe x). On peut ne pas prendre en considération l'indice du caractère (parfois, on détecte plus de contours) et vérifier juste l'existence s'il n'y a pas des caractères répétés dans le même mot. Ceci augmente la précision, car la vérification de l'ordre élimine parfois des caractères bien reconnus, mais mal indexés à cause des problèmes de détection de contour expliqués précédemment.
- La deuxième métrique consiste à calculer une matrice de confusion pour chaque mot reconnu : on compare le mot qui existe dans la liste des labels et le mot composé par les caractères détectés. Ceci nous donne une matrice contenant des 0 et 1. La norme de différence entre cette matrice et la matrice identité qui correspond à une vraie détection de tout le mot nous donne une vue générale sur l'erreur de détection. En effet, si cette norme est proche de 0 on a une bonne précision, par contre s'il dépasse 1 on a une mauvaise précision.

Score calculé pour chaque mot :          
<p align="center">
  <img width="150" height="100" src="https://latex.codecogs.com/gif.latex?N%20%3D%20%5Cfrac%7B%5Cmid%20%5Cmid%20cf%5C_mat%20-%20Id%20%5Cmid%5Cmid%5C%20%7D%7B%5Cmid%20%5Cmid%20Id%20%5Cmid%5Cmid%5C%20%7D">
</p>

### Avantages et limitations
Généralement, cette méthode est plus précise que la première. Cependant, elle a des limitations qui causent des fausses détections :  
  
- Erreurs dues à l'algorithme de **Pytesseract**, par exemple 'b' peut être confondu avec 6.  
- Erreurs dues au traitement de l'image comme dans la première méthode.
- Temps d'exécution supérieur à celui de la première méthode à cause de l'utilisation de la bibliothèque **Pytesseract**.

**Remarque** : dans les deux méthodes, on filtre les contours d'une surface inférieure à ![](https://latex.codecogs.com/gif.latex?25%20pixel%5E2) pour éviter des fausses détections (ex. les points des lettres i et j).
# Exécution
Pour exécuter, vous aurez besoin d'installer les deux bibliothèques **Pytesseract** et **OpenCV**. Après l'exécution, vous aurez l'option de choisir entre le test de l'algorithme sur une image ou d'exécuter une des deux méthodes. Dans ce dernier choix, vous aurez en sortie un dictionnaire contenant les annotations des caractères de chaque image :

	{
	    "ABCDEFGH.jpg": 
	        [
	            {
	                "geometry": [[xmin1, ymin1], [xmax1, ymax1]],
	                "value": "char1"
	            },
	            {
	                "geometry": [[xmin2, ymin2], [xmax2, ymax2]],
	                "value": "char2"
	            },
	            ...
	        ],
	    ...
	}
Voici un exemple d'exécution (test de la méthode 2) :
![Bobst](https://i.imgur.com/EJcKGF6.png)
