# 𝕌𝕥𝕚𝕝𝕚𝕤𝕒𝕥𝕚𝕠𝕟 𝕕𝕖 𝕝𝕒 𝕤𝕥𝕒𝕔𝕜 𝕕𝕠𝕔𝕜𝕖𝕣 #

## Avec docker-compose ##

### Configurer l'environnement

- éditer la config dans le fichier `.env`
- créer le réseau docker `dev_local`

```
docker network create dev_local
```


### Utilisation avec Traefik sous Windows

#### Installation de Unbound
https://korben.info/installer-serveur-dns-unbound.html

#### Configuration de Unbound
https://chez-oim.org/index.php?topic=1599.0

```
curl -O https://www.internic.net/domain/named.cache
```

- copier service.conf, et named.cache dans C:/Program Files/Unbound/
- redémarrer Unbound depuis `services.msc`

changer le dns de la carte réseau depuis `ncpa.cpl`

#### Démarrer Traefik

```
docker-compose -f traefik.yml up
```
Éditer le fichier `jupyter.yml` et commenter la section `ports`, puis


### Démarrer Jupyter

```
docker-compose -f jupyter.yml up
```

## Avec Kubernetes ##

:construction:

### [Helm](https://helm.sh) (K8s package manager)

[//]:#(https://www.baeldung.com/kubernetes-helm)

#### Installer Helm

Sous Windows
```
choco install kubernetes-helm
```

Sous Mac ou Linux ou WSL
```
brew install kubernetes-helm
```

#### Initialisation

Vérifier la disponibilité du cluster K8s via `kubectl cluster-info` puis
```
helm init
```

#### Installation du chart Jupyter

À l'installation Helm ne dispose que du repo [stable](https://hub.helm.sh/charts/stable).
On peut parcourir les charts sur [hub.helm.sh](https://hub.helm.sh/) et ajouter les repos correspondants au besoin.
On peut également utiliser son propre repo, pour installer un chart :

*TODO : Le repo suivant n'existe pas*
```
helm repo add wow-repo https://wow.github.io/k8s-charts
helm install wow-repo/jupyter --name=jupyter
```


