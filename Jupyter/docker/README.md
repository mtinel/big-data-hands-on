## Configurer l'environnement ##

- éditer la config dans le fichier `.env`
- créer le réseau docker `dev_local`

```
docker network create dev_local
```

## Utilisation sans Traefik ##

```
docker-compose -f jupyter.yml up
```


## Utilisation avec Traefik sous Windows ##

### Installation de Unbound
https://korben.info/installer-serveur-dns-unbound.html

### Configuration de Unbound
https://chez-oim.org/index.php?topic=1599.0

```
curl -O https://www.internic.net/domain/named.cache
```

- copier service.conf, et named.cache dans C:/Program Files/Unbound/
- redémarrer Unbound depuis `services.msc`

changer le dns de la carte réseau depuis `ncpa.cpl`

### Démarrer Traefik ###

```
docker-compose -f traefik.yml up
```
Éditer le fichier `jupyter.yml` et commenter la section `ports`, puis

```
docker-compose -f jupyter.yml up
```