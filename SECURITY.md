# Rapport de sécurité et de conformité HIPAA
## Projet `slicer-remote-tool-server` - serveur d'inférence GPU distant pour 3D Slicer

*Version 1.0 - juillet 2026*
*Périmètre : architecture, concepts, gestion système, bibliothèques, coût en performance. Aucun code.*

---

## 0. Résumé exécutif

Votre architecture actuelle est saine sur le plan logiciel (registry auto-découvert, validation typée avant exécution, nettoyage des fichiers temporaires, TLS annoncé comme obligatoire, `DATA_DIR` monté en lecture seule). Elle est en revanche **non conforme HIPAA en l'état**, principalement pour quatre raisons structurelles :

1. **Le token Bearer statique partagé** empêche toute identification unique d'utilisateur - c'est une exigence *required* du §164.312(a)(2)(i), pas une option.
2. **Aucune piste d'audit persistante** - le §164.312(b) impose des enregistrements d'activité conservés et exploitables. Vos logs actuels sont volontairement pauvres (bonne intention) mais ne constituent pas un audit trail.
3. **La dé-identification est *supposée* côté client, pas *vérifiée* côté serveur.** Un commentaire dans `main.py` n'est pas un contrôle. C'est le point le plus grave.
4. **Aucun chiffrement au repos** ni gestion de clés : les fichiers temporaires d'imagerie médicale atterrissent en clair sur le disque du serveur.

**La décision la plus structurante du projet n'est pas technique, elle est juridique** : si les données qui arrivent sur le serveur sont *réellement* dé-identifiées au sens du §164.514, elles ne sont plus des PHI, et **le serveur sort du périmètre HIPAA**. Vous passez d'un projet de conformité lourd (BAA, audit, chiffrement de bout en bout, plan de continuité, notification de brèche) à un projet d'hygiène de sécurité classique. Investir massivement dans la robustesse et la *vérifiabilité* de l'anonymisation côté client est donc le meilleur retour sur effort de tout ce rapport.

Attention : cette bascule ne fonctionne que si l'anonymisation est faite **avant** que la donnée ne quitte l'établissement de santé, et qu'elle est **prouvable**. Le serveur doit se comporter comme s'il ne faisait pas confiance au client.

---

## 1. Cadre réglementaire

### 1.1 Votre rôle : Business Associate

L'hôpital ou le laboratoire qui utilise le module Slicer est un **Covered Entity (CE)**. Vous, qui hébergez le serveur et traitez la donnée pour son compte, êtes un **Business Associate (BA)**. Depuis l'HITECH Act (2013), les BA sont **directement responsables** devant l'OCR (Office for Civil Rights), avec les mêmes sanctions civiles et pénales.

Conséquences immédiates :

- Un **Business Associate Agreement (BAA)** signé est obligatoire avec chaque établissement client, *avant* le premier octet de PHI. C'est un contrat, pas une formalité : il définit les usages autorisés, les obligations de sécurité, les délais de notification et le sort des données en fin de contrat.
- Si vous sous-traitez quoi que ce soit (hébergement, sauvegarde, monitoring, envoi d'e-mails, stockage objet), il vous faut un **BAA en cascade** avec ce sous-traitant. « Le serveur est à nous » simplifie beaucoup ce point - gardez-le ainsi.
- Vous devez pouvoir répondre à une demande d'accès, de comptabilité des divulgations (*accounting of disclosures*) ou d'audit émanant du CE.

### 1.2 Les trois règles applicables

| Règle | Ce qu'elle impose chez vous |
|---|---|
| **Privacy Rule** (§164.500+) | Minimum nécessaire, usages autorisés uniquement, définition de la dé-identification (§164.514) |
| **Security Rule** (§164.302–318) | Garanties administratives, physiques et techniques sur les ePHI. C'est le cœur de votre travail d'ingénierie |
| **Breach Notification Rule** (§164.400+) | Notification au CE sans délai déraisonnable, ≤ 60 jours. Le chiffrement conforme NIST crée une *safe harbor* : une donnée chiffrée volée n'est pas une brèche notifiable |

Ce dernier point mérite d'être souligné : **le chiffrement est le seul contrôle qui vous exonère de l'obligation de notification**. À lui seul il justifie l'investissement.

### 1.3 État du droit en juillet 2026

Le NPRM de refonte de la Security Rule <cite index="7-1">a été publié au Federal Register le 6 janvier 2025, la période de commentaires s'est close le 7 mars 2025, et l'OCR examine encore environ 4 745 commentaires</cite>. <cite index="8-1">Dans le Unified Agenda de l'automne 2026, HHS a déplacé le projet (RIN 0945-AA22) vers son agenda « Long-Term Actions » avec juillet 2027 comme horizon d'action finale - ce qui indique généralement qu'aucune règle finale n'est attendue dans les douze mois.</cite> <cite index="8-1">La Security Rule actuelle reste pleinement applicable et l'OCR continue de sanctionner sur les fondamentaux : analyse de risque, mesures de sécurité, gestion des fournisseurs, formation.</cite>

**Ce que ça change pour vous : rien, ou plutôt, tout dans le bon sens.** Le NPRM, s'il est finalisé, <cite index="7-1">supprimerait la distinction entre spécifications « required » et « addressable »</cite> et rendrait obligatoires le chiffrement au repos et en transit, le MFA, l'inventaire d'actifs, les tests de vulnérabilité et la revue annuelle. Concevez dès maintenant selon le NPRM : c'est ce que vous devriez faire de toute façon, et vous serez prêts si la règle passe. <cite index="7-1">Le délai de mise en conformité serait de 240 jours après publication.</cite>

Notez aussi que <cite index="3-1">l'absence ou l'insuffisance d'analyse de risque reste le manquement le plus fréquemment relevé</cite> dans les enquêtes de l'OCR - voir §9.

### 1.4 Avertissement : si le projet est européen

Le dépôt, le contexte et la langue suggèrent un projet français ou européen. Si des données de patients européens transitent, **HIPAA n'est pas le référentiel applicable**, ou pas le seul :

- **RGPD** : les données de santé sont des données sensibles (art. 9). Base légale, DPIA (art. 35) quasi certainement obligatoire pour un traitement d'imagerie médicale à grande échelle, registre des traitements, DPO.
- **Certification HDS** (Hébergeur de Données de Santé, art. L.1111-8 CSP) : en France, **héberger des données de santé à caractère personnel pour le compte d'un tiers exige une certification HDS**. C'est une certification d'organisme accrédité (basée ISO 27001 + exigences spécifiques), pas une déclaration. « Le serveur est à nous » ne vous en exonère pas - au contraire, c'est précisément le cas visé.
- **Transferts hors UE** : le commentaire « deploy in the appropriate jurisdiction » dans votre `main.py` est pertinent. Un serveur hors UE déclenche le chapitre V du RGPD.
- Le seuil d'**anonymisation** du RGPD (considérant 26 : irréversibilité, résistance à la singularisation, la corrélation et l'inférence) est **plus strict** que le Safe Harbor HIPAA. Une donnée « Safe Harbor » peut rester une donnée personnelle pseudonymisée au sens du RGPD.

**Recommandation** : faites trancher ce point par un juriste avant d'investir dans l'architecture. Si HDS s'applique, la certification conditionne le calendrier de tout le projet (12 à 18 mois typiquement). Le reste de ce rapport reste valable dans les deux cas - les contrôles techniques sont très largement communs.

---

## 2. Audit de l'existant : écarts identifiés

| # | Constat dans l'architecture actuelle | Exigence | Gravité |
|---|---|---|---|
| 1 | Token Bearer statique unique partagé par tous les clients | §164.312(a)(2)(i) identification unique de l'utilisateur | **Critique** |
| 2 | Aucun audit trail persistant | §164.312(b) | **Critique** |
| 3 | Dé-identification supposée côté client, non vérifiée | §164.514 + principe de zero-trust | **Critique** |
| 4 | Fichiers temporaires en clair sur disque (`TEMP_DIR`) | §164.312(a)(2)(iv) chiffrement au repos | **Critique** |
| 5 | Pas de sauvegarde ni de plan de reprise | §164.308(a)(7) contingency plan | **Élevée** |
| 6 | `GET /tools` non authentifié → divulgation de la surface fonctionnelle | Minimum nécessaire | Moyenne |
| 7 | `ALLOWED_EXTENSIONS` = validation par extension uniquement | Validation de contenu, anti-traversal, anti-zip-bomb | Élevée |
| 8 | Architecture synchrone bloquante, pas de rate limiting | §164.306(a)(2) disponibilité - DoS trivial | Élevée |
| 9 | Pas d'analyse de risque formalisée ni de politiques écrites | §164.308(a)(1) - *le manquement le plus sanctionné* | **Critique** |
| 10 | Pas de contrôle d'intégrité sur les données stockées/transmises | §164.312(c)(1) et (e)(2)(i) | Moyenne |
| 11 | Uvicorn potentiellement exposé directement | Bonne pratique / durcissement | Élevée |
| 12 | Pas de session timeout / expiration de jeton | §164.312(a)(2)(iii) déconnexion automatique | Moyenne |

Les points 1 à 4 et 9 sont bloquants pour une mise en production avec de la vraie donnée patient.

---

## 3. Anonymisation et dé-identification

C'est le chapitre le plus important. Traitez-le en premier.

### 3.1 Les deux voies légales du §164.514

**Safe Harbor (§164.514(b)(2))** - Suppression de 18 catégories d'identifiants : noms, toutes les subdivisions géographiques inférieures à l'État (les 3 premiers chiffres du code postal sont conservables si la zone couvre > 20 000 habitants), **toutes les dates plus précises que l'année** (dates de naissance, d'admission, de sortie, de décès), âges > 89 ans (à agréger en « 90+ »), téléphones, fax, e-mails, numéros de sécurité sociale, numéros de dossier médical, numéros d'assurance, numéros de compte, numéros de licence, plaques d'immatriculation, identifiants d'appareil et **numéros de série**, URL, adresses IP, identifiants biométriques (empreintes, voix), **photographies du visage entier et images comparables**, et tout autre identifiant unique. Plus l'absence de connaissance réelle qu'une ré-identification serait possible.

*Simple à auditer, mécanique, défendable. C'est ce que vous devez viser.*

**Expert Determination (§164.514(b)(1))** - Un statisticien qualifié atteste par écrit que le risque de ré-identification est « très faible ». Permet de conserver plus d'information (dates relatives, âges élevés) mais coûte cher, doit être renouvelé, et l'attestation doit être conservée.

Deux pièges à connaître :

- **Les identifiants d'appareil et numéros de série (catégorie 16)** touchent directement l'imagerie : `DeviceSerialNumber`, `StationName`, les identifiants de scanner. Ils sont souvent oubliés.
- **« Photographies du visage entier et images comparables » (catégorie 17)** est la disposition qui rend un volume CT ou IRM crânien problématique : voir §3.2.

### 3.2 Spécificités de l'imagerie médicale - ce qui rend le problème difficile

Un fichier DICOM n'est pas un CSV. Quatre couches d'identifiants coexistent :

**a) Les métadonnées standard.** Environ 150 tags DICOM sont porteurs d'identité. Le référentiel de référence est le **DICOM PS3.15 Annexe E**, qui définit le *Basic Application Level Confidentiality Profile* et une dizaine d'*options* (Retain Longitudinal Temporal, Retain Device Identity, Clean Descriptors, etc.). Ne réinventez pas cette liste : implémentez le profil.

**b) Les private tags et les éléments non standard.** Chaque constructeur (Siemens, GE, Philips, Canon) ajoute des blocs privés dont le contenu est non documenté et contient régulièrement des identifiants, des noms d'opérateur, des chemins de fichiers réseau. **Règle : supprimer tous les private tags par défaut**, et ne réintroduire au cas par cas que ceux dont vous avez besoin, après inspection manuelle. Idem pour les *curve data*, *overlay planes* et *SQ nested sequences* (les séquences imbriquées sont un classique des fuites : un tag propre au niveau racine, un `PatientName` oublié trois niveaux plus bas).

**c) Le texte incrusté dans les pixels (*burned-in annotation*).** Ultrasons, radiologie interventionnelle, captures d'écran de consoles : le nom du patient est *dans l'image*. Le tag `BurnedInAnnotation` existe mais est peu fiable - souvent absent ou `NO` à tort. Il faut soit exclure ces modalités, soit passer par de l'OCR (Tesseract / EasyOCR) et masquer les régions détectées. C'est coûteux et imparfait ; l'exclusion par liste blanche de modalités est plus sûre.

**d) Le visage reconstructible.** C'est le point le plus sous-estimé, et il est central pour vous puisque 3D Slicer manipule des volumes. **Un CT ou une IRM de la tête permet de reconstruire une surface faciale, et la reconnaissance faciale automatique sur ces reconstructions atteint des taux d'identification élevés** contre une base de photographies. Plusieurs publications (notamment dans le *NEJM* en 2019) l'ont démontré expérimentalement. Une donnée « anonymisée » au sens des tags reste donc identifiante au sens de la catégorie 17 du Safe Harbor.

Trois familles de contre-mesures, par coût croissant :
- **Skull stripping / brain extraction** - on ne garde que le cerveau. Radical, mais destructeur si votre outil a besoin des structures faciales (cas de `surg_mov_pred`, chirurgie orthognathique : là c'est rédhibitoire).
- **Defacing** - on supprime les voxels du visage (`pydeface`, `mri_deface`, `mridefacer`). Fiable, coûteux (recalage sur atlas).
- **Refacing / face-swapping** - on remplace le visage par un visage synthétique (`@afni_refacer`), ce qui préserve les traitements sensibles à la géométrie faciale.

**Si votre cas d'usage exige les structures faciales, le defacing est impossible et le Safe Harbor devient inatteignable.** Vous basculez alors nécessairement sur l'Expert Determination, ou vous restez en régime PHI complet avec toutes les obligations associées. **Tranchez ce point tôt** : il conditionne toute l'architecture.

**e) Les UIDs.** `StudyInstanceUID`, `SeriesInstanceUID`, `SOPInstanceUID`, `FrameOfReferenceUID` sont des identifiants uniques qui permettent de recorréler avec le PACS d'origine. Ils doivent être **remappés de façon déterministe mais non inversible**, tout en préservant les relations internes (un même `FrameOfReferenceUID` doit rester cohérent entre séries d'une même étude, sinon le recalage casse). La technique standard : `nouveau_UID = racine_UID_organisation + HMAC-SHA256(clé_secrète, UID_original)` tronqué et formaté selon les contraintes DICOM (64 caractères max, chiffres et points).

**f) Les formats dérivés.** NIfTI (`.nii.gz`) est souvent considéré comme « déjà anonyme » - c'est faux : le header NIfTI contient un champ `descrip` libre, et les conversions DICOM→NIfTI (`dcm2niix`) génèrent des fichiers JSON *sidecar* BIDS qui embarquent quantité de métadonnées. Les fichiers de scène Slicer (`.mrml`, `.mrb`) contiennent des **chemins de fichiers absolus** qui incluent souvent le nom du patient.

### 3.3 Où anonymiser, et comment le vérifier

Votre `claude.md` dit : *« de-identification happens client-side »*. **C'est la bonne décision architecturale** - la PHI ne doit jamais quitter le réseau de l'établissement. Mais telle qu'elle est formulée, c'est une intention, pas un contrôle.

Le modèle à mettre en place est un **double rideau** :

**Rideau 1 - Client (module Slicer) : la dé-identification effective.**
Un pipeline obligatoire, non contournable par l'utilisateur, exécuté avant tout appel réseau :
1. Application du profil DICOM PS3.15 + suppression des private tags
2. Remappage HMAC des UIDs
3. Décalage de dates cohérent par patient (*date shifting* : un offset aléatoire fixe par patient, qui préserve les intervalles) ou suppression selon le profil retenu
4. Defacing si applicable
5. **Vérification post-traitement** : re-scan du fichier produit, et refus d'envoi si un identifiant résiduel est détecté
6. Journalisation locale de l'opération (quel fichier, quel profil, quel horodatage, quel opérateur)

Une case à cocher « j'ai anonymisé » dans l'UI est **inacceptable** : la conformité ne peut pas dépendre d'un clic.

**Rideau 2 - Serveur : le portier (*de-identification gate*).**
Le serveur ne fait pas confiance au client. Avant d'appeler `tool.invoke()`, il exécute un contrôle indépendant :
- Parsing du fichier reçu, vérification que les tags de la liste noire sont absents ou vides
- Vérification du format réel (magic bytes), pas seulement de l'extension
- Rejet en **`422`** avec un message **générique** (« payload rejected: potential identifier detected in tag group X ») - surtout **ne jamais renvoyer la valeur détectée**, sinon vous transformez votre message d'erreur en canal de fuite de PHI
- Incrémentation d'un compteur d'alerte et notification de sécurité : un client qui envoie de la PHI est un incident, pas une erreur d'utilisateur

Ce portier a un double intérêt : il vous protège techniquement, et il constitue **la preuve documentaire** que vous avez mis en place une mesure raisonnable - ce qui est exactement ce que l'OCR regarde en cas d'enquête.

Architecturalement, ce portier s'insère naturellement dans votre design existant : c'est un middleware FastAPI ou une étape ajoutée dans `Tool.invoke()` avant `validate()`, ou mieux, un `ArgSpec` de type `"phi_file"` qui déclenche le contrôle. Zéro modification du core, conformément à votre contrainte d'extensibilité.

### 3.4 Pseudonymisation et table de correspondance

Vos utilisateurs auront besoin de retrouver quel résultat correspond à quel patient. La règle :

- Le serveur ne voit qu'un **pseudonyme** (`HMAC-SHA256(clé_établissement, MRN)`, tronqué)
- La **table de correspondance pseudonyme ↔ patient reste chez le Covered Entity**, jamais chez vous, jamais dans un backup chez vous
- La clé HMAC est détenue par l'établissement uniquement
- Si vous détenez la clé ou la table, **les données redeviennent des PHI** : c'est le critère de « clé de codage » du §164.514(c)

C'est un point de conception, pas d'implémentation : ne créez jamais une fonctionnalité serveur « retrouver le patient d'origine ».

### 3.5 Bibliothèques

| Besoin | Bibliothèque | Note |
|---|---|---|
| Manipulation DICOM | `pydicom` | Socle incontournable |
| Dé-identification DICOM clé en main | `dicognito` | Simple, cohérent, remappe les UIDs. Bon point de départ |
| Dé-identification avec règles configurables | `deid` (Stanford) | Recettes déclaratives, gère l'OCR de pixels |
| Profil PS3.15 strict | `dicom-anonymizer` | Implémente les actions du standard |
| Pipeline de production éprouvé | **RSNA CTP** (Java, hors Python) | Standard de fait des essais cliniques. À considérer sérieusement en amont du serveur |
| Réseau DICOM | `pynetdicom` | Si vous devez parler à un PACS |
| Defacing | `pydeface`, `quickshear`, `@afni_refacer` | `quickshear` est ~10× plus rapide, moins précis |
| NIfTI | `nibabel` | Nettoyage des headers |
| OCR de texte incrusté | `pytesseract`, `easyocr` | Dernier recours |
| Détection de PHI en texte libre | `presidio-analyzer` (Microsoft) | Pour les champs `StudyDescription`, comptes rendus |

---

## 4. Chiffrement en transit

### Ce qu'il faut

- **TLS 1.3 exclusivement** (TLS 1.2 acceptable en repli avec suites AEAD uniquement ; TLS 1.0/1.1 et SSLv3 désactivés). Suites : `TLS_AES_256_GCM_SHA384`, `TLS_CHACHA20_POLY1305_SHA256`.
- **Certificat d'une AC reconnue** en production (Let's Encrypt via ACME, renouvellement automatisé) ou PKI interne si le serveur n'est pas exposé. Le certificat auto-signé documenté dans votre README est **strictement réservé au développement** - assurez-vous que `verify_tls=False` est impossible à activer autrement que par une variable d'environnement de dev, jamais depuis l'UI Slicer.
- **HSTS** avec `max-age` long, **OCSP stapling**, **Perfect Forward Secrecy** (ECDHE, X25519).
- **mTLS (authentification mutuelle par certificat client)** - fortement recommandé ici. Votre population de clients est petite, connue et gérée : c'est le cas d'usage idéal. Chaque installation Slicer reçoit un certificat client, révocable individuellement. Cela résout partiellement le problème #1 (identification unique) et ajoute une couche indépendante du jeton applicatif.
- **Certificate pinning** côté client Slicer : le module embarque l'empreinte du certificat serveur attendu. Protège contre une AC compromise et contre les proxies TLS d'entreprise qui déchiffreraient votre trafic médical.
- **Terminaison TLS sur un reverse proxy** (nginx, Caddy, Traefik) devant Uvicorn - jamais Uvicorn nu exposé sur Internet. Le proxy gère TLS, le rate limiting, les limites de taille de corps, les timeouts, les en-têtes de sécurité.

### Bibliothèques / outils

`cryptography` (pyca) pour tout le reste ; côté client `requests` avec `verify=` pointant sur votre bundle CA ; `certifi` ; pour le pinning, un `HTTPAdapter` personnalisé. Configuration TLS : générez-la avec le **Mozilla SSL Configuration Generator** (profil *modern*) plutôt qu'à la main. Testez avec `testssl.sh` et SSL Labs.

---

## 5. Authentification, autorisation, traçabilité des accès

### Le problème du token statique

Un `API_TOKEN` unique en variable d'environnement, comparé en temps constant (bon réflexe), pose quatre problèmes rédhibitoires :

1. **Pas d'imputabilité** : impossible de savoir *qui* a lancé quel traitement. L'audit trail devient inutile.
2. **Pas de révocation granulaire** : un poste compromis oblige à faire tourner le secret sur *tous* les postes.
3. **Pas d'expiration** : un token qui fuit reste valide indéfiniment.
4. **Pas de MFA**, que le NPRM rendrait obligatoire.

### La cible

**Modèle recommandé : OIDC + JWT courts.**

- Un **fournisseur d'identité** (Keycloak auto-hébergé, ou l'IdP de l'établissement via fédération SAML/OIDC) authentifie l'utilisateur avec MFA.
- Le module Slicer obtient un **access token JWT de courte durée** (5–15 min) et un refresh token. Le flow adapté à une application de bureau est **Authorization Code + PKCE**, avec ouverture du navigateur système - jamais de mot de passe saisi dans Slicer.
- Le serveur **valide le JWT localement** contre le JWKS du fournisseur (mis en cache) : signature, `exp`, `aud`, `iss`. Pas d'appel réseau par requête.
- Le JWT porte l'identité (`sub`), l'établissement (`tenant`) et les rôles.
- **RBAC** : quel utilisateur a le droit d'exécuter quel `Tool`, et d'accéder à quels fichiers de `DATA_DIR`. Cela s'exprime très bien comme un attribut de classe sur votre `Tool` (`required_role`), vérifié dans `invoke()` - encore une fois sans toucher au core.
- **Cloisonnement multi-établissement** : les données et modèles de l'hôpital A ne doivent jamais être atteignables par l'hôpital B. Votre `DATA_DIR/<tool_name>/` doit devenir `DATA_DIR/<tenant>/<tool_name>/`, et la résolution passer par le `tenant` du JWT, jamais par un paramètre client.

**Solution de transition** (si OIDC est trop lourd à court terme) : tokens opaques longs, **un par utilisateur**, stockés hachés (Argon2id ou HMAC-SHA256 avec pepper) dans une base, avec date d'expiration, date de dernier usage, et possibilité de révocation individuelle. C'est le minimum acceptable, pas une cible.

### Contrôles complémentaires

- **Rate limiting** par identité et par IP (`slowapi`, ou au niveau du reverse proxy). Protège la disponibilité (§164.306(a)(2)) et limite l'exfiltration en masse.
- **Timeout de session / expiration automatique** (§164.312(a)(2)(iii)) : le JWT court le fait naturellement.
- **Verrouillage après échecs répétés**, avec journalisation (§164.308(a)(5)(ii)(C)).
- **Authentifier `GET /tools`** : la liste de vos outils et de leurs schémas d'arguments est de la surface d'attaque gratuite. `GET /health` peut rester ouvert mais ne doit rien révéler (pas de version, pas de nom d'outil, pas de hostname).

### Bibliothèques

`authlib` (le plus complet, gère OIDC/OAuth2 côté client et serveur), `pyjwt` ou `python-jose` pour la validation, `argon2-cffi` pour le hachage de secrets, `passlib` en couche d'abstraction, `slowapi` pour le rate limiting, `secrets` + `hmac.compare_digest` de la stdlib pour la génération et la comparaison en temps constant.

---

## 6. Persistance : chiffrement au repos, intégrité, sauvegardes

C'est ici que se joue votre exigence de « persistance à toute épreuve ».

### 6.1 Les trois niveaux de chiffrement - ils sont complémentaires, pas alternatifs

**Niveau 1 - Chiffrement de volume (LUKS2 / dm-crypt).**
Chiffre le disque entier. Protège contre le vol physique du disque, la mise au rebut, la RMA d'un disque défaillant. **Ne protège contre rien pendant que la machine tourne** - une fois le volume monté, tout processus root lit en clair. C'est un socle nécessaire, jamais suffisant. Algorithme : AES-256-XTS. Déverrouillage au boot : TPM 2.0 avec scellement sur PCR, ou Clevis/Tang pour un déverrouillage réseau, ou saisie manuelle (fiable mais incompatible avec le redémarrage automatique).

**Niveau 2 - Chiffrement de système de fichiers (fscrypt, gocryptfs, eCryptfs).**
Granularité par répertoire, clés distinctes. Utile pour cloisonner par établissement.

**Niveau 3 - Chiffrement applicatif (*envelope encryption*).**
**C'est le niveau qui compte réellement pour HIPAA.** Chaque fichier est chiffré par une **DEK** (Data Encryption Key) unique, en **AES-256-GCM**. La DEK est elle-même chiffrée par une **KEK** (Key Encryption Key) détenue par un gestionnaire de clés. Le fichier stocké contient : nonce + DEK chiffrée + ciphertext + tag d'authentification.

Bénéfices décisifs :
- La donnée reste chiffrée même si l'OS est compromis (tant que la KEK ne l'est pas)
- **Crypto-shredding** : détruire la DEK détruit la donnée, instantanément et irrévocablement, y compris dans toutes les sauvegardes. C'est la seule façon crédible de garantir une suppression sur un stockage répliqué et sauvegardé
- Rotation de clé sans réécrire les données (on re-chiffre seulement les DEK)
- Cloisonnement cryptographique par établissement (une KEK par tenant)

**Détail technique important** : AES-GCM ne permet pas de chiffrer un flux de plusieurs gigaoctets d'un bloc (limite de 64 GiB par clé/nonce, et surtout impossibilité d'authentifier sans tout bufferiser). Pour des volumes CT/IRM, utilisez un **AEAD en streaming à segments** : découpage en blocs de 1 à 4 Mo, chaque bloc chiffré avec un nonce dérivé du numéro de segment, plus un marqueur de fin de flux pour empêcher la troncature. C'est le modèle de *Tink Streaming AEAD*, de `age`, ou de `libsodium secretstream`. **Ne l'implémentez pas vous-même** : utilisez `pynacl` (`crypto_secretstream_xchacha20poly1305`) ou `tink`.

### 6.2 Gestion des clés

C'est le vrai sujet. Un chiffrement fort avec une clé dans un fichier `.env` à côté de la donnée n'apporte rien.

- **Cible : HSM ou KMS.** Si le serveur est chez vous : **HashiCorp Vault** (moteur Transit : Vault chiffre/déchiffre les DEK sans jamais exposer la KEK), éventuellement adossé à un HSM PKCS#11 (Thales, Utimaco, YubiHSM 2 pour un budget réduit) ou à SoftHSM en développement.
- **Séparation stricte** : la KEK ne doit jamais être sur la machine qui stocke les données chiffrées, ni dans les mêmes sauvegardes.
- **Rotation** : KEK annuelle, DEK par objet (donc jamais réutilisée).
- **Procédure de récupération** : secrets de descellement Vault partagés selon **Shamir** entre plusieurs porteurs, conservés hors ligne. Testée.
- **Zéro secret dans le code, l'image Docker ou Git.** Injection par `systemd LoadCredential`, Docker secrets, ou récupération Vault au démarrage via AppRole/JWT auth. Ajoutez un scanner de secrets en CI (`gitleaks`, `detect-secrets`).

### 6.3 Intégrité

Le §164.312(c)(1) exige de protéger les ePHI contre l'altération ou la destruction non autorisée, et le §164.312(e)(2)(i) l'équivalent en transit.

- **AES-GCM fournit l'intégrité gratuitement** (tag d'authentification) : c'est la raison de préférer un AEAD à un mode de chiffrement seul. Un fichier altéré échoue au déchiffrement.
- **Empreinte SHA-256** calculée côté client, transmise dans l'en-tête, revérifiée côté serveur : détecte les corruptions de transport et documente l'intégrité de bout en bout.
- **Système de fichiers à checksums** : ZFS ou Btrfs avec *scrubbing* périodique, contre le *bit rot* silencieux. Sur des volumes d'imagerie de plusieurs To conservés des années, ce n'est pas théorique.
- **RAID n'est pas une sauvegarde.** Le RAID protège de la panne disque, pas de la suppression, du chiffrement par ransomware, ni de l'erreur humaine.

### 6.4 Sauvegardes et reprise d'activité

Le §164.308(a)(7) impose un *contingency plan* : sauvegarde, restauration, mode dégradé, analyse de criticité, et **procédures testées**.

- **Règle 3-2-1-1-0** : 3 copies, 2 supports différents, 1 hors site, **1 immuable ou hors ligne**, 0 erreur au test de restauration.
- **Immuabilité anti-ransomware** : *object lock* WORM (MinIO, S3 Object Lock), ou snapshots ZFS avec délégation restreinte, ou bandes LTO hors ligne. Un attaquant qui obtient root ne doit pas pouvoir effacer les sauvegardes - c'est aujourd'hui le mode opératoire standard des attaques contre les hôpitaux.
- **Sauvegardes chiffrées** avec une clé distincte de la production, restaurables sans dépendre du serveur d'origine.
- **RPO / RTO définis et contractualisés** dans le BAA.
- **Test de restauration complet documenté, au moins semestriel.** Une sauvegarde jamais restaurée n'existe pas. Le rapport de test est la preuve que l'OCR demandera.
- Outils : `restic` ou `borgbackup` (chiffrement natif, déduplication, snapshots) ; `rclone` pour la réplication hors site.

### 6.5 Rétention et destruction

- **Politique de rétention écrite** : combien de temps gardez-vous une donnée de traitement ? La réponse par défaut devrait être **le temps du traitement, plus rien**. Un serveur d'inférence stateless est infiniment plus simple à défendre qu'un serveur qui archive.
- HIPAA impose 6 ans de conservation pour la **documentation de conformité** (politiques, analyses de risque, logs d'audit) - pas pour les données patient elles-mêmes, dont la rétention relève du droit de l'État et du contrat.
- **Destruction sécurisée** : NIST SP 800-88 Rev.1. Sur SSD/NVMe, l'écrasement logique ne fonctionne pas de façon fiable (*wear leveling*, over-provisioning) : utilisez le *cryptographic erase* natif du disque, ou mieux, appuyez-vous sur le crypto-shredding applicatif (§6.1) qui rend la question caduque.

### 6.6 Le cas particulier de vos fichiers temporaires

Votre `main.py` streame les uploads dans `TEMP_DIR` et garantit le nettoyage en `try/finally` + `BackgroundTask`. C'est bien conçu, mais insuffisant :

- **`TEMP_DIR` doit être sur un volume chiffré**, ou mieux sur un **tmpfs** (RAM) si les volumes le permettent.
- **Désactivez le swap**, ou chiffrez-le avec une clé aléatoire régénérée à chaque boot. Sinon une image médicale peut être écrite en clair sur disque par le noyau, hors de tout contrôle applicatif.
- **`mkstemp` avec permissions 0600**, répertoire parent 0700, propriétaire = utilisateur de service dédié non privilégié.
- **Un crash ou un `SIGKILL` laisse des orphelins.** Ajoutez un balayage périodique (`systemd-tmpfiles` avec âge maximal, ou une tâche de démarrage) qui purge tout fichier de plus de N minutes. Votre `finally` ne couvre pas l'OOM killer.
- **Attention au `is_temporary` du `DataStore`** : votre conception est correcte (les chemins persistants de `LocalDataStore` ne sont jamais supprimés), mais un futur backend distant qui matérialise un blob en local devra le faire dans le `TEMP_DIR` chiffré, pas dans `/tmp`.
- Sur du chiffrement de volume, `shred` est inutile ; le nettoyage logique suffit puisque le média est chiffré.

---

## 7. Journalisation et piste d'audit

Le §164.312(b) exige des mécanismes matériels, logiciels ou procéduraux enregistrant et examinant l'activité sur les systèmes contenant des ePHI. Le §164.308(a)(1)(ii)(D) exige la **revue régulière** de ces enregistrements - les produire ne suffit pas.

### Ce qu'il faut journaliser

Pour chaque requête : horodatage UTC synchronisé NTP, identité de l'utilisateur (`sub` du JWT), établissement, adresse IP source, endpoint, nom de l'outil, identifiant de corrélation de requête, code HTTP, durée, taille des données, résultat du contrôle de dé-identification, et - surtout - les événements d'authentification (succès, échec, verrouillage), les changements de configuration, les accès administrateur, les démarrages/arrêts de service.

### Ce qu'il ne faut **jamais** journaliser

Votre `claude.md` le dit déjà et c'est excellent : jamais de contenu de fichier, jamais de valeur d'argument, jamais de métadonnée patient, jamais de token. Ajoutez : jamais de nom de fichier client (il contient souvent le nom du patient - journalisez un hash), jamais de stack trace complète renvoyée au client.

**Piège classique** : les traces d'exception de FastAPI/Starlette peuvent inclure les valeurs des variables locales, donc potentiellement le contenu d'un argument. Configurez un gestionnaire d'exception global qui journalise un identifiant d'erreur et renvoie au client uniquement cet identifiant. Si vous utilisez Sentry ou équivalent, activez le scrubbing agressif - ou ne l'utilisez pas du tout (c'est un sous-traitant, donc un BAA de plus).

### Comment le rendre défendable

- **Format structuré** (JSON) - `structlog` ou `python-json-logger`. Un log parsable est un log exploitable.
- **Inaltérabilité** : chaînage cryptographique (chaque entrée contient le hash de la précédente : toute suppression ou modification casse la chaîne), et/ou export immédiat vers un collecteur externe en append-only. Un attaquant qui compromet le serveur ne doit pas pouvoir réécrire l'historique.
- **Centralisation** : SIEM (Wazuh est open source et couvre bien le besoin, Graylog, ELK). Séparation de privilège : le serveur écrit, il ne peut pas effacer.
- **Rétention 6 ans**, cohérente avec l'obligation documentaire.
- **Alerting** : échecs d'authentification répétés, rejets du portier de dé-identification, volumes de téléchargement anormaux, accès hors horaires.
- **Revue périodique documentée** - un compte rendu écrit, même court, même mensuel. C'est ce que l'auditeur demandera.

---

## 8. Durcissement système et infrastructure

### Réseau
Segmentation : le serveur GPU dans un VLAN dédié, sans accès Internet sortant (ou seulement via proxy avec liste blanche - un serveur qui ne peut pas sortir ne peut pas exfiltrer). Pare-feu en deny-by-default. Pas de SSH exposé sur Internet : bastion ou VPN (WireGuard). Reverse proxy en frontal, WAF (ModSecurity/Coraza) en option. `fail2ban` ou `crowdsec`.

### Système d'exploitation
Utilisateur de service dédié, non privilégié, sans shell. Durcissement selon les **CIS Benchmarks**. Mises à jour de sécurité automatiques (`unattended-upgrades`). SSH par clé uniquement, `PermitRootLogin no`. `auditd` configuré sur les accès aux répertoires de données. AppArmor ou SELinux en mode enforcing. Unité systemd durcie : `ProtectSystem=strict`, `PrivateTmp=yes`, `NoNewPrivileges=yes`, `MemoryDenyWriteExecute`, `RestrictAddressFamilies`.

### Conteneurs
Votre `docker-compose.yml` monte déjà `./DATA:/data:ro` - bon réflexe, généralisez-le. Image de base minimale (`python:slim`, `distroless`, ou Chainguard). `USER` non-root explicite. `read_only: true` sur le rootfs avec des `tmpfs` explicites pour les répertoires inscriptibles. `cap_drop: ALL`. Profil seccomp. Pas de `--privileged` ; pour le GPU, `nvidia-container-toolkit` avec les capacités minimales. Jamais de socket Docker monté dans un conteneur.

### Chaîne d'approvisionnement logicielle
Dépendances épinglées (`requirements.txt` avec hashes, ou `uv`/`poetry.lock`). **SBOM** généré (`syft`, CycloneDX). Scan de vulnérabilités : `pip-audit` en CI, `trivy` ou `grype` sur les images, Dependabot activé. SAST : `bandit` et `semgrep`. Le NPRM rendrait obligatoire un inventaire d'actifs et une cartographie réseau - le SBOM en est la brique logicielle.

### Validation des entrées (durcissement de votre `/run/{tool_name}`)
Au-delà de `ALLOWED_EXTENSIONS` :
- Vérification des **magic bytes** (`python-magic`) : un fichier `.dcm` doit commencer par `DICM` à l'offset 128
- **Neutralisation du nom de fichier** : ne jamais réutiliser le nom fourni par le client pour construire un chemin. Générez un UUID côté serveur. Sinon, path traversal.
- **Protection anti-zip-bomb** sur les archives : limite de ratio de décompression et de nombre d'entrées
- **`defusedxml`** si vous parsez du XML (XXE)
- **Limite de taille appliquée au niveau du proxy**, avant que le corps n'atteigne l'application ; votre `MAX_UPLOAD_MB` → `413` est un second rempart, pas le premier
- **Timeouts** sur `tool.run()` : un outil qui boucle bloque tout dans une architecture synchrone

### Disponibilité
Votre modèle synchrone bloquant est le talon d'Achille : quelques requêtes lourdes concurrentes suffisent à saturer le GPU et à faire expirer tous les autres clients. La disponibilité est une exigence HIPAA explicite (§164.306(a)(2)). Vous notez la file d'attente comme hors périmètre - c'est un choix légitime pour l'itération actuelle, mais **planifiez-la** : limite de concurrence, file d'attente, backpressure explicite (`429` ou `503` avec `Retry-After`) plutôt qu'un timeout silencieux.

---

## 9. Le volet administratif - celui qui est réellement sanctionné

C'est la partie que les équipes techniques négligent systématiquement, et c'est précisément là que l'OCR sanctionne : <cite index="3-1">l'analyse de risque incomplète ou absente reste le motif principal</cite>.

| Exigence | Référence | Livrable concret |
|---|---|---|
| **Analyse de risque** | §164.308(a)(1)(ii)(A) | Document formel : inventaire des actifs, flux de données, menaces, vulnérabilités, probabilité × impact. **À faire en premier, avant tout développement sécurité** - c'est elle qui justifie vos choix techniques |
| Gestion du risque | §164.308(a)(1)(ii)(B) | Plan de remédiation daté et suivi |
| Politique de sanction | §164.308(a)(1)(ii)(C) | Écrite, communiquée |
| Revue d'activité | §164.308(a)(1)(ii)(D) | Comptes rendus de revue des logs |
| Responsable sécurité désigné | §164.308(a)(2) | Nommément, par écrit |
| Gestion des accès | §164.308(a)(3),(4) | Procédures d'attribution, de revue et de **révocation** (départ d'un collaborateur) |
| Formation | §164.308(a)(5) | Sessions tracées, émargées, renouvelées |
| Réponse à incident | §164.308(a)(6) | Runbook : détection, confinement, éradication, notification ≤ 60 j, retour d'expérience |
| Plan de continuité | §164.308(a)(7) | Sauvegarde, restauration, mode dégradé, **tests** |
| Évaluation périodique | §164.308(a)(8) | Audit annuel, pentest |
| Contrats sous-traitants | §164.308(b)(1) | BAA signés et à jour |
| **Garanties physiques** | §164.310 | *« Le serveur est à nous »* → la sécurité physique est **votre** responsabilité : local fermé, baie verrouillée, contrôle et journal des accès, vidéosurveillance, procédure de mise au rebut des disques |

Une remarque de fond : **la documentation n'est pas de la bureaucratie, c'est votre défense**. En cas d'incident, la différence entre une sanction lourde et un classement sans suite tient souvent à la capacité à produire une analyse de risque à jour et des preuves de mesures raisonnables.

---

## 10. Récapitulatif des bibliothèques Python

| Domaine | Bibliothèque | Rôle |
|---|---|---|
| **Cryptographie généraliste** | `cryptography` (pyca) | AES-GCM, X25519, dérivation, X.509. **Le standard**. Backend OpenSSL, donc AES-NI natif |
| Cryptographie haut niveau | `pynacl` (libsodium) | API difficile à mal utiliser ; `secretstream` pour l'AEAD en streaming |
| AEAD en streaming | `tink` | Implémentation de référence du chiffrement de gros fichiers |
| Hachage de mots de passe/secrets | `argon2-cffi` | Argon2id, lauréat du PHC. À préférer à bcrypt |
| Abstraction d'auth | `passlib` | Couche de compatibilité et de migration |
| Aléa cryptographique | `secrets` (stdlib) | Tokens, nonces. **Jamais `random`** |
| Comparaison temps constant | `hmac.compare_digest` (stdlib) | Déjà utilisé chez vous - bon |
| JWT / OIDC | `authlib`, `pyjwt` | Validation locale contre JWKS |
| Gestion de clés | `hvac` | Client HashiCorp Vault |
| HSM | `python-pkcs11` | Si HSM matériel |
| DICOM | `pydicom`, `pynetdicom` | Socle |
| Anonymisation DICOM | `dicognito`, `deid`, `dicom-anonymizer` | Voir §3.5 |
| Defacing | `pydeface`, `quickshear` | Volumes crâniens |
| NIfTI | `nibabel` | Nettoyage de headers |
| Détection de PHI textuelle | `presidio-analyzer` | Champs libres |
| OCR | `pytesseract`, `easyocr` | Texte incrusté |
| Validation de type de fichier | `python-magic` | Magic bytes |
| XML sécurisé | `defusedxml` | Anti-XXE |
| Logs structurés | `structlog` | JSON, contexte, redaction |
| Rate limiting | `slowapi` | Limite par identité |
| Validation | `pydantic` v2 | Déjà cohérent avec votre `ArgSpec` |
| Audit sécurité du code | `bandit`, `semgrep`, `pip-audit` | CI |
| Détection de secrets | `gitleaks`, `detect-secrets` | Pre-commit + CI |
| Observabilité | `opentelemetry-*` | Traçage sans PHI |

**Principe transversal : ne jamais implémenter de primitive cryptographique soi-même**, et ne jamais concevoir soi-même un protocole cryptographique. Utilisez `cryptography` ou `pynacl`. Bannissez `pycrypto` (non maintenu, vulnérable), méfiez-vous de `pycryptodome` utilisé naïvement (il permet trop facilement ECB ou une réutilisation de nonce).

---

## 11. Coût en temps d'exécution

C'est votre question la plus concrète, et la réponse est contre-intuitive.

### 11.1 Les ordres de grandeur

Sur un serveur x86-64 moderne avec AES-NI/VAES (ce que vous avez forcément si vous avez un GPU) :

| Opération | Débit / latence | Commentaire |
|---|---|---|
| AES-256-GCM | **2–8 Go/s par cœur** | Accéléré matériellement. Un volume de 1 Go : **~0,2 s** |
| ChaCha20-Poly1305 | 1–2 Go/s par cœur | Repli sur matériel sans AES-NI (ARM ancien) |
| SHA-256 | 1–2 Go/s (2–5 avec SHA-NI) | Empreinte de 1 Go : **~0,5 s** |
| BLAKE3 | 5–10 Go/s | Alternative si le hachage devient limitant |
| Handshake TLS 1.3 | **1 RTT** (~10–50 ms réseau) + ~1–2 ms CPU | Amorti par le keep-alive et la reprise de session |
| Chiffrement en masse TLS | 2–8 % de CPU au débit ligne | Invisible sous 10 Gbit/s |
| mTLS (surcoût) | ~1–2 ms par handshake | Négligeable |
| Vérification JWT (EdDSA/RS256) | **30–100 µs** | Avec JWKS en cache. Sans cache : +10–50 ms de réseau |
| Hachage Argon2id | **100–500 ms** *volontairement* | Uniquement à la connexion, jamais par requête |
| Appel KMS/Vault (`GenerateDataKey`) | **10–60 ms** | À amortir : une DEK par fichier ou par lot, pas par bloc |
| LUKS2 / dm-crypt | **−5 à −20 %** de débit séquentiel NVMe | Sur charge GPU-bound : invisible |
| `fsync` par entrée de log | 0,1–1 ms (NVMe), 1–10 ms (réseau) | **Piège** : à grouper (*group commit*) |
| Chaînage SHA-256 du log | < 10 µs par entrée | Négligeable |

### 11.2 Le coût réel : l'anonymisation, pas la cryptographie

| Opération | Coût typique |
|---|---|
| Anonymisation DICOM, headers seuls (`stop_before_pixels=True`) | **1–5 ms par fichier** → série CT de 300 coupes : **0,5–2 s** |
| Anonymisation DICOM avec réécriture des pixels | 10–50 ms/coupe (non compressé), **50–200 ms/coupe** (JPEG2000 lossless) → **15–60 s par série** |
| OCR de détection de texte incrusté | **100–500 ms par coupe** - prohibitif sur un volume complet |
| `quickshear` (defacing rapide) | **2–10 s par volume** |
| `pydeface` / `mri_deface` (recalage sur atlas) | **30 s – 3 min par volume** |
| `@afni_refacer` | **1–5 min par volume** |
| Portier de dé-identification côté serveur (headers) | **1–5 ms par fichier** |

### 11.3 Bilan sur un aller-retour typique

Scénario réaliste : volume NIfTI de 200 Mo, inférence GPU de 30 s.

| Poste | Surcoût |
|---|---|
| Handshake TLS (première requête) | +30 ms |
| Chiffrement TLS du transfert | +40 ms CPU (le réseau domine largement) |
| Validation JWT | +0,1 ms |
| Portier de dé-identification serveur | +5 ms |
| Écriture chiffrée du fichier temporaire (AES-GCM) | +50 ms |
| Empreinte SHA-256 d'intégrité | +100 ms |
| Récupération de la DEK (Vault, mise en cache) | +30 ms (première fois seulement) |
| Journalisation d'audit avec `fsync` groupé | +1 ms |
| Pénalité LUKS sur les I/O | +~10 ms |
| **Total côté serveur** | **≈ 0,27 s, soit < 1 % d'un traitement de 30 s** |
| *Anonymisation côté client (avec defacing)* | *+60 s - mais hors serveur, une seule fois par jeu de données* |

### 11.4 Conclusions à retenir

1. **La cryptographie ne coûte rien** dans votre contexte. Sur une charge GPU-bound avec des transferts réseau de plusieurs centaines de mégaoctets, le chiffrement représente moins de 1 % du temps total. **Aucun arbitrage sécurité/performance n'est justifié ici.** C'est un argument à utiliser en interne : le débat « le chiffrement va ralentir le serveur » n'a pas lieu d'être en 2026.
2. **Le coût réel est dans l'anonymisation**, et surtout dans le defacing. Il s'exécute côté client, en amont, et se parallélise (multiprocessing par volume). Prévoyez-le dans l'UX : barre de progression, traitement par lot en tâche de fond, mise en cache d'un jeu déjà anonymisé (avec purge).
3. **Les vrais pièges de performance ne sont pas cryptographiques** :
   - Un `fsync` par entrée de log → sérialise tout. Groupez.
   - Un appel KMS par bloc au lieu d'un par fichier → multiplie la latence réseau par mille.
   - Argon2id appelé par requête au lieu d'une fois à la connexion → +300 ms par appel.
   - Une introspection OAuth2 distante par requête au lieu d'une validation JWT locale → +20 ms par appel.
   - Chiffrer un volume de 4 Go en un seul buffer mémoire → OOM. Streamez par segments.
4. **Mesurez plutôt que d'estimer.** `openssl speed -evp aes-256-gcm` sur votre machine cible donne le chiffre exact en dix secondes. Profilez avec `py-spy` ou `scalene` avant d'optimiser quoi que ce soit.

---

## 12. Feuille de route proposée

### Phase 0 - Avant toute ligne de code (2 à 4 semaines)
1. **Trancher la question juridique** : HIPAA seul, ou RGPD + HDS ? Consultez un juriste. Cela conditionne tout.
2. **Analyse de risque formelle** (§164.308(a)(1)(ii)(A)). C'est l'exigence n°1 et la plus sanctionnée.
3. **Trancher la question du defacing** : vos outils ont-ils besoin des structures faciales ? Si oui, le Safe Harbor est hors de portée et il faut planifier une Expert Determination.
4. Modèle de **BAA** et désignation d'un responsable sécurité.

### Phase 1 - Bloquants techniques (4 à 8 semaines)
5. Pipeline d'anonymisation client, non contournable, avec vérification post-traitement
6. Portier de dé-identification côté serveur (middleware ou `ArgSpec` de type `"phi_file"`)
7. Authentification par identité individuelle : OIDC + JWT courts, ou a minima tokens per-user révocables. mTLS en complément
8. Audit trail persistant, structuré, chaîné, centralisé
9. Volume chiffré (LUKS2) + `TEMP_DIR` sur tmpfs ou volume chiffré + swap désactivé

### Phase 2 - Consolidation (8 à 12 semaines)
10. Chiffrement applicatif par enveloppe + Vault (KEK/DEK, crypto-shredding)
11. Sauvegardes immuables + test de restauration documenté
12. Durcissement OS/conteneur, reverse proxy, rate limiting, validation de contenu
13. Politiques écrites, formation, runbook d'incident

### Phase 3 - Vérification et pérennisation
14. Pentest externe et revue de code sécurité par un tiers
15. SBOM, scan de dépendances en CI, revue périodique des logs
16. Alignement anticipé sur le NPRM (MFA généralisé, inventaire d'actifs, tests annuels)

---

## Trois messages à retenir

1. **La dé-identification vérifiée côté serveur est votre meilleur investissement.** Elle peut faire sortir le serveur du périmètre HIPAA, et à défaut, elle constitue la preuve de diligence qui vous protège.
2. **Le token statique partagé doit disparaître.** Sans identité individuelle, aucun audit trail n'a de valeur, et l'exigence est *required*, pas *addressable*.
3. **La performance n'est pas un argument contre la sécurité ici.** Le surcoût cryptographique est inférieur à 1 % ; le seul coût significatif est l'anonymisation, qui s'exécute côté client et se parallélise.

---

*Ce rapport est un document d'ingénierie et ne constitue pas un avis juridique. Les décisions de qualification réglementaire (statut de Business Associate, applicabilité du RGPD et de la certification HDS, choix entre Safe Harbor et Expert Determination) doivent être validées par un conseil juridique compétent et, le cas échéant, par un statisticien qualifié.*