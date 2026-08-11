#!/usr/bin/env bash
#
# install-docker-gpu.sh
#
# Installe, depuis une machine Ubuntu vierge :
#   1) Docker Engine + le plugin compose (méthode officielle Docker)
#   2) Le driver NVIDIA, si une carte est détectée et qu'aucun driver
#      fonctionnel n'est déjà présent
#   3) Le NVIDIA Container Toolkit, et sa configuration dans Docker
#
# Conçu pour un client qui installe le serveur sur sa propre machine, sans
# assumer que quoi que ce soit est déjà en place. Chaque étape vérifie
# d'abord si elle est nécessaire (idempotent : on peut relancer le script
# sans casser une install déjà fonctionnelle).
#
# Usage :
#   chmod +x install-docker-gpu.sh
#   ./install-docker-gpu.sh
#
# Ou directement en copier-coller dans un terminal, en gardant le script
# entier (ne pas exécuter des morceaux séparément, l'ordre compte).
#
# Testé pour Ubuntu 22.04 / 24.04. Nécessite sudo. Ne pas lancer en root
# direct (le script doit connaître l'utilisateur réel pour le groupe docker).

set -euo pipefail

# ---------------------------------------------------------------------------
# Aides d'affichage
# ---------------------------------------------------------------------------

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

step()  { echo -e "\n${BOLD}==> $*${RESET}"; }
ok()    { echo -e "${GREEN}    ok${RESET}  $*"; }
warn()  { echo -e "${YELLOW}    ! ${RESET} $*"; }
fail()  { echo -e "${RED}    erreur${RESET}  $*"; }

REPORT=()
add_report() { REPORT+=("$1"); }

# ---------------------------------------------------------------------------
# Garde-fous de départ
# ---------------------------------------------------------------------------

if [[ "${EUID}" -eq 0 ]]; then
  fail "Ne lance pas ce script directement en root (pas de 'sudo ./install-docker-gpu.sh')."
  fail "Lance-le en tant qu'utilisateur normal ; le script demandera sudo quand il en a besoin."
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  fail "sudo n'est pas installé. Installe-le d'abord (apt install sudo, en root) puis relance."
  exit 1
fi

if [[ ! -f /etc/os-release ]] || ! grep -qi ubuntu /etc/os-release; then
  warn "Ce script est pensé pour Ubuntu. Il peut fonctionner sur un dérivé Debian mais n'a pas été testé."
fi

REAL_USER="${SUDO_USER:-$USER}"
step "Préparation"
ok "Utilisateur ciblé : ${REAL_USER}"
sudo -v   # demande le mot de passe une fois, tôt

# ---------------------------------------------------------------------------
# 1) Docker Engine
# ---------------------------------------------------------------------------

step "1/4 — Docker Engine"

if command -v docker >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
  ok "Docker est déjà installé et le daemon répond. Étape sautée."
  add_report "Docker Engine : déjà présent, non réinstallé"
else
  echo "    Installation de Docker Engine (méthode officielle)..."

  # Paquets tiers connus pour entrer en conflit avec Docker Engine officiel.
  for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
    if dpkg -l | grep -q "^ii  $pkg "; then
      warn "Paquet potentiellement conflictuel détecté : $pkg (laissé en place, pas de désinstallation automatique)"
    fi
  done

  sudo apt-get update -qq
  sudo apt-get install -y -qq ca-certificates curl gnupg

  sudo install -m 0755 -d /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
    sudo chmod a+r /etc/apt/keyrings/docker.asc
  fi

  ARCH="$(dpkg --print-architecture)"
  CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
  echo \
    "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  if sudo docker info >/dev/null 2>&1; then
    ok "Docker Engine installé et le daemon répond."
    add_report "Docker Engine : installé (docker-ce + plugin compose)"
  else
    fail "Docker installé mais le daemon ne répond pas. Vérifie : sudo systemctl status docker"
    exit 1
  fi
fi

# Ajout de l'utilisateur au groupe docker, pour ne plus avoir besoin de sudo.
if id -nG "${REAL_USER}" | grep -qw docker; then
  ok "${REAL_USER} est déjà dans le groupe docker."
else
  sudo usermod -aG docker "${REAL_USER}"
  warn "${REAL_USER} ajouté au groupe docker. Une déconnexion/reconnexion (ou 'newgrp docker') est nécessaire pour que ça prenne effet."
  add_report "Groupe docker : utilisateur ajouté — RECONNEXION NÉCESSAIRE"
fi

# ---------------------------------------------------------------------------
# 2) Driver NVIDIA (uniquement si une carte est présente)
# ---------------------------------------------------------------------------

step "2/4 — Driver NVIDIA"

HAS_NVIDIA_CARD=false
if command -v lspci >/dev/null 2>&1 && lspci | grep -qi nvidia; then
  HAS_NVIDIA_CARD=true
fi

if ! $HAS_NVIDIA_CARD; then
  warn "Aucune carte NVIDIA détectée sur cette machine (lspci). Le serveur tournera en mode CPU."
  add_report "GPU : aucune carte NVIDIA détectée — installation en mode CPU uniquement"
else
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ok "Driver NVIDIA déjà installé et fonctionnel :"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/      /'
    add_report "Driver NVIDIA : déjà présent, non réinstallé"
  else
    echo "    Carte NVIDIA détectée mais driver absent ou non fonctionnel. Installation..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq ubuntu-drivers-common
    RECOMMENDED="$(ubuntu-drivers devices 2>/dev/null | awk '/recommended/ {print $3}' | head -n1)"
    if [[ -n "${RECOMMENDED}" ]]; then
      echo "    Installation du driver recommandé : ${RECOMMENDED}"
      sudo apt-get install -y -qq "${RECOMMENDED}"
    else
      warn "Aucun driver 'recommended' trouvé automatiquement. Tentative avec 'ubuntu-drivers autoinstall'."
      sudo ubuntu-drivers autoinstall
    fi
    warn "Driver NVIDIA installé. Un REDÉMARRAGE de la machine est nécessaire avant qu'il soit chargé."
    add_report "Driver NVIDIA : installé — REDÉMARRAGE NÉCESSAIRE avant de continuer"
    add_report "  -> après redémarrage, relance ce script pour finir l'étape 3 (NVIDIA Container Toolkit)"
    echo
    fail "Redémarre la machine maintenant (sudo reboot), puis relance ce script pour continuer."
    _NEEDS_REBOOT=true
  fi
fi

# Si le driver vient d'être installé et nécessite un reboot, on s'arrête ici :
# le toolkit et sa configuration ont besoin du module nvidia chargé pour être
# testés correctement.
if [[ "${_NEEDS_REBOOT:-false}" == "true" ]]; then
  echo
  echo -e "${BOLD}Résumé (installation partielle, redémarrage requis) :${RESET}"
  for line in "${REPORT[@]}"; do echo "  - $line"; done
  exit 0
fi

# ---------------------------------------------------------------------------
# 3) NVIDIA Container Toolkit (uniquement si une carte GPU est présente)
# ---------------------------------------------------------------------------

step "3/4 — NVIDIA Container Toolkit"

if ! $HAS_NVIDIA_CARD; then
  ok "Pas de carte NVIDIA : étape sautée."
  add_report "NVIDIA Container Toolkit : non installé (pas de GPU sur cette machine)"
else
  if command -v nvidia-ctk >/dev/null 2>&1; then
    ok "nvidia-ctk déjà installé ($(nvidia-ctk --version | head -n1))."
    add_report "NVIDIA Container Toolkit : déjà présent, non réinstallé"
  else
    echo "    Ajout du dépôt NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit

    ok "nvidia-ctk installé ($(nvidia-ctk --version | head -n1))."
    add_report "NVIDIA Container Toolkit : installé"
  fi

  echo "    Configuration de Docker pour utiliser le runtime nvidia..."
  sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
  sudo systemctl restart docker
  ok "Docker reconfiguré et redémarré avec le runtime nvidia par défaut."
  add_report "Docker : configuré avec le runtime nvidia par défaut"
fi

# ---------------------------------------------------------------------------
# 4) Vérification finale
# ---------------------------------------------------------------------------

step "4/4 — Vérification"

DOCKER_OK=false
if sudo docker run --rm hello-world >/dev/null 2>&1; then
  DOCKER_OK=true
  ok "Docker fonctionne (hello-world)."
  add_report "Test Docker : OK"
else
  fail "Docker ne fonctionne pas correctement — 'docker run hello-world' a échoué."
  add_report "Test Docker : ÉCHEC"
fi

GPU_OK=false
if $HAS_NVIDIA_CARD; then
  if sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    GPU_OK=true
    ok "Le GPU est accessible depuis un conteneur Docker."
    add_report "Test GPU dans un conteneur : OK"
  else
    fail "Le GPU n'est PAS accessible depuis un conteneur (docker run --gpus all a échoué)."
    add_report "Test GPU dans un conteneur : ÉCHEC — voir la sortie ci-dessus"
  fi
fi

# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

echo
echo -e "${BOLD}=================== Résumé ===================${RESET}"
for line in "${REPORT[@]}"; do echo "  - $line"; done
echo -e "${BOLD}================================================${RESET}"
echo

if id -nG "${REAL_USER}" | grep -qw docker && [[ "$(id -u)" != "0" ]] && ! groups | grep -qw docker; then
  warn "Le groupe docker a été ajouté à ${REAL_USER} pendant cette session, mais le shell actuel ne l'a pas encore."
  warn "Déconnecte-toi et reconnecte-toi (ou lance 'newgrp docker') avant d'utiliser 'docker' sans sudo."
fi

if $DOCKER_OK && ( ! $HAS_NVIDIA_CARD || $GPU_OK ); then
  echo -e "${GREEN}${BOLD}Installation terminée avec succès.${RESET}"
  if $HAS_NVIDIA_CARD; then
    echo "Tu peux maintenant démarrer le serveur avec le GPU :"
    echo "    python3 scripts/server_ctl.py up"
    echo "(le script détecte automatiquement le GPU et choisit le bon service)"
  else
    echo "Aucun GPU sur cette machine : le serveur démarrera en mode CPU automatiquement :"
    echo "    python3 scripts/server_ctl.py up"
  fi
  exit 0
else
  echo -e "${RED}${BOLD}Installation terminée avec des erreurs. Voir le résumé ci-dessus.${RESET}"
  exit 1
fi
