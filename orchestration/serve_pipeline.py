"""Crée le déploiement Prefect du pipeline de données et le sert.

Usage (le serveur Prefect doit déjà tourner, voir docs/orchestration.md) :
    python orchestration/serve_pipeline.py               # déclenché à la main ou chaque nuit à 03:00
    python orchestration/serve_pipeline.py --no-schedule # déclenché à la main seulement

Le processus reste au premier plan : il écoute les exécutions demandées depuis l'interface
Prefect ou avec `prefect deployment run parkvision-pipeline/parkvision-nightly`. L'arrêter
(Ctrl+C) met le déploiement hors service.
"""

import argparse

from parkvision_flow import parkvision_pipeline

DEPLOYMENT_NAME = "parkvision-nightly"
DEFAULT_CRON = "0 3 * * *"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-schedule", action="store_true", help="pas de planification, déclenchement manuel")
    parser.add_argument("--cron", default=DEFAULT_CRON, help="expression cron (défaut : tous les jours à 03:00)")
    args = parser.parse_args()
    parkvision_pipeline.serve(
        name=DEPLOYMENT_NAME,
        cron=None if args.no_schedule else args.cron,
        tags=["data", "dvc"],
    )


if __name__ == "__main__":
    main()
