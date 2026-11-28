"""Rejoue une vidéo dans l'API comme le ferait le navigateur, sans ouvrir de page.

L'analyse vidéo suit la lecture : une image n'est analysée que lorsque le lecteur déclare
être arrivé à son instant. Ce script joue le rôle du lecteur : il envoie l'instant de
lecture juste après la dernière image analysée, puis attend la fin de la session.

    python reports/monitoring/replay_video.py reports/demo/demo_2013-03-15.mp4
    python reports/monitoring/replay_video.py "https://www.youtube.com/watch?v=..."

L'API doit tourner (voir docs/monitoring.md). Avec PARKVISION_MONITORING_ENABLED=true,
chaque image analysée est ajoutée au journal de monitoring.
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

TERMINAL = {"completed", "failed", "cancelled"}


def replay(source: str, base: str, timeout: float) -> dict:
    with httpx.Client(base_url=base, timeout=30) as client:
        if source.startswith(("http://", "https://")):
            response = client.post("/video/youtube", json={"url": source})
        else:
            video = Path(source)
            with video.open("rb") as handle:
                response = client.post("/video/upload", files={"file": (video.name, handle, "video/mp4")})
        response.raise_for_status()
        session_id = response.json()["id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            session = client.get(f"/video/sessions/{session_id}").json()
            if session["status"] in TERMINAL:
                return session
            frontier = session.get("analyzed_frontier_seconds")
            target = 0.0 if frontier is None else frontier + 1.0
            client.post(f"/video/sessions/{session_id}/playback",
                        json={"current_time_seconds": target, "playing": True, "ended": False})
            time.sleep(0.3)
        client.delete(f"/video/sessions/{session_id}")
        sys.exit("délai dépassé : session annulée")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="fichier MP4 local ou lien YouTube")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    session = replay(args.source, args.base, args.timeout)
    print(f"{args.source} : {session['status']}, {session['frames_processed']} images analysées")
    if session["status"] != "completed":
        sys.exit(f"échec : {session.get('error')}")


if __name__ == "__main__":
    main()
