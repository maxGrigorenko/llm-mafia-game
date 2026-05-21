import re
import time

class PhraseGraph:
    """Helper class for building per-phrase relationship graphs.

    The class provides a single static method that scans a message for mentions
    of other alive players and assigns an edge weight (+1, -1, or 0) based on
    the presence of positive or negative keywords anywhere in the message.
    """

    # Case‑insensitive keyword sets used for sentiment detection.
    _POSITIVE_KEYWORDS = {
        "trust",
        "innocent",
        "ally",
        "friend",
        "protect",
        "good",
        "villager",
        "help",
        "safe",
        "honest",
    }

    _NEGATIVE_KEYWORDS = {
        "mafia",
        "suspect",
        "kill",
        "vote",
        "guilty",
        "liar",
        "evil",
        "enemy",
        "bad",
        "threat",
        "danger",
    }

    @staticmethod
    def build_graph(message: str, speaker: str, alive_players: list,
                    current_round: int, phase: str) -> dict:
        """Build a per‑phrase relationship graph for a single spoken message.

        Parameters
        ----------
        message : str
            The message the speaker has just produced (already cleaned of
            private thinking tags).
        speaker : str
            The ``player_name`` of the player who uttered the message.
        alive_players : list[Player]
            List of currently alive ``Player`` objects.
        current_round : int
            Current round number.
        phase : str
            Current game phase string (e.g., ``"day_discussion"``,
            ``"day_voting"``, ``"night"``).

        Returns
        -------
        dict
            A dictionary with the following keys:

            * ``"round"`` : ``current_round``
            * ``"phase"`` : ``phase``
            * ``"timestamp"`` : ``float`` – ``time.time()`` value
            * ``"speaker"`` : ``speaker``
            * ``"edges"`` : ``list[tuple]`` – each tuple is
              ``(speaker_player_name, mentioned_player_name, weight)``
              where *weight* is ``1`` (positive), ``-1`` (negative) or ``0``
              (neutral).
        """
        # Determine which alive player names (other than the speaker) appear
        # in the message.  Use case‑insensitive word‑boundary matching.
        candidate_names = [
            p.player_name for p in alive_players
            if p.player_name.lower() != speaker.lower()
        ]

        mentioned = set()
        for name in candidate_names:
            if re.search(r'\b' + re.escape(name) + r'\b', message, re.IGNORECASE):
                mentioned.add(name)

        # Compute the edge weight based on keyword presence in the whole
        # message.  A negative keyword overrides everything.
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in PhraseGraph._NEGATIVE_KEYWORDS):
            weight = -1
        elif any(kw in msg_lower for kw in PhraseGraph._POSITIVE_KEYWORDS):
            weight = 1
        else:
            weight = 0

        edges = [(speaker, target, weight) for target in mentioned]

        return {
            "round": current_round,
            "phase": phase,
            "timestamp": time.time(),
            "speaker": speaker,
            "edges": edges,
        }
