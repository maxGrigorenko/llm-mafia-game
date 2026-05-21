import re
import time
import json

from openrouter import get_llm_response


class PhraseGraph:
    """Helper class for building per-phrase relationship graphs.

    The class provides a static method that scans a message for mentions
    of other alive players and asks an LLM agent to assign edge weights
    (+1 positive, -1 negative, 0 neutral) based on the message content.
    """

    @staticmethod
    def _build_llm_prompt(message: str, speaker: str, mentioned_players: list) -> str:
        """Build a prompt that asks the LLM to rate the speaker's attitude
        toward each mentioned player.

        Parameters
        ----------
        message : str
            The message spoken by the player.
        speaker : str
            The name of the speaking player.
        mentioned_players : list[str]
            Names of players mentioned in the message.

        Returns
        -------
        str
            A prompt string for the LLM.
        """
        players_list = ", ".join(mentioned_players)
        prompt = (
            f"You are analyzing a Mafia game conversation.\n\n"
            f"The player \"{speaker}\" said the following message:\n"
            f"\"\"\"\n{message}\n\"\"\"\n\n"
            f"Based solely on this message, rate the attitude of \"{speaker}\" "
            f"toward each of the following players: {players_list}.\n\n"
            f"Use only these weight values:\n"
            f"  +1 = positive attitude (trust, support, alliance, innocence)\n"
            f"  -1 = negative attitude (suspicion, accusation, hostility, distrust)\n"
            f"   0 = neutral or no clear attitude\n\n"
            f"Respond ONLY with a valid JSON object mapping each player name to their weight.\n"
            f"Example: {{\"Alice\": 1, \"Bob\": -1, \"Carol\": 0}}\n\n"
            f"JSON response:"
        )
        return prompt

    @staticmethod
    def _parse_llm_weights(response: str, mentioned_players: list) -> dict:
        """Parse the LLM response to extract weight values for each mentioned player.

        Falls back to weight 0 (neutral) for any player that cannot be parsed.

        Parameters
        ----------
        response : str
            Raw string response from the LLM.
        mentioned_players : list[str]
            Names of players that should appear in the result.

        Returns
        -------
        dict
            Mapping of player_name -> weight (int: -1, 0, or 1).
        """
        weights = {name: 0 for name in mentioned_players}

        # Try to extract JSON object from the response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                for name in mentioned_players:
                    # Try exact match first, then case-insensitive
                    if name in parsed:
                        raw = parsed[name]
                    else:
                        raw = next(
                            (v for k, v in parsed.items() if k.lower() == name.lower()),
                            None,
                        )
                    if raw is not None:
                        try:
                            weight = int(raw)
                            # Clamp to valid range
                            weights[name] = max(-1, min(1, weight))
                        except (ValueError, TypeError):
                            weights[name] = 0
            except json.JSONDecodeError:
                pass

        return weights

    @staticmethod
    def build_graph(
        message: str,
        speaker: str,
        alive_players: list,
        current_round: int,
        phase: str,
        model_name: str,
    ) -> dict:
        """Build a per-phrase relationship graph for a single spoken message.

        The LLM model identified by ``model_name`` is asked to assign an edge
        weight for each player mentioned in the message.  The weights are:

        * ``+1`` – positive attitude (trust, support, innocence, …)
        * ``-1`` – negative attitude (suspicion, accusation, hostility, …)
        * ``0``  – neutral or no clear attitude

        Parameters
        ----------
        message : str
            The message the speaker has just produced (cleaned of private
            thinking tags).
        speaker : str
            The ``player_name`` of the player who uttered the message.
        alive_players : list[Player]
            List of currently alive ``Player`` objects.
        current_round : int
            Current round number.
        phase : str
            Current game phase string (e.g., ``"day_discussion"``,
            ``"day_voting"``, ``"night"``).
        model_name : str
            The LLM model identifier used to evaluate edge weights.

        Returns
        -------
        dict
            A dictionary with the following keys:

            * ``"round"``     : ``current_round``
            * ``"phase"``     : ``phase``
            * ``"timestamp"`` : ``float`` – ``time.time()`` value
            * ``"speaker"``   : ``speaker``
            * ``"edges"``     : ``list[tuple]`` – each tuple is
              ``(speaker_player_name, mentioned_player_name, weight)``
        """
        # Find which alive players (other than the speaker) are mentioned in
        # the message using case-insensitive word-boundary matching.
        candidate_names = [
            p.player_name
            for p in alive_players
            if p.player_name.lower() != speaker.lower()
        ]

        mentioned = []
        for name in candidate_names:
            if re.search(r"\b" + re.escape(name) + r"\b", message, re.IGNORECASE):
                mentioned.append(name)

        edges = []

        if mentioned:
            # Ask the LLM to rate the speaker's attitude toward each mentioned player
            prompt = PhraseGraph._build_llm_prompt(message, speaker, mentioned)
            try:
                llm_response = get_llm_response(model_name, prompt)
                weights = PhraseGraph._parse_llm_weights(llm_response, mentioned)
            except Exception as e:
                # On any error fall back to neutral weights
                print(f"[PhraseGraph] LLM weight evaluation failed for {speaker}: {e}")
                weights = {name: 0 for name in mentioned}

            edges = [(speaker, target, weights[target]) for target in mentioned]

        return {
            "round": current_round,
            "phase": phase,
            "timestamp": time.time(),
            "speaker": speaker,
            "edges": edges,
        }
