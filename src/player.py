"""
Player class for the LLM Mafia Game Competition.
"""
import re
import config
from openrouter import get_llm_response
from game_templates import (
    Role,
    GAME_RULES,
    CONFIRMATION_VOTE_EXPLANATIONS,
    PROMPT_TEMPLATES,
    CONFIRMATION_VOTE_TEMPLATES,
    THINKING_TAGS,
    ACTION_PATTERNS,
    VOTE_PATTERNS,
    CONFIRMATION_VOTE_PATTERNS,
)
from trust_graph import TrustGraph


class Player:
    """Represents an LLM player in the Mafia game."""

    def __init__(self, model_name, player_name, role, language=None, use_graph=False, game=None):
        """
        Initialize a player.

        Args:
            model_name (str): The name of the LLM model (hidden from other players, used only for API calls).
            player_name (str): The unique visible name of the player in the game (used for all identification).
            role (Role): The role of the player in the game.
            language (str, optional): The language for the player. Defaults to English.
            use_graph (bool, optional): Whether this player uses graph-based reasoning.
            game (MafiaGame, optional): Reference to the game instance.
        """
        self.model_name = model_name      # Hidden: only used for LLM API calls
        self.player_name = player_name    # Visible: used for all in-game identification
        self.role = role
        self.alive = True
        self.use_graph = use_graph
        self.trust_graph = None           # TrustGraph instance (replaces self.graph)
        self.protected = False            # Whether the player is protected by the doctor
        self.language = language if language else "English"
        self.game = game

    def __str__(self):
        """Return a string representation of the player."""
        return f"{self.player_name} ({self.role.value}) [Model: {self.model_name}] ({self.use_graph=})"

    # ------------------------------------------------------------------
    # Graph delegation methods
    # ------------------------------------------------------------------

    def init_graph(self, all_players):
        """
        Initialize subjective directed trust graph for this player.
        Delegates to TrustGraph.
        """
        if not self.use_graph:
            return None

        self.trust_graph = TrustGraph(self.player_name, self.role, all_players)
        return self.trust_graph.graph

    def graph_to_prompt(self, all_players):
        """
        Convert subjective graph to compact text description for LLM prompt.
        Delegates to TrustGraph.
        """
        if not self.use_graph or self.trust_graph is None:
            return ""

        return self.trust_graph.to_prompt(all_players)

    def update_graph(self, all_players, current_round):
        """
        Update the subjective trust graph based on the last round's discussion.
        Delegates to TrustGraph.

        Args:
            all_players (list): List of all players in the game.
            current_round (int): Current round number.
        """
        if not self.use_graph or self.trust_graph is None:
            return

        last_round_history = self.discussion_history_last_round_without_thinking()

        self.trust_graph.update(
            all_players=all_players,
            current_round=current_round,
            discussion_history=last_round_history,
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------
    # Discussion history helpers
    # ------------------------------------------------------------------

    def discussion_history_without_thinking(self):
        return self.game.discussion_history_without_thinking()

    def discussion_history_last_round_without_thinking(self):
        return self.game.discussion_history_last_round_without_thinking()

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _find_target_player(self, target_name, all_players, exclude_mafia=False):
        """
        Find a target player by player_name.

        Args:
            target_name (str): The player_name of the target player.
            all_players (list): List of all players in the game.
            exclude_mafia (bool, optional): Whether to exclude Mafia members from targets.

        Returns:
            Player or None: The target player if found, None otherwise.
        """
        for player in all_players:
            if not player.alive:
                continue

            if exclude_mafia and player.role == Role.MAFIA:
                continue

            # Match against player_name only (not model_name)
            if target_name.lower() in player.player_name.lower():
                return player

        return None

    def generate_prompt(
        self, game_state, all_players, mafia_members=None, discussion_history=None
    ):
        """
        Generate a prompt for the player based on their role.
        All player references in prompts use player_name (not model_name).

        Args:
            game_state (dict): The current state of the game.
            all_players (list): List of all players in the game.
            mafia_members (list, optional): List of mafia members (only for Mafia role).
            discussion_history (str, optional): History of previous discussions.

        Returns:
            str: The prompt for the player.
        """
        if discussion_history is None:
            discussion_history = ""

        context = self._get_discussion_context(all_players, discussion_history)

        # Get list of alive player names (using player_name only)
        player_names = [p.player_name for p in all_players if p.alive]

        # Get the appropriate language, defaulting to English if not supported
        language = self.language if self.language in GAME_RULES else "English"

        # Get game rules for the player's language
        game_rules = GAME_RULES[language]

        if self.role == Role.MAFIA:
            # For Mafia members: list allies by player_name
            mafia_names = [
                p.player_name for p in mafia_members if p != self and p.alive
            ]
            mafia_list = f"{', '.join(mafia_names) if mafia_names else 'None (you are the only Mafia left)'}"
            if language == "Spanish":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else 'Ninguno (eres el único miembro de la Mafia que queda)'}"
            elif language == "French":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else 'Aucun (vous êtes le seul membre de la Mafia restant)'}"
            elif language == "Korean":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else '없음 (당신이 유일하게 남은 마피아입니다)'}"

            prompt = PROMPT_TEMPLATES[language][Role.MAFIA].format(
                model_name=self.player_name,  # Use player_name in prompts (not model_name)
                game_rules=game_rules,
                mafia_members=mafia_list,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=context,
            )
        elif self.role == Role.DOCTOR:
            prompt = PROMPT_TEMPLATES[language][Role.DOCTOR].format(
                model_name=self.player_name,  # Use player_name in prompts (not model_name)
                game_rules=game_rules,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=context,
            )
        else:  # Role.VILLAGER
            prompt = PROMPT_TEMPLATES[language][Role.VILLAGER].format(
                model_name=self.player_name,  # Use player_name in prompts (not model_name)
                game_rules=game_rules,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=context,
            )

        return prompt

    def _get_discussion_context(self, all_players, full_discussion_history):
        """
        Get appropriate discussion context based on whether player uses graph.

        For graph users: short recent history + graph state
        For non-graph users: full history

        Args:
            all_players (list): List of all players.
            full_discussion_history (str): Complete discussion history.

        Returns:
            str: Discussion context to include in prompt.
        """
        if not self.use_graph or self.trust_graph is None:
            # Non-graph players get full history
            return full_discussion_history

        # === Graph-based players get compressed context ===

        # Try to get last round's discussion
        last_round = self.discussion_history_last_round_without_thinking()

        # Fallback logic
        MIN_CONTEXT_LENGTH = 500
        MAX_CONTEXT_LENGTH = 3000

        if last_round and len(last_round.strip()) >= MIN_CONTEXT_LENGTH:
            recent_discussion = last_round
        elif full_discussion_history:
            # Fallback: take last N characters of full history
            recent_discussion = full_discussion_history[-MAX_CONTEXT_LENGTH:]
            newline_pos = recent_discussion.find('\n')
            if 0 < newline_pos < 200:
                recent_discussion = recent_discussion[newline_pos + 1:]
            recent_discussion = f"[...previous discussion...]\n{recent_discussion}"
        else:
            recent_discussion = "[No discussion yet]"

        # Truncate if too long
        if len(recent_discussion) > MAX_CONTEXT_LENGTH:
            recent_discussion = recent_discussion[-MAX_CONTEXT_LENGTH:]
            newline_pos = recent_discussion.find('\n')
            if 0 < newline_pos < 200:
                recent_discussion = recent_discussion[newline_pos + 1:]
            recent_discussion = f"[...truncated...]\n{recent_discussion}"

        # Get graph representation (uses player_name for all nodes)
        graph_prompt = self.graph_to_prompt(all_players)

        # Combine graph + recent discussion
        context = f"""{graph_prompt}

=== RECENT DISCUSSION ===
{recent_discussion}"""

        return context

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def get_response(self, prompt):
        """
        Get a response from the LLM model using OpenRouter API.
        Uses model_name for the API call (hidden from other players).

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The response from the model with private thoughts removed.
        """
        print(f"\n\n{len(prompt)=} {prompt=}\n\n")
        response = get_llm_response(self.model_name, prompt)

        # Remove any <think></think> tags and their contents before sharing with other players
        cleaned_response = re.sub(r"<[tT][hH][iI][nN][kK]>.*?</[tT][hH][iI][nN][kK]>", "", response, flags=re.DOTALL)

        # Clean up any extra whitespace that might have been created
        cleaned_response = re.sub(r"\n\s*\n", "\n\n", cleaned_response)
        cleaned_response = cleaned_response.strip()

        return cleaned_response

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_night_action(self, response, all_players):
        """
        Parse the night action from the player's response.
        Targets are identified by player_name.

        Args:
            response (str): The response from the player (already cleaned of thinking tags).
            all_players (list): List of all players in the game.

        Returns:
            tuple: (action_type, target_player) or (None, None) if no valid action.
        """
        if self.role == Role.MAFIA:
            pattern = ACTION_PATTERNS.get(self.language, ACTION_PATTERNS["English"])[
                Role.MAFIA
            ]
            match = re.search(pattern, response, re.IGNORECASE)

            if match:
                target_name = match.group(1).strip()
                # Find target by player_name, excluding Mafia members
                target_player = self._find_target_player(
                    target_name, all_players, exclude_mafia=True
                )
                if target_player:
                    return "kill", target_player
            return None, None

        elif self.role == Role.DOCTOR:
            pattern = ACTION_PATTERNS.get(self.language, ACTION_PATTERNS["English"])[
                Role.DOCTOR
            ]
            match = re.search(pattern, response, re.IGNORECASE)

            if match:
                target_name = match.group(1).strip()
                # Find target by player_name
                target_player = self._find_target_player(target_name, all_players)
                if target_player:
                    return "protect", target_player
            return None, None
        else:
            # Villagers don't have night actions
            return None, None

    def parse_day_vote(self, response, all_players):
        """
        Parse the day vote from the player's response.
        Vote target is identified by player_name.

        Args:
            response (str): The response from the player (already cleaned of thinking tags).
            all_players (list): List of all players in the game.

        Returns:
            Player or None: The player being voted for, or None if no valid vote.
        """
        pattern = VOTE_PATTERNS.get(self.language, VOTE_PATTERNS["English"])
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            target_name = match.group(1).strip()
            # Find target by player_name
            return self._find_target_player(target_name, all_players)
        return None

    def get_confirmation_vote(self, game_state, all_players, discussion_history):
        """
        Get a confirmation vote from the player on whether to eliminate another player.
        The player to eliminate is identified by player_name.

        Args:
            game_state (dict): The current state of the game, including who is up for elimination.
            all_players (list): List of all players.
            discussion_history (str): Discussion history string.

        Returns:
            str: "agree" or "disagree" indicating the player's vote
        """
        player_to_eliminate = game_state["confirmation_vote_for"]
        game_state_str = game_state["game_state"]

        context = self._get_discussion_context(all_players, discussion_history)

        # Get the appropriate language, defaulting to English if not supported
        language = (
            self.language
            if self.language in CONFIRMATION_VOTE_EXPLANATIONS
            else "English"
        )

        # Get confirmation vote explanation for the player's language
        confirmation_explanation = CONFIRMATION_VOTE_EXPLANATIONS[language].format(
            player_to_eliminate=player_to_eliminate
        )

        if self.role == Role.VILLAGER:
            role_string = "Villager"
        elif self.role == Role.DOCTOR:
            role_string = "Doctor"
        elif self.role == Role.MAFIA:
            role_string = "Mafia"
        else:
            print("Error: role not found")
            role_string = "Participant"

        # Generate prompt using player_name for identification
        prompt = CONFIRMATION_VOTE_TEMPLATES[language].format(
            model_name=self.player_name,  # Use player_name in prompts (not model_name)
            role_string=role_string,
            player_to_eliminate=player_to_eliminate,
            confirmation_explanation=confirmation_explanation,
            game_state_str=game_state_str,
            thinking_tag=THINKING_TAGS[language],
            discussion_history=context
        )

        response = self.get_response(prompt)

        # Parse the response for agree/disagree based on language
        language = (
            self.language if self.language in CONFIRMATION_VOTE_PATTERNS else "English"
        )
        if re.search(CONFIRMATION_VOTE_PATTERNS[language]["agree"], response.lower()):
            return "agree"
        else:
            return "disagree"
