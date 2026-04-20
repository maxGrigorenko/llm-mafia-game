"""
PromptBuilder class for the LLM Mafia Game Competition.
Responsible for building prompts for LLM based on player role and language.
"""
from game_templates import (
    Role,
    GAME_RULES,
    CONFIRMATION_VOTE_EXPLANATIONS,
    PROMPT_TEMPLATES,
    CONFIRMATION_VOTE_TEMPLATES,
    THINKING_TAGS,
)


class PromptBuilder:
    """Builds prompts for LLM based on player role and language."""

    def _get_mafia_list(self, player, mafia_members, language):
        """
        Build the string listing Mafia allies for the current player.

        Args:
            player: The Player instance (used to exclude self from list).
            mafia_members (list): List of all mafia player instances.
            language (str): Language for the response.

        Returns:
            str: Formatted string of alive Mafia allies.
        """
        mafia_names = [
            p.player_name for p in mafia_members if p != player and p.alive
        ]

        if language == "Spanish":
            return f"{', '.join(mafia_names) if mafia_names else 'Ninguno (eres el único miembro de la Mafia que queda)'}"
        elif language == "French":
            return f"{', '.join(mafia_names) if mafia_names else 'Aucun (vous êtes le seul membre de la Mafia restant)'}"
        elif language == "Korean":
            return f"{', '.join(mafia_names) if mafia_names else '없음 (당신이 유일하게 남은 마피아입니다)'}"
        else:  # English and default
            return f"{', '.join(mafia_names) if mafia_names else 'None (you are the only Mafia left)'}"

    def _get_discussion_context(self, player, all_players, full_discussion_history):
        """
        Get appropriate discussion context based on whether player uses graph.

        For graph users: short recent history + graph state.
        For non-graph users: full history.

        Args:
            player: The Player instance.
            all_players (list): List of all players.
            full_discussion_history (str): Complete discussion history.

        Returns:
            str: Discussion context to include in prompt.
        """
        if not player.use_graph or player.trust_graph is None:
            # Non-graph players get full history
            return full_discussion_history

        # === Graph-based players get compressed context ===

        # Try to get last round's discussion
        last_round = player.discussion_history_last_round_without_thinking()

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
        graph_prompt = player.graph_to_prompt(all_players)

        # Combine graph + recent discussion
        context = f"""{graph_prompt}

=== RECENT DISCUSSION ===
{recent_discussion}"""

        return context

    def build_discussion_prompt(self, player, game_state, all_players, mafia_members=None, discussion_history=None):
        """
        Generate a discussion prompt for the player based on their role.
        All player references in prompts use player_name (not model_name).

        Args:
            player: The Player instance.
            game_state (dict): The current state of the game.
            all_players (list): List of all players in the game.
            mafia_members (list, optional): List of mafia members (only for Mafia role).
            discussion_history (str, optional): History of previous discussions.

        Returns:
            str: The prompt for the player.
        """
        if discussion_history is None:
            discussion_history = ""

        context = self._get_discussion_context(player, all_players, discussion_history)

        # Get list of alive player names (using player_name only)
        player_names = [p.player_name for p in all_players if p.alive]

        # Get the appropriate language, defaulting to English if not supported
        language = player.language if player.language in GAME_RULES else "English"

        # Get game rules for the player's language
        game_rules = GAME_RULES[language]

        if player.role == Role.MAFIA:
            mafia_list = self._get_mafia_list(player, mafia_members or [], language)

            prompt = PROMPT_TEMPLATES[language][Role.MAFIA].format(
                model_name=player.player_name,  # Use player_name in prompts (not model_name)
                game_rules=game_rules,
                mafia_members=mafia_list,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=context,
            )
        elif player.role == Role.DOCTOR:
            prompt = PROMPT_TEMPLATES[language][Role.DOCTOR].format(
                model_name=player.player_name,  # Use player_name in prompts (not model_name)
                game_rules=game_rules,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=context,
            )
        else:  # Role.VILLAGER
            prompt = PROMPT_TEMPLATES[language][Role.VILLAGER].format(
                model_name=player.player_name,  # Use player_name in prompts (not model_name)
                game_rules=game_rules,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=context,
            )

        return prompt

    def build_confirmation_vote_prompt(self, player, game_state, all_players, discussion_history):
        """
        Generate a confirmation vote prompt for the player.
        The player to eliminate is identified by player_name.

        Args:
            player: The Player instance.
            game_state (dict): The current state of the game, including who is up for elimination.
            all_players (list): List of all players.
            discussion_history (str): Discussion history string.

        Returns:
            str: The confirmation vote prompt for the player.
        """
        player_to_eliminate = game_state["confirmation_vote_for"]
        game_state_str = game_state["game_state"]

        context = self._get_discussion_context(player, all_players, discussion_history)

        # Get the appropriate language, defaulting to English if not supported
        language = (
            player.language
            if player.language in CONFIRMATION_VOTE_EXPLANATIONS
            else "English"
        )

        # Get confirmation vote explanation for the player's language
        confirmation_explanation = CONFIRMATION_VOTE_EXPLANATIONS[language].format(
            player_to_eliminate=player_to_eliminate
        )

        if player.role == Role.VILLAGER:
            role_string = "Villager"
        elif player.role == Role.DOCTOR:
            role_string = "Doctor"
        elif player.role == Role.MAFIA:
            role_string = "Mafia"
        else:
            print("Error: role not found")
            role_string = "Participant"

        # Generate prompt using player_name for identification
        prompt = CONFIRMATION_VOTE_TEMPLATES[language].format(
            model_name=player.player_name,  # Use player_name in prompts (not model_name)
            role_string=role_string,
            player_to_eliminate=player_to_eliminate,
            confirmation_explanation=confirmation_explanation,
            game_state_str=game_state_str,
            thinking_tag=THINKING_TAGS[language],
            discussion_history=context,
        )

        return prompt
