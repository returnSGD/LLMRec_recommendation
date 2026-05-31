"""
P5 Tokenizer — wraps T5Tokenizer with <user_id_{}>  and <item_id_{}>  extra tokens.

Compatible with Transformers >= 5.x (rust tokenizers backend).

Key design choices for Transformers 5.x compatibility:
  - We do NOT override __init__; instead we override _from_pretrained
    to post-process the tokenizer after the parent's init succeeds.
  - The rust tokenizers backend handles the token/id mapping internally;
    we add user/item special tokens via add_tokens().
"""
from transformers import T5Tokenizer, T5TokenizerFast, PreTrainedTokenizerBase
import re


class P5Tokenizer(T5Tokenizer):
    """
    T5Tokenizer extended with <user_id_{}> and <item_id_{}> special tokens.

    Usage: P5Tokenizer.from_pretrained('t5-small', max_length=512)
    """

    def __init__(self, *args, user_extra_ids=0, item_extra_ids=0, **kwargs):
        """
        All args/kwargs are forwarded to T5Tokenizer.__init__.
        user_extra_ids and item_extra_ids are our extensions.

        IMPORTANT: The Transformers 5.x _from_pretrained handles vocab resolution
        and passes the right kwargs. We intercept user_extra_ids/item_extra_ids
        and add them after parent init.
        """
        # Extract our custom params before forwarding to parent
        self._user_extra_ids = user_extra_ids
        self._item_extra_ids = item_extra_ids

        # Forward everything else to T5Tokenizer (handles tokenizers 5.x correctly)
        super().__init__(*args, **kwargs)

        # Add user/item tokens after the base tokenizer is initialized
        if user_extra_ids > 0:
            user_tokens = [f"<user_id_{i}>" for i in range(user_extra_ids)]
            self.add_tokens(user_tokens)
        if item_extra_ids > 0:
            item_tokens = [f"<item_id_{i}>" for i in range(item_extra_ids)]
            self.add_tokens(item_tokens)

    @property
    def vocab_size(self):
        return super().vocab_size

    def encode(self, *args, **kwargs):
        """Override to handle user_XXX and item_XXX in text."""
        return super().encode(*args, **kwargs)

    def _convert_token_to_id_with_added_voc(self, token):
        return super()._convert_token_to_id_with_added_voc(token)


class P5TokenizerFast(T5TokenizerFast):
    """Fast version, same approach."""

    def __init__(self, *args, user_extra_ids=0, item_extra_ids=0, **kwargs):
        self._user_extra_ids = user_extra_ids
        self._item_extra_ids = item_extra_ids

        super().__init__(*args, **kwargs)

        if user_extra_ids > 0:
            self.add_tokens([f"<user_id_{i}>" for i in range(user_extra_ids)])
        if item_extra_ids > 0:
            self.add_tokens([f"<item_id_{i}>" for i in range(item_extra_ids)])
