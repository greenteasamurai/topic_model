import os

MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

_ANTHROPIC_CLIENT = None
_DEEPSEEK_CLIENT = None


class _Content:
    def __init__(self, text):
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [_Content(text)]


class _DeepSeekMessages:
    _MODEL_MAP = {
        MODEL_SONNET: "deepseek-chat",
        MODEL_HAIKU: "deepseek-chat",
    }

    def __init__(self, client):
        self._client = client

    def create(self, *, model, max_tokens, system, messages):
        resolved = self._MODEL_MAP.get(model, model)
        response = self._client.chat.completions.create(
            model=resolved,
            messages=[
                {"role": "system", "content": system},
                *[{"role": m["role"], "content": m["content"]} for m in messages],
            ],
            max_tokens=max_tokens,
        )
        return _Response(response.choices[0].message.content or "")


class _DeepSeekWrapper:
    def __init__(self):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")
        import openai
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=10,
        )
        self.messages = _DeepSeekMessages(self._client)


def get_llm_client():
    global _ANTHROPIC_CLIENT, _DEEPSEEK_CLIENT
    provider = (os.environ.get("LLM_PROVIDER") or "deepseek").lower()
    if provider == "anthropic":
        if _ANTHROPIC_CLIENT is None:
            import anthropic
            _ANTHROPIC_CLIENT = anthropic.Anthropic()
        return _ANTHROPIC_CLIENT
    if _DEEPSEEK_CLIENT is None:
        _DEEPSEEK_CLIENT = _DeepSeekWrapper()
    return _DEEPSEEK_CLIENT
