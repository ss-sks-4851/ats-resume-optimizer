from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()   # reads .env so API keys become available

MODELS = {
    "google": ("google_genai", "gemini-3.8-flash"),
    "groq":   ("groq", "openai/gpt-oss-120b"),
}

def get_llm(provider: str = "google", model: str | None = None, temperature: float = 0):
    if provider not in MODELS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(MODELS)}")

    model_provider, default_model = MODELS[provider]
    return init_chat_model(
        model or default_model,
        model_provider=model_provider,
        temperature=temperature, 
        timeout=90,
        max_retries=2,
    )

def text_of(response) -> str:
    """Return the reply as a plain string.
    Some models return a list of content blocks instead of a string."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )