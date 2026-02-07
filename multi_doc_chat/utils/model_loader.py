import os
import sys
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_groq import ChatGroq

from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException


# API KEY MANAGER 


class ApiKeyManager:
    def __init__(self, require_google: bool = False, require_groq: bool = False):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        missing = []
        if require_google and not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if require_groq and not self.groq_api_key:
            missing.append("GROQ_API_KEY")

        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException(
                f"Missing API keys: {', '.join(missing)}", sys
            )

        log.info(
            "API keys loaded",
            google="yes" if self.google_api_key else "no",
            groq="yes" if self.groq_api_key else "no",
        )

    def get_google_key(self) -> str:
        return self.google_api_key

    def get_groq_key(self) -> str:
        return self.groq_api_key



# MODEL LOADER


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        self.config = load_config()

        embedding_provider = self.config["embedding_model"]["provider"]
        llm_provider = os.getenv("LLM_PROVIDER", "groq")

        self.api_key_mgr = ApiKeyManager(
            require_google=(embedding_provider == "google" or llm_provider == "google"),
            require_groq=(llm_provider == "groq"),
        )


    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)

            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get_google_key()
            )

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "groq")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get_groq_key(),
                temperature=temperature,
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get_google_key(),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


# LOCAL TEST

if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embeddings
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    emb_result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result Length: {len(emb_result)}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    llm_result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {llm_result.content}")
