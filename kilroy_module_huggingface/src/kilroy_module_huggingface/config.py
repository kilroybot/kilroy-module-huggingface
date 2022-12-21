from pathlib import Path
from typing import Iterable, Optional, TextIO, Tuple, Dict, Any

from omegaconf import OmegaConf
from platformdirs import user_cache_dir
from pydantic import BaseModel, Extra
from pydantic.env_settings import BaseSettings, SettingsSourceCallable

from kilroy_module_huggingface import resource_text

CACHE_DIR = Path(user_cache_dir("kilroybot"))


class BaseConfig(BaseSettings):
    class Config:
        env_prefix = "kilroy_module_huggingface_"
        env_nested_delimiter = "__"
        env_file = ".env"
        extra = Extra.allow

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> Tuple[SettingsSourceCallable, ...]:
            return env_settings, init_settings, file_secret_settings


class ServerParams(BaseModel):
    host: str = "0.0.0.0"
    port: int = 11000


class Config(BaseModel):
    server: ServerParams = ServerParams()
    module: Dict[str, Any] = {}
    state_directory: Path = CACHE_DIR / "kilroy-module-huggingface" / "state"


def get_config(
    f: Optional[TextIO] = None, overrides: Optional[Iterable[str]] = None
) -> Config:
    config = OmegaConf.create(resource_text("config.yaml"))
    if f is not None:
        config = OmegaConf.merge(config, OmegaConf.load(f))
    if overrides is not None:
        config = OmegaConf.merge(
            config, OmegaConf.from_dotlist(list(overrides))
        )
    config = OmegaConf.to_container(config, resolve=True)
    return Config.parse_obj(config)
