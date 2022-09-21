from pathlib import Path
from typing import Annotated, Iterable, Literal, Optional, TextIO, Tuple, Union

from omegaconf import OmegaConf
from platformdirs import user_cache_dir
from pydantic import BaseModel, Extra, Field
from pydantic.env_settings import BaseSettings, SettingsSourceCallable

from kilroy_module_huggingface import resource_text
from kilroy_module_huggingface.modules.basic import Params as BasicParams
from kilroy_module_huggingface.modules.reward import (
    Params as RewardModelParams,
)

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


class CommonConfig(BaseConfig):
    server: ServerParams = ServerParams()
    state_directory: Path = CACHE_DIR / "kilroy-module-huggingface" / "state"


class BasicConfig(CommonConfig):
    module_type: Literal["basic"] = "basic"
    module: BasicParams


class RewardModelConfig(CommonConfig):
    module_type: Literal["rewardModel"] = "rewardModel"
    module: RewardModelParams


class Config(BaseModel):
    __root__: Annotated[
        Union[BasicConfig, RewardModelConfig],
        Field(discriminator="module_type"),
    ]


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
