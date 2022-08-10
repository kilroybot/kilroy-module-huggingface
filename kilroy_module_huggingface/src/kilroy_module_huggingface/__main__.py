"""Main script.

This module provides basic CLI entrypoint.

"""
import asyncio
import logging
from asyncio import FIRST_EXCEPTION
from enum import Enum
from logging import Logger
from typing import Dict

import typer
from kilroy_module_server_py_sdk import ModuleServer

from kilroy_module_huggingface.config import get_config
from kilroy_module_huggingface.module import HuggingfaceModule

cli = typer.Typer()  # this is actually callable and thus can be an entry point


class Verbosity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def get_logger(verbosity: Verbosity) -> Logger:
    logging.basicConfig()
    logger = logging.getLogger("kilroy-face-twitter")
    logger.setLevel(verbosity.value)
    return logger


async def run(config: Dict, logger: Logger) -> None:
    module = await HuggingfaceModule.build(**config.get("module", {}))
    server = ModuleServer(module, logger)

    tasks = (
        asyncio.create_task(module.init()),
        asyncio.create_task(server.run(**config.get("server", {}))),
    )

    done, pending = await asyncio.wait(tasks, return_when=FIRST_EXCEPTION)

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    for task in done:
        task.result()

    await module.cleanup()


@cli.command()
def main(
    verbosity: Verbosity = typer.Option(
        default="INFO", help="Verbosity level."
    )
) -> None:
    """Command line interface for kilroy-module-huggingface."""

    config = get_config()
    logger = get_logger(verbosity)

    asyncio.run(run(config, logger))


if __name__ == "__main__":
    # entry point for "python -m"
    cli()
