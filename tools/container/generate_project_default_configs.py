"""Generate default configuration file template for each project."""
from pathlib import Path

from core.settings import (
    DEFAULT_CONFIG_JSON_FILENAME,
    BackboneConfig,
    ExperimentSettings,
)


def generate():
    """Generate default settings for each project."""
    insert_text = "<insert>"
    for project in Path("projects").glob("*"):
        if project.is_dir():
            ExperimentSettings(
                output_directory=insert_text,
                data_directory=insert_text,
                backbone=BackboneConfig(insert_text, {insert_text: insert_text}),
            ).serialize(str(project / DEFAULT_CONFIG_JSON_FILENAME), mark_env=False)


if __name__ == "__main__":
    generate()
