import spacy
from booknlp.booknlp import BookNLP
from os import cpu_count

from pathlib import Path

from loguru import logger

model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "big" if cpu_count() > 10 else "small",
    # "pronominalCorefOnly": False,
}

booknlp = BookNLP("en", model_params)

lotr_out = Path("data/silmarillion_out")

for file in sorted(Path("data/silmarillion").iterdir(), key=lambda file: file.name):
    if file.name.startswith("."):
        continue
    logger.info(f"Processing {file.name}")
    booknlp.process(str(file), lotr_out, file.stem)
