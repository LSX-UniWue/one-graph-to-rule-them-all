from collections import deque, Counter
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Set

from gensim.models import Word2Vec

import wuenlp
from loguru import logger
from wuenlp.impl.UIMANLPStructs import UIMAToken, UIMASentence, UIMACharacterReference

wuenlp.set_typesystem(Path("LOTRTypesystem.xml"))
from cas import LOTRDocument, LOTREntityReference, SimpleCorefEntity, get_all_xmis_without_interactions


def train_w2v(tokens, ITERATIONS=20, VECTOR_SIZE=300, MINIMUM_TOKEN_OCCURRENCES=5):
    NUM_CPUS = min(cpu_count(), 10)

    model = Word2Vec(tokens, vector_size=VECTOR_SIZE, workers=NUM_CPUS, epochs=ITERATIONS,
                     min_count=MINIMUM_TOKEN_OCCURRENCES)

    return model


def get_w2v_character_tokens(path: Path, non_character_representation: str = "text",
                             reference_type=LOTREntityReference) -> (List, Counter):
    assert non_character_representation in (
        "text", None, "UNKNOWN"), "non_character_representation must be either of ('text', None, 'UNKNOWN')"
    logger.info(f"Processing file {path}...")

    doc = LOTRDocument.from_xmi(path)

    characters = Counter()

    def _get_representation(sentence: UIMASentence, reference_type=LOTREntityReference):
        for token in sentence.tokens:
            entity_references: List[reference_type] = token.covering(reference_type)
            if not entity_references:
                if non_character_representation == "text":
                    yield token.text
                elif non_character_representation is None:
                    yield None
                else:
                    yield "<UNKNOWN>"
            else:
                entity_reference: reference_type = entity_references[0]
                if token.begin == entity_reference.begin:
                    if type(entity_reference) == LOTREntityReference and entity_reference.simple_coref.anno is not None:
                        name = entity_reference.simple_coref.name
                    elif type(entity_reference) == UIMACharacterReference:
                        name = entity_reference.text.title().replace(" ", "_")
                    else:
                        name = entity_reference.text
                    characters.update({name: 1})
                    yield name
                else:
                    yield None

    sentences = [[token for token in _get_representation(sentence, reference_type=reference_type) if token is not None]
                 for sentence in doc.sentences]

    return sentences, characters


def main():
    use_movies = False
    vector_size = 300

    out_dir = Path("data/tolkien_w2v_movies" if use_movies else "data/tolkien_w2v")
    out_dir.mkdir(exist_ok=True)
    for min_occurrences in (0, 5):
        for non_character_representation in ("text", ):#None, "UNKNOWN"):
            logger.info(f"Building {non_character_representation}-embeddings with min_occurrences {min_occurrences}")
            all_book_xmis = get_all_xmis_without_interactions()
            lotr_xmis = [path for path in all_book_xmis if "lotr" in path.stem]
            hobbit_xmi = [path for path in all_book_xmis if "hobbit" in path.stem]
            silmarillion_xmi = [path for path in all_book_xmis if "silmarillion" in path.stem]
            movie_xmis = [Path("movies/lotr_1.xmi"), Path("movies/lotr_2.xmi"),
                          Path("movies/lotr_3.xmi")]
            in_file_dict = {"lotr_movies": movie_xmis} if use_movies else \
                {"all": all_book_xmis,
                 "lotr": lotr_xmis,
                 "hobbit": hobbit_xmi,
                 "silmarillion": silmarillion_xmi}

            print(in_file_dict)
            for name, in_files in in_file_dict.items():
                sentences = []
                characters = Counter()
                logger.info(
                    f"Building {non_character_representation}-embeddings with min_occurrences {min_occurrences} for {name}...")
                for file in in_files:
                    _tokens, _characters = get_w2v_character_tokens(file,
                                                                    non_character_representation=non_character_representation,
                                                                    reference_type=UIMACharacterReference if use_movies else LOTREntityReference)

                    sentences.extend(_tokens)
                    characters.update(_characters)

                print(characters)
                model = train_w2v(sentences, MINIMUM_TOKEN_OCCURRENCES=min_occurrences, VECTOR_SIZE=vector_size)

                save_model(characters, min_occurrences, model, name, non_character_representation, out_dir, vector_size)


def save_model(characters, min_occurrences, model, name, non_character_representation, out_dir, vector_size):
    vectors_csv_ = out_dir / (f"{name}_{min_occurrences}_{non_character_representation}_dim_{vector_size}_vectors.csv")
    words_csv_ = out_dir / (f"{name}_{min_occurrences}_{non_character_representation}_dim_{vector_size}_words.csv")
    model_file = out_dir / f"{name}_{min_occurrences}_w2v_{non_character_representation}_dim_{vector_size}.bin"

    model.save(str(model_file))
    with vectors_csv_.open("w") as vectors:
        with words_csv_.open("w") as words:
            for word, idx in model.wv.key_to_index.items():
                if word in characters:
                    vectors.write("\t".join(map(str, model.wv[word].tolist())))
                    vectors.write("\n")
                    words.write(word + "\n")


if __name__ == '__main__':
    main()
