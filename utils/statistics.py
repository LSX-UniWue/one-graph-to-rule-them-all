from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Union

import matplotlib
import numpy as np
from numpy import cumsum
from scipy.ndimage.filters import uniform_filter1d

from matplotlib import pyplot as plt

from IPython.core.display import display
from loguru import logger
import pandas as pd
from wuenlp import set_typesystem
from wuenlp.impl.UIMANLPStructs import UIMADocument, UIMAToken, UIMACharacterReference, UIMAChapter

set_typesystem(Path("LOTRTypesystem.xml"))

from cas import get_all_xmis_with_interactions, LOTRDocument


def get_plots(path: Path):
    doc = UIMADocument.from_xmi(path)

    def is_name(token: UIMAToken):
        covering_reference: List[UIMACharacterReference] = token.covering(UIMACharacterReference)
        if covering_reference:
            if covering_reference[0].reference_type == "PROP":
                return 1
            else:
                return 0
        else:
            return 0

    def is_pronoun(token: UIMAToken):
        covering_reference: List[UIMACharacterReference] = token.covering(UIMACharacterReference)
        if covering_reference:
            if covering_reference[0].reference_type == "PRON":
                return 1
            else:
                return 0
        else:
            return 0

    def get_chapter_length(chapter: UIMAChapter):
        return
        tokens = chapter.tokens

        begin_index = to

    tokens = doc.tokens
    is_name_array = [is_name(token) for token in tokens]
    is_pronoun_array = [is_pronoun(token) for token in tokens]

    size = 10000
    name_ratio = np.convolve(is_name_array, np.ones(size) / size, mode='valid')
    pronoun_ratio = np.convolve(is_pronoun_array, np.ones(size) / size, mode='valid')

    chapters = doc.chapters

    chapter_positions = list(cumsum([get_chapter_length(chapter) for chapter in chapters]))
    chapter_names = [chapter.name for chapter in chapters]

    plt.plot(name_ratio)
    plt.plot(pronoun_ratio)
    plt.legend(("Names", "Pronouns"))
    plt.xticks(chapter_positions, chapter_names)
    plt.title(f"{path} {size}")
    plt.show()


name_map = {
    "hobbit_with_interactions.xmi": "Hobbit",
    "lotr_1_with_interactions.xmi": "LotR 1",
    "lotr_2_with_interactions.xmi": "LotR 2",
    "lotr_3_with_interactions.xmi": "LotR 3",
    "silmarillion_with_interactions.xmi": "Silmarillion",
}


def get_basic_statistics(path: Path) -> dict:
    doc = UIMADocument.from_xmi(path)

    character_references = doc.character_references
    num_explicit_references = len([cr for cr in character_references if cr.reference_type == "PROP"])
    num_pronouns = len([cr for cr in character_references if cr.reference_type == "PRON"])
    num_noms = len([cr for cr in character_references if cr.reference_type == "NOM"])
    num_character_mentions = len(character_references)

    stats = {
        "Text": name_map[path.name],
        "\#tokens": len(doc.tokens),
        "\#character mentions": num_character_mentions,
        # "\#pronouns": num_pronouns,
        # "\#explicit references": num_explicit_references,
        "\% Pronouns": num_pronouns / num_character_mentions * 100,
        "\% Nominal mentions": num_noms / num_character_mentions * 100,
        "\% Explicit references": num_explicit_references / num_character_mentions * 100,
    }

    logger.info(Counter([cr.reference_type for cr in character_references]))
    return stats


def get_mention_statistics(xmi_path_or_paths: Union[Path, List[Path]]) -> Counter:
    if type(xmi_path_or_paths) == list:
        xmi_path = xmi_path_or_paths[0]
    else:
        xmi_path = xmi_path_or_paths
    doc = LOTRDocument.from_xmi(xmi_path)

    mentions = doc.lotr_entity_references

    referred_entities = [reference.simple_coref.name for reference in mentions]

    if type(xmi_path_or_paths) == list:
        for xmi_path in xmi_path_or_paths[1:]:
            doc = LOTRDocument.from_xmi(xmi_path)

            mentions = doc.lotr_entity_references
            referred_entities += [reference.simple_coref.name for reference in mentions]

    return Counter(referred_entities)


def get_interaction_statistics(path_or_paths: Union[Path, List[Path]]) -> Counter:
    if type(path_or_paths) == list:
        path = path_or_paths[0]
    else:
        path = path_or_paths
    interactions_df = pd.read_csv(path)

    interactions = []

    for i, row in interactions_df.iterrows():
        ne1 = row["NE1"]
        ne2 = row["NE2"]

        char_pair = tuple(sorted((ne1, ne2)))
        interactions.append(char_pair)

    if type(path_or_paths) == list:
        for path in path_or_paths[1:]:
            interactions_df = pd.read_csv(path)
            for i, row in interactions_df.iterrows():
                ne1 = row["NE1"]
                ne2 = row["NE2"]

                char_pair = tuple(sorted((ne1, ne2)))
                interactions.append(char_pair)

    return Counter(interactions)


if __name__ == '__main__':
    xmis_paths = get_all_xmis_with_interactions()

    for by in ("window", "sentence"):
        for scale, transform in (("no_scale", lambda x: x), ("log_scale", np.log)):
            stats = []
            interaction_stats = []

            out_folder = Path(f"plots/{by}_{scale}/")
            out_folder.mkdir(exist_ok=True)
            logger.info(f"Plotting with {scale} for {by}...")
            for path in ["lotr_all", "all"] + xmis_paths:
                logger.info(path)

                if path == "lotr_all":
                    mention_statistics = get_mention_statistics([path for path in xmis_paths if "lotr" in path.name])
                    interaction_statistics = get_interaction_statistics(
                        [path.with_name("_".join(path.stem.split("_")[:-2]) + f".interactions_by_{by}") for path in
                         xmis_paths if "lotr" in path.name])
                elif path == "all":
                    mention_statistics = get_mention_statistics(xmis_paths)
                    interaction_statistics = get_interaction_statistics(
                        [path.with_name("_".join(path.stem.split("_")[:-2]) + f".interactions_by_{by}") for path in
                         xmis_paths])
                else:
                    mention_statistics = get_mention_statistics(path)
                    interaction_statistics = get_interaction_statistics(
                        path.with_name("_".join(path.stem.split("_")[:-2]) + f".interactions_by_{by}"))
                    stats.append(get_basic_statistics(path))
                print(mention_statistics)
                interaction_stats.append(interaction_statistics)

                # characters = {a for a, b in interaction_statistics.keys()}
                # characters.update({b for a, b in interaction_statistics.keys()})

                characters = [m[0] for m in mention_statistics.most_common(10)]

                char_to_int = defaultdict(lambda: len(char_to_int))

                interaction_matrix = np.zeros((len(characters), len(characters)))
                for c1 in characters:
                    for c2 in characters:
                        interaction_matrix[char_to_int[c1]][char_to_int[c2]] = transform(
                            interaction_statistics[tuple(sorted((c1, c2)))])

                cmap = matplotlib.cm.get_cmap("cividis").copy()
                if scale == "no_scale":
                    cmap.set_over("white")
                    cmap.set_under("black")
                fig, ax = plt.subplots(1, 1)
                values = sorted(set(interaction_matrix.flatten()))
                logger.info(values)
                img = ax.imshow(interaction_matrix, interpolation='none', vmax=values[-2], cmap=cmap, vmin=1)
                ax.set_xticks(list(range(len(characters))))
                ax.set_xticklabels(characters, rotation=90)
                ax.set_yticks(list(range(len(characters))))
                ax.set_yticklabels(characters)

                clb = plt.colorbar(img, cmap=cmap, extend="both")
                clb.ax.tick_params(labelsize=8)
                # clb.cmap.set_over("white")
                # clb.ax.set_title('Your Label', fontsize=8)

                plt.tight_layout()
                # plt.show()
                plt.savefig(out_folder / f"{path if type(path) == str else path.stem}.png", bbox_inches='tight',
                            dpi=300)

                print(interaction_statistics)
                print(sum(interaction_statistics.values()))

                # get_plots(path)

            stats_frame = pd.DataFrame(stats)
            stats_frame = stats_frame.set_index("Text")

            stats_frame.to_html("stats.html")
            stats_frame.to_latex("stats.tex")
            stats_frame.T.to_csv("stats.csv")
