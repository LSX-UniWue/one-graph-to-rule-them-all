import inspect
import sys
from collections import deque, Counter
from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm
from wuenlp import set_typesystem
from wuenlp.base.NLPStructs import BaseFeatureStructure

from wuenlp.utils.style import add_style_to_document

set_typesystem(Path("LOTRTypesystem.xml"))
from wuenlp.impl.UIMANLPStructs import UIMASentence, UIMAEntityReference, UIMAEntity, UIMADocument, UIMAAnnotation, \
    UIMAFeatureStructure, UIMASpeech, type_to_class
from wuenlp.utils.import_booknlp import doc_from_booknlp


class SimpleCorefEntity(UIMAAnnotation):
    uima_type = "de.uniwue.lotr.SimpleCorefEntity"

    @property
    def name(self) -> str:
        return self.anno["Name"]

    @name.setter
    def name(self, n: str):
        self._set_feature_value("Name", n)


class LOTREntityReference(UIMAAnnotation):
    uima_type = "de.uniwue.lotr.LOTREntityReference"

    @property
    def simple_coref(self) -> SimpleCorefEntity:
        return SimpleCorefEntity(self.anno["SimpleCoref"])

    @simple_coref.setter
    def simple_coref(self, ent: SimpleCorefEntity):
        self._set_feature_value("SimpleCoref", ent)


class LOTRDocument(UIMADocument):
    @property
    def simple_coref_entities(self) -> List[SimpleCorefEntity]:
        return [SimpleCorefEntity(anno) for anno in self.cas.get_fs_of_type(SimpleCorefEntity.uima_type)]

    @property
    def lotr_entity_references(self) -> List[LOTREntityReference]:
        return [LOTREntityReference(anno) for anno in self.cas.get_fs_of_type(LOTREntityReference.uima_type)]

    def create_simple_coref_entity(self, begin: int, end: int, name: str, add_to_document: bool) -> SimpleCorefEntity:
        sce: SimpleCorefEntity = self.create_anno(SimpleCorefEntity, begin=begin, end=end,
                                                  add_to_document=add_to_document)
        sce.name = name

        return sce

    def create_lotr_entity_reference(self, begin: int, end: int,
                                     simple_coref_target: SimpleCorefEntity,
                                     add_to_document: bool) -> LOTREntityReference:
        ler: LOTREntityReference = self.create_anno(LOTREntityReference, begin=begin, end=end,
                                                    add_to_document=add_to_document)
        ler.simple_coref = simple_coref_target

        return ler


def build_cas_from_booknlp(book, lotr_path):
    in_tokens = lotr_path.joinpath(f"{book}.tokens")
    in_entities = lotr_path.joinpath(f"{book}.entities")
    in_speeches = lotr_path.joinpath(f"{book}.quotes")
    doc = doc_from_booknlp(in_tokens, in_entities, in_speeches)
    chapter_start = 0
    prev_chapter = None
    chapter_name = doc.sentences[0].text
    for token in doc.tokens:
        if token.text == "Chapter" and token.next.text.isnumeric():
            chapter = doc.create_chapter(begin=chapter_start, end=token.previous.end, add_to_document=True)
            chapter.name = chapter_name

            sentence: UIMASentence = token.covering(UIMASentence)[0]

            chapter_name = ""
            for tok in sentence.tokens[2:]:
                if not tok.text.isalnum() and tok.text != "-":
                    break
                if tok.next and tok.next.text.islower() and tok.next.next and tok.next.next.text.islower() and tok.next.next.next and tok.next.next.next.text.islower():
                    break
                chapter_name += " " + tok.text

            logger.info(chapter_name)

            chapter_start = token.begin
    chapter = doc.create_chapter(begin=chapter_start, end=token.end, add_to_document=True)
    doc.cas.serialize(str((lotr_path / book).with_suffix(".xmi")))


def simple_coref(path: Path):
    doc = LOTRDocument.from_xmi(path)

    if doc._types_with_circles == {"de.uniwue.wuenlp.Token"}:
        doc.remove_circular_parse_trees()

    for entity in doc.simple_coref_entities:
        doc.remove_annotation(entity)

    for reference in doc.lotr_entity_references:
        doc.remove_annotation(reference)

    merges = {
        "Sam": "Sam Gamgee",
        "Tom": "Tom Bombadil",
        "the Elves": "Elves",
        "Baggins": None,
        "Mr. Baggins": None,
        "the Hobbits": "Hobbits",
        "Mr. Bilbo": "Bilbo",
        "Gandalf the Grey": "Gandalf",
        "Butterbur": "Mr. Buterbur",
        "the Black Riders": "Black Riders",
        "Sméagol": "Gollum",
        "Doom": None,
        "Peregrin": "Pippin",
        "Númenor": None,
        "Mr. Underhill": "Frodo",
        "the Lady Galadriel": "Galadriel",
        "Bilbo Baggins": "Bilbo",
        "Merry Brandybuck": "Merry",
        "Fatty": "Fatty Bolger",
        "Master Elrond": "Elrond",
        "Frodo Baggins": "Frodo",
        "Old Tom Bombadil": "Tom Bombadil",
        "201;omer": "Eomer",
        "Ugl&#250;k": "Ugluk",
        "Wormtongue": "Grima",
        "Grishn&#225;kh": "Grishnakh",
        "H&#225;ma": "Hama",
        "Mithrandir": "Gandalf",
        "Samwise": "Sam",
        "Mindolluin": None,
        "Strider": "Aragorn",
        "the Lord Denethor": "Denethor",
        "the Lady Éowyn": "Éowyn",
        "Master Meriadoc": "Merry",
        "King Théoden": "Théoden",
        "the Lord Aragorn": "Aragorn",
        "Mr. Frodo": "Frodo",
        "Mordor": "Sauron",
        "Elbereth": "Varda",
        "Th&#233;oden": "Théoden",
        "Sharkey": "Saruman",
        "the Riders": "Black Riders",
        "Gorgoroth": None,
        "Hullo": None,
        "Telperion": None,
        "the Silverlode": None,
        "the Precious": None,
        "Anfauglith": None,
        "Gurthang": None,
        "Master Peregrin": "Pippin",
        "Peregrin Took": "Pippin",
        "the Nazgûl": "Black Riders",
        "Curunír": "Saruman",
        "the orcs": "orcs",
        # original merges end here
        'Mr. Butterbur': 'Barliman Butterbur', 'Mr. Buterbur': 'Barliman Butterbur', 'Barliman': 'Barliman Butterbur',
        'Gr&#237;ma': 'Grima Wormtongue', 'Grima': 'Grima Wormtongue', 'Worm': 'Grima Wormtongue',
        'Beleg Cúthalion': 'Beleg', 'Beleg Strongbow': 'Beleg',
        'Círdan the Shipwright': 'Círdan',
        'Eorl the Young': 'Eorl',
        'Idril Celebrindal': 'Idril',
        'Mrs. Cotton': 'Rosie Cotton', 'Rose': 'Rosie Cotton', 'Rosie': 'Rosie Cotton', 'Cotton': 'Rosie Cotton',
        'Sam Gamgee': 'Sam',
        'Thorin Oakenshield': 'Thorin',
        'Thrór': 'Thror',
        'the Eldalië': 'Elves', 'the Eldar': 'Elves', "the Firstborn": 'Elves',
        'Mr.  Frodo': 'Frodo',
        'Morgoth Bauglir': 'Melkor',
        'Stinker': 'Gollum',
        'gollum': 'Gollum',
        'Master': "Frodo",
        'the Orcs': 'Orcs', 'The Orcs': 'Orcs',
        'the Ents': 'Ents', 'The Ents': 'Ents',
        'the Nine': 'Black Riders',
        'Luthien': 'Tinuviel',
        'Bill Ferny': 'Ferny',
        'Bombadil': 'Tom Bombadil',
        'Turin Turambar': 'Turin',
        # from ONE mapping
        'Lady': 'Galadriel',
    }

    entities = deque()
    entity_set = set()
    uima_entities = {}
    ref: UIMAEntityReference
    for ref in tqdm(doc.entity_references):
        ref_text = ref.text
        if ref.reference_type == "PROP" and ref.system_referred_entity.entity_type == "PER":
            if ref_text in merges:
                if merges[ref_text] is None:
                    continue
                else:
                    ref_text = merges[ref_text]
            if ref_text not in uima_entities:
                uima_entities[ref_text] = doc.create_simple_coref_entity(name=ref_text, begin=ref.begin, end=ref.end,
                                                                         add_to_document=True)

            if uima_entities[ref_text] is not None:
                doc.create_lotr_entity_reference(begin=ref.begin, end=ref.end,
                                                 simple_coref_target=uima_entities[ref_text],
                                                 add_to_document=True)
            entities.append(ref_text)
            entity_set.add(ref_text)

    entity_counter = Counter(entities)
    top_entities = entity_counter.most_common(30)

    print(entity_counter)
    print(top_entities)

    doc.serialize(path)


custom_styles = {
    LOTREntityReference: {
        "style": "STYLE_BOX",
        "background": "#ff3419",
        "foreground": "#ff3419",
        "SimpleCoref": {
            "style": "STYLE_STRING",
            "value": "function get_value(anno){return anno.features.Name;};",
            "position": "bottom-right",
            "foreground": "#ff0000",
        },
    },
    UIMASpeech: {
        "style": "STYLE_BACKGROUND",
        "background": "#568e66",
        "foreground": "#568e66",
        "Speaker": {"style": "STYLE_STRING", "label": "jsonId", "position": "left",
                    "value": "function get_value(anno){return anno.features.SystemReferredEntity.features.Name;};"},
        "SpokenTo": {"style": "STYLE_STRING", "label": "ID", "position": "right",
                     "value": "function get_value(anno){return ' to ' + anno.features.SystemReferredEntity.features.Name;};"},
    },
}


def _filter_dir(path: Path, with_interactions: bool = False):
    return [file for file in path.iterdir() if
            file.suffix == ".xmi" and ("with_interactions" in file.stem) == with_interactions]


def get_all_xmis_without_interactions() -> List[Path]:
    return sorted(_filter_dir(Path("data/lotr_out")) + _filter_dir(Path("data/hobbit_out")) + _filter_dir(
        Path("data/silmarillion_out")))


def get_all_xmis_with_interactions() -> List[Path]:
    return sorted(_filter_dir(Path("data/lotr_out"), with_interactions=True) +
                  _filter_dir(Path("data/hobbit_out"), with_interactions=True) +
                  _filter_dir(Path("data/silmarillion_out"), with_interactions=True))


def main():
    lotr_path = Path("data/lotr_out")
    books = sorted({file.name[:file.name.find(".")] for file in lotr_path.iterdir() if
                    not file.name.startswith(".") and "with_interactions" not in file.stem})

    logger.info(f"Found books: {books}")

    for book in books:
        build_cas_from_booknlp(book, lotr_path)

        lotr_xmi = (lotr_path / book).with_suffix(".xmi")
        simple_coref(lotr_xmi)

        add_style_to_document(lotr_xmi, lotr_xmi, LOTRDocument,
                              custom_styles)


def simple_coref_all_files():
    for xmi in get_all_xmis_without_interactions():
        logger.info(f"Processing {xmi}...")
        simple_coref(xmi)


uima_classes = inspect.getmembers(sys.modules[__name__],
                                  lambda cls: inspect.isclass(cls) and not inspect.isabstract(
                                      cls) and issubclass(
                                      cls, BaseFeatureStructure))

type_to_class.update({_class[1].uima_type: _class[1] for _class in uima_classes})

if __name__ == '__main__':
    build_cas_from_booknlp("silmarillion", Path("data/silmarillion_out"))  # do this for all files
    simple_coref_all_files()
