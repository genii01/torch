import transformers
from transformers import RobertaForSequenceClassification


class ROBERTA_KLUE_Classifier:
    def __init__(self, config):
        self.classifier = RobertaForSequenceClassification.from_pretrained(
            "klue/roberta-large", num_labels=config.num_classes
        )


class ROBERTA_S_Multi_Classifier:
    def __init__(self, config):
        self.classifier = RobertaForSequenceClassification.from_pretrained(
            "jhgan/ko-sroberta-multitask", num_labels=config.num_classes
        )
