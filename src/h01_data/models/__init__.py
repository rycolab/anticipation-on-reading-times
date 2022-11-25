# pylint: disable=global-statement,invalid-name

from .gpt2 import \
    EnglishGptSmall, EnglishGptMedium, EnglishGptLarge, EnglishGptXl


# global model
model = None

MODELS = {
    "gpt-small": EnglishGptSmall,
    "gpt-medium": EnglishGptMedium,
    "gpt-large": EnglishGptLarge,
    "gpt-xl": EnglishGptXl,
}


def get_corpus_mean(model_name):
    model_cls = MODELS[model_name]
    return model_cls.get_corpus_mean()


def score(sentence, model_name):
    global model

    model_cls = MODELS[model_name]

    if not isinstance(model, model_cls):
        model = model_cls()
    score_value = model.score(sentence)

    return score_value


def get_entropies(sentence, model_name):
    global model

    model_cls = MODELS[model_name]

    if not isinstance(model, model_cls):
        model = model_cls()
    score_value = model.get_entropies(sentence)

    return score_value


def make_renyi_entropies_func(alpha):
    def get_renyi_entropies(sentence, model_name):
        global model

        model_cls = MODELS[model_name]

        if not isinstance(model, model_cls):
            model = model_cls()
        score_value = model.get_renyi_entropies(sentence, alpha)

        return score_value

    return get_renyi_entropies
