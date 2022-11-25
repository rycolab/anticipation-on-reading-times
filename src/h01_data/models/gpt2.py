import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils.utils import clear_cache
from utils.constants import STRIDE


class GPTBaseModel():

    def __init__(self):
        self.model, self.tokenizer = self.get_model()

        self.device = self.model.device
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    @classmethod
    # estimated from wikitext-103
    def get_corpus_mean(cls):
        return cls.corpus_mean

    def get_model(self):
        clear_cache()

        model = GPT2LMHeadModel.from_pretrained(self.model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        return model, tokenizer

    def score(self, sentence, BOS=True):
        return self.get_models_output(sentence, process_func=self._get_surprisal, BOS=BOS)

    def get_entropies(self, sentence, BOS=True):
        return self.get_models_output(sentence, process_func=self._get_entropies, BOS=BOS)

    def get_renyi_entropies(self, sentence, alpha, BOS=True):
        return self.get_models_output(sentence, process_func=self._make_renyie_entropy_func(alpha), BOS=BOS)

    def get_models_output(self, sentence, process_func, BOS=True):
        with torch.no_grad():
            all_results = torch.tensor([], device=self.device)
            offset_mapping = []
            start_ind = 0

            while True:
                encodings = self.tokenizer(sentence[start_ind:], max_length=1022, truncation=True, return_offsets_mapping=True)
                if BOS:
                    tensor_input = torch.tensor([[self.bos_token_id] + encodings['input_ids'] + [self.eos_token_id]], device=self.device)
                else:
                    tensor_input = torch.tensor([encodings['input_ids'] + [self.eos_token_id]], device=self.device)
                output = self.model(tensor_input, labels=tensor_input)
                shift_logits = output['logits'][..., :-1, :].contiguous()
                shift_labels = tensor_input[..., 1:].contiguous()

                results = process_func(shift_logits, shift_labels, output)

                offset = 0 if start_ind == 0 else STRIDE - 1
                all_results = torch.cat([all_results, results[offset:-1]])
                offset_mapping.extend([(i + start_ind, j + start_ind) for i, j in encodings['offset_mapping'][offset:]])
                if encodings['offset_mapping'][-1][1] + start_ind == len(sentence):
                    break
                start_ind += encodings['offset_mapping'][-STRIDE][1]

            return np.asarray(all_results.cpu()), offset_mapping

    @staticmethod
    def _get_surprisal(logits, labels, output):
        surprisals = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
        assert torch.isclose(torch.exp(sum(surprisals) / len(surprisals)), torch.exp(output['loss']))

        return surprisals

    @staticmethod
    def _get_entropies(logits, _, __):
        probs = F.softmax(logits, dim=-1)
        surprisals = - F.log_softmax(logits, dim=-1)
        entropies = (probs * surprisals).sum(-1)

        return entropies.view(-1)

    @staticmethod
    def _get_argmin_entropies(logits, _, __):
        surprisals = - F.log_softmax(logits, dim=-1)
        entropies = surprisals.min(-1)[0]

        return entropies.view(-1)

    def _make_renyie_entropy_func(self, alpha):
        def func(logits, _, __):
            probs = F.softmax(logits, dim=-1)
            summed_probs = torch.pow(probs, alpha).sum(-1)

            entropies = torch.log(summed_probs) / (1 - alpha)
            return entropies.view(-1)

        if alpha == 1:
            return self._get_entropies
        if alpha == float('inf'):
            return self._get_argmin_entropies
        return func


class EnglishGptXl(GPTBaseModel):
    corpus_mean = 3.8845
    language = 'english'
    model_name = 'gpt2-xl'


class EnglishGptLarge(GPTBaseModel):
    corpus_mean = 3.8845
    language = 'english'
    model_name = 'gpt2-large'


class EnglishGptMedium(GPTBaseModel):
    corpus_mean = 3.8845
    language = 'english'
    model_name = 'gpt2-medium'


class EnglishGptSmall(GPTBaseModel):
    corpus_mean = 3.8845
    language = 'english'
    model_name = 'gpt2'
