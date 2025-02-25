'''
Evaluation module for PLDR-LLM.
This module used vllm_causalllms.py as starting point.
'''


from typing import List, Tuple, Optional, Literal, Union

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM as LM
import copy
from tqdm import tqdm
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
)

import torch
import torch.nn.functional as F
from functools import partial

eval_logger = eval_logger

def create_masks(inp, device):
    '''
    inp: tensor of shape [batch_size, seq_len]
    Create masks for decoder layer for pldr model.
    Used in the attention block in the decoder.
    It is used to pad and mask future tokens in the input received by the decoder.
    '''
    look_ahead_mask = create_look_ahead_mask(inp.size()[1], device)
    dec_target_padding_mask = create_padding_mask(inp, device)
    combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask

def create_padding_mask(seq, device):
    '''
    inp: tensor of shape [batch_size, seq_len]
    Create a mask for padding in the input for decoder.
    '''
    seq = torch.eq(seq, 0)
    seq=seq.to(device=device, dtype=torch.float32)

    return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size, device):
    '''
    The values that remain as 1 are multiplied with a small number
    so these entries vanish in attention calculation.
    '''
    mask = 1 - torch.tril(torch.ones((size, size)))
    return mask.to(device=device)

def top_k_logits(logits, k):
    '''Top-k sampling'''
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = torch.topk(logits, k=k, sorted=True)
        min_values = values[:, -1, None]
        return torch.where(
                        logits < min_values,
                        torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                        logits,
                        )
    return logits if k==0 else _top_k()

def top_p_logits(logits, p, device):
    """Nucleus sampling"""
    batch, _ = logits.size()
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    indices = torch.stack([
                        torch.arange(0, batch).to(device),
                        # number of indices to include
                        torch.maximum(torch.sum((cumulative_probs <= p).to(torch.int32), dim=-1) - 1, torch.tensor(0))
                        ], dim=-1).to(torch.int32).tolist()
    min_values = torch.tensor([sorted_logits[i[0],i[1]] for i in indices]).to(device)
    return torch.where(
        logits < min_values,
        torch.ones_like(logits) * -1e10,
        logits,
    )

@register_model("pldrllm")
class pldrllm(LM):
    _DEFAULT_MAX_LENGTH = 1024

    def __init__(self,
            model,
            tokenizer,
            batch_size: Union[str, int]=1,
            max_length:int=None,
            max_gen_toks:int=256,
            temperature:float=1.0,
            top_p:float=1.0,
            top_k:int=0,
            enable_kvcache:bool=True,
            enable_Gcache:bool=True,
            Gcachelst_init:list=None,
            model_type="pldrllm", #or "pldrllm_with_g",
            device:str='cuda:0'
    ):
        super().__init__()

        self.model=model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks
        self.temperature=temperature
        self.top_p=top_p
        self.top_k=top_k
        self.enable_kvcache=enable_kvcache,
        self.enable_Gcache=enable_Gcache,
        self.model_type=model_type
        self.Gcachelst_init=Gcachelst_init
        if device is None or device.startswith('cuda'):
            self.device=torch.device(device)
            torch.cuda.set_device(self.device)
        elif device == 'cpu':
            self.device=torch.device('cpu')
        else:
            raise ValueError("Device not recognized. Choose cpu or cuda.")
        self.tokenize=partial(self.tokenizer.encode, add_bos=False, add_eos=True, 
                             trim_leading_whitespace=False, prefix=None)
        self.detokenize=self.tokenizer.decode

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenize('')[0]

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def tok_encode(
        self,
        string: Union[tuple[str], list[str], str],
        left_truncate_len=None,
        add_special_tokens=False
    ):

        """This version can encode list of strings or a single string. """
        if isinstance(string, str):
            encoding=[self.tokenize(string)]
        elif isinstance(string, list) or isinstance(string, tuple):
            encoding=[self.tokenize(s) for s in string]
        else:
            raise ValueError("Not a string, list of strings or tuple of strings.")

        if not add_special_tokens:
            encoding=[encoding[i][:-1] for i in range(len(encoding))]

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = [encoding[i][-left_truncate_len:] for i in range(len(encoding))]

        return encoding


    def _model_generate(
        self,
        requests: List[List[int]] = None,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        temperature=1.0, 
        top_k=0, 
        top_p=1.0,
        get_logprobs=False 
    ):

        outputs=[]
        for req in requests:
            output= self.generate_text(sentence=req, 
                                  until=stop, 
                                  temperature=temperature, 
                                  top_k=top_k, 
                                  top_p=top_p, 
                                  max_length=max_tokens,
                                  get_logprobs=get_logprobs
                                  )
            outputs.append(output)
        return outputs 
    
    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation, add_special_tokens=False)[0]
        context_enc = self.tok_encode(context, add_special_tokens=False)[0]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(continuation)[0]
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests]):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string)[0],
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]]= self.tok_encode(context, add_special_tokens=False)
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            if not until:
                until = [[self.detokenize([self.eot_token_id])]]
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            if "temperature" in kwargs.keys():
                temperature = kwargs.pop("temperature")
            else:
                temperature = self.temperature
            
            if "top_k" in kwargs.keys():
                top_k = kwargs.pop("top_k")
            else:
                top_k = self.top_k

            if "top_p" in kwargs.keys():
                top_p = kwargs.pop("top_p")
            else:
                top_p = self.top_p

            # set the max length in tokens of inputs ("context_enc")
            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            cntxtncont= self._model_generate(
                requests=context_encoding,
                max_tokens=max_gen_toks,
                stop=until,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                get_logprobs=False
            )
            cont=[x[len(y):] for x, y in zip(cntxtncont, context_encoding)]

            # cache generations
            for output, context in zip(cont, context):
                generated_text = self.detokenize(output)
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()

        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(total=len(requests), disable=disable_tqdm)
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, get_logprobs=True) 

            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Tokens from context+continuations
        :param outputs: RequestOutput
            Contains prompt
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        continuation_logprobs_lists = outputs

        continuation_logprobs = sum(
            logprob_list[token]
            for token, logprob_list in zip(
                tokens[ctxlen:], continuation_logprobs_lists[ctxlen:]
            ))

        is_greedy=True
        for token, logprob_list in zip(
            tokens[ctxlen:], continuation_logprobs_lists[ctxlen:]
        ):
            # Get the token with the maximum log probability from the logprob_list
            if logprob_list:
                top_token = logprob_list.index(max(logprob_list))
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy
    

    def generate_text(self, sentence:list[int],
                          until:Union[list[str], str], 
                          temperature=1.0, 
                          top_k=0, 
                          top_p=1.0,
                          get_logprobs=False, 
                          max_length=None):
        '''
        Generate tokens from context or log probabilites of tokens.
        Args:
            sentence: source sentence as input.
            temperature: parameter to determine how deterministic the output is between (0,1]. Less deterministic on logits if temperature=1
            top_k: value to select from top k largest logits, select all if k==0
            top_p: cumulative probability threshold to select from logits for nucleus sampling. Select all if p == 1
            get_logprobs: If True, get log probs as output.
            max_length: maximum number of iterations to run.
        Returns:
            output_lst:list[int] token ids of context+continuation if get_logprobs is False
            all_logprob_lst: list[dict(vocab_indx:logprob)] for each token in sentence if get_logprobs is True.
        '''
        assert 0.0 <= temperature <=1.0, "set temperature between [0, 1]"
        assert 0.0 < top_p <=1.0, "set nucleus sampling probability between (0, 1], p=1 to skip top_p sampling"
        assert top_k >= 0, "set top_k above 0 or 0 to skip top_k sampling"

        max_length=max_length if max_length is not None else 100

        pldr_input = torch.tensor(sentence).to(self.device)
        if isinstance(until, str):
            end = self.tok_encode([until], add_special_tokens=False) 
        elif isinstance(until, list):
            end = self.tok_encode(until, add_special_tokens=False) 

        output=pldr_input
        output=output[None,:]
        seq_in=output
        kvcachelst=None
        Gcachelst=self.Gcachelst_init if self.Gcachelst_init is not None else None
        if Gcachelst is not None:
            Gcachelst=[[AW.to(self.device), avAp.to(self.device)] for AW, avAp in Gcachelst]
        cached=False

        if not get_logprobs:
            for _ in range(max_length):
                if not cached:
                    combined_mask = create_masks(seq_in, device=self.device)
                else:
                    combined_mask=None

                self.model.eval()
                if self.model_type=="pldrllm":
                    predictions, _, att_weights, kvcachelst_nxt = self.model([seq_in, combined_mask],
                                                                kvcachelst=kvcachelst,
                                                                Gcachelst=Gcachelst)
                    if self.enable_Gcache:
                        Gcachelst=[[t[0],t[4]] for t in att_weights]
                elif self.model_type=="pldrllm_with_g":
                    predictions, _, kvcachelst_nxt = self.model([seq_in, combined_mask],
                                                                kvcachelst=kvcachelst)
                else:
                    raise ValueError("Specify correct model_type: pldrllm or pldrllm_with_g")

                if self.enable_kvcache:
                    kvcachelst=kvcachelst_nxt
                    cached=True     
                
                predictions = predictions[:, -1, :]
                
                #scale logits for temperature sampling
                if 0 < temperature <= 1:
                    #temperature, top_k and nucleus sampling are stackable
                    if temperature < 1:
                        predictions = predictions/temperature

                    #top_p sampling
                    if top_p < 1:
                        predictions=top_p_logits(logits=predictions, p=top_p)

                    #top_k sampling
                    if top_k > 0:
                        predictions=top_k_logits(logits=predictions, k=top_k)
                elif temperature == 0: #condition for greedy search
                    predictions=top_k_logits(logits=predictions, k=1)
                else:
                    raise ValueError("Temperature needs to be 0 for greedy sampling or (0,1] for temperature/top_k/top_p sampling")

                predictions=torch.distributions.categorical.Categorical(logits=predictions) #(batch_size, vocab_size)
                predicted_id=predictions.sample()
                predicted_id=predicted_id[None,:]

                if cached:
                    seq_in=predicted_id # (batch_size, 1)
                else:
                    seq_in=torch.concat([seq_in, predicted_id], axis=-1)
                    
                output = torch.concat([output, predicted_id], axis=-1)

                output_lst=output[0].tolist()
                cont_lst=output_lst[len(sentence):]
                
                for end_token in end:
                    end_tok_len=len(end_token)
                    if len(cont_lst) >= end_tok_len:
                        if cont_lst[-end_tok_len:]==end_token:
                            break

            return output_lst        
        else:
            all_logprob_lst=[None]
            inp=output[:, :-1] 
            combined_mask = create_masks(inp, device=self.device)
            self.model.eval()
            if self.model_type=="pldrllm":
                predictions, _, _, _ = self.model([inp, combined_mask],
                                                kvcachelst=kvcachelst,
                                                Gcachelst=Gcachelst)
            elif self.model_type=="pldrllm_with_g":
                predictions, _, _ = self.model([inp, combined_mask],
                                                kvcachelst=kvcachelst)
            else:
                raise ValueError("Specify correct model_type: pldrllm or pldrllm_with_g")

            predictions=F.log_softmax(predictions, dim=-1)
            predictions=predictions.tolist()

            return all_logprob_lst+predictions[0]
