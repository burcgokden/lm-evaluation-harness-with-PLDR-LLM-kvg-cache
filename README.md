# Language Model Evaluation Harness Suite with PLDR-LLM with KV-cache and G-cache support

This repository is a fork of the LM Evaluation Harness Suite with PLDR-LLM model support pinned at version 0.4.3. This version was used to evaluate PLDR-LLM models on benchmark datasets for the research paper: [PLDR-LLMs Learn A Generalizable Tensor Operator That Can Replace Its Own Deep Neural Net At Inference](https://arxiv.org/abs/2502.13502).

# How to evaluate PLDR-LLMs on LM  Evaluation Harness Suite

- Clone this repository.
- Main branch has the PLDR-LLM support with LM Eval Harness Suite version pinned at 0.4.3.
- Install lm_eval module as described at [LM Evaluation Harness repository](https://github.com/EleutherAI/lm-evaluation-harness/tree/main#install):
    ```sh
    cd lm-eval-harness-with-PLDR-LLM
    pip install -e .
    ```

- Add path of src/ and desired model folder (eg. pldr_model_v510/) to sys.path from [PLDR-LLM with KV-cache and G-cache github repository](https://github.com/burcgokden/PLDR-LLM-with-KVG-cache).

    ```python
    import os
    import sys

    src_path=os.path.abspath("./PLDR-LLM-with-KVG-cache/src")
    pldr_v510_path=os.path.abspath("./PLDR-LLM-with-KVG-cache/src/pldr_model_v510")

    sys.path.insert(0, src_path)
    sys.path.insert(0, pldr_v510_path)
    ```

- Pretrained PLDR-LLM models as pytorch saved model files and tokenizers used in the research paper can be found at [https://huggingface.co/fromthesky](https://huggingface.co/fromthesky) . 
- Unzip the tokenizer model files. The extracted files are sentencepiece model (.model) and vocabulary (.vocab) files.
- Use the hyperparameter dictionary, .pth model file and the tokenizer model to load the model as described in the PLDR-LLM with KV-cache and G-cahe repository under the section for [model evaluation](https://github.com/burcgokden/PLDR-LLM-with-KVG-cache/tree/main#pldr-llm-model-evaluation). Loading the model and tokenizer will initialize e2e_obj and inp_obj objects.

*Note:* The model inference is fast, however the support module for LM evaluation harness for PLDR-LLM is not optimized, evaluations of benchmarks take longer than implementations for huggingface models, for example.

- For pldr_model_v510/pldr_model_v510_dag versions model_type is 'pldrllm'; for pldr_model_v510G/pldr_model_v510Gi version model_type is 'pldrllm_with_g' :

    ```python
    import lm_eval

    model2eval=lm_eval.models.pldrllm.pldrllm(model=e2e_obj.pldr_model, 
                                            tokenizer=inp_obj.tokenizer,
                                            batch_size=8, 
                                            max_length=1024, 
                                            max_gen_toks=256,
                                            temperature=1.0, top_k=0, top_p=1.0,
                                            enable_kvcache=True,
                                            enable_Gcache=True,
                                            Gcachelst_init=None,
                                            model_type="pldrllm",
                                            device='cuda:0'
                                            )

    task_manager = lm_eval.tasks.TaskManager()
    evb_results=lm_eval.simple_evaluate(model=model2eval, tasks=["pldrllm_zeroshot"], 
                                        task_manager=task_manager, limit=None)

    #print results
    print("SHOWING RESULTS:")
    print(eval_results["results"])
    ```


