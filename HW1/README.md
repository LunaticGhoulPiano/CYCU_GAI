# Description
![image](https://github.com/LunaticGhoulPiano/CYCU_GAI/HW1/blob/master/Description_PPT.jpg?raw=true)

# Preprations
- Apply a Groq api key
- Make a file ".env" and set:
```
GROQ_API_KEY=<YOUR_GROQ_API_KEY>
```

# Commands
- Install libraries
```
python -m pip install -r requirements.txt
```
- Run LLM
```
python main.py
```

# About tokenizer
- Lunary AI
    - Originally this homework should use OpenAI's api, but I don't want to pay any money. Hence I use the free api **Groq**, and this cause another problem: I can't use the simply-install-and-use tokenizer ```tiktoken```, which is for OpenAI's model only.
    - I have tried using the corresponding tokenizer, but I don't know how to deal with those problems, for example:
        - Groq provide ```llama-3.3-70b-versatile``` -> ```meta-llama/Meta-Llama-3-8B``` on [huggingface](https://huggingface.co/docs/transformers/model_doc/llama3)
        - I run the official code:
        ```python
        import transformers
        import torch

        model_id = "meta-llama/Meta-Llama-3-8B"

        pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        pipeline("Hey how are you doing today?")
        ```
        - Error messages:
        ```
        Traceback (most recent call last):
        ...

        Cannot access gated repo for url https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/config.json.
        Access to model meta-llama/Meta-Llama-3-8B is restricted and you are not in the authorized list. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B to ask for access.

        The above exception was the direct cause of the following exception:

        Traceback (most recent call last):
        ...
            raise EnvironmentError(
        OSError: You are trying to access a gated repo.
        Make sure to have access to it at https://huggingface.co/meta-llama/Meta-Llama-3-8B.
        403 Client Error. (Request ID: Root=1-67f01c2...)

        Cannot access gated repo for url https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/config.json.
        Access to model meta-llama/Meta-Llama-3-8B is restricted and you are not in the authorized list. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B to ask for access.
        ```
    - And after searching, I found [Llama 3 Tokenizer from Lunary AI](https://lunary.ai/llama3-tokenizer):
    ![image](https://github.com/LunaticGhoulPiano/CYCU_GAI/HW1/blob/master/Lunary_Llama3_Tokenizer.jpg?raw=true)
    - So I use post request to send and get the result that this webpage displaied. According to this webpage, it support ```Llama 3.1 70B```, ```Llama 3 70B```, and ```Llama 3.1 8B```, so I choose the coressponding model ```llama-3.1-8b-instant``` which Groq supports.