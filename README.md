# librime-llama

librime-llama is an experimental Rime plugin which tries to takes advantage of pre-trained large language models to offer better candidate sorting.

⚠️ This is published only for the purpose of demonstration. Do not use unless you know what you are doing.

## How does it work?

librime-llama works in a similar way to librime-octagram, which is an n-gram (or "any-gram") implementation.

Rime's grammar support provides a way to "contextually" weight each candidate, with the context provided as an argument. librime-octagram, for example, queries a gram DB and returns such contextual weights. librime-llama just finds them by querying llama.cpp.

As the first step, librime-llama computes the sentence embedding as a digest of the context. The idea is to summarize the "topic" of the context.

Then, for each candidate, librime-llama computes the word embedding, and then computes a "topical" similarity by computing the cosine similarity between the word embedding and the sentence digest. The similarity will be added into the dynamic weight, which Rime's sentence-making algorithm takes into account.

## How good is it?

It does not work as well as expected. The 'grammar' support of Rime tries to enumerate combinations and compute how strong two words are related. This works well for n-gram, but (1) causes severe delays and (2) messes up the candidate sorting quite a bit.

Theoretically, librime-llama works best when the context window is very long (so that we know the "topic" of the session), and then determines which next words align with the topic. But this is not the case for Rime today.

## How fast is it?

(Using the recommended model below.) Each call to `Llama::Query` takes 3 ~ 6 ms. If you use an even smaller model, it can take as little as 1 ms.

The performance is still not great because Rime tries to enumerate too many candidates, and the total latencies add up quickly. Also, Rime only provides a very short context window which unfortunately limits the capability of a transformer.

Tip: Since getting embeddings is a small task, please disable GPU and use pure CPU computation. For example, on macOS you would use:

```
-DLLAMA_METAL=OFF -DLLAMA_NATIVE=ON -DLLAMA_ACCELERATE=ON -DGGML_METAL=OFF
```

(This is already hardcoded in CMakeLists.txt.)

## How to use?

First, move `octagram` to somewhere not found by Rime.

Then, get the model file and save it as `/tmp/model.gguf`:

- [paraphrase-multilingual-MiniLM-L12-118M-v2-Q8_0.gguf](https://huggingface.co/mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf/tree/main)

Finally, compile librime-llama and puts the library under the plugins directory.
