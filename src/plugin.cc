#include <algorithm>
#include <rime/config.h>
#include <rime/resource.h>
#include <rime/service.h>
#include <utf8.h>

#include "plugin.h"

extern "C" {
#include "llama.h"
}

// #define TIME
#ifdef TIME
#include <chrono>
#endif

namespace rime {

static float dot(std::vector<float> a, std::vector<float> b);

Llama::Llama(Config* config, LlamaComponent* component)
  : component_(component)
{}

Llama::~Llama() {}

double Llama::Query(const string& context, const string& word, bool is_rear) {
  LOG(INFO) << "llama: query grammar '" << context << "' with word '" << word << "'";
  component_->Init();

#ifdef TIME
  using namespace std::chrono;
  auto start = high_resolution_clock::now();
#endif

  // Context embedding
  std::vector<float> context_emb;
  if (context == last_context_) {
    context_emb = last_context_emb_;
  } else {
    last_context_ = context;
    context_emb = last_context_emb_ = component_->GetEmbedding(context);
  }

  // Word embedding
  std::vector<float> word_emb = component_->GetEmbedding(word);

  // Returns topical similarity
  float similarity = dot(context_emb, word_emb);
  LOG(INFO) << "similarity between '" << context << "' and '" << word << "' is "<< similarity;

#ifdef TIME
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start);
  LOG(INFO) << "llama: query completed in " << duration.count() << " ms";
#endif
  return similarity;
}

LlamaComponent::LlamaComponent() {
  Init();
}

LlamaComponent::~LlamaComponent() {
  if (ctx_) llama_free(ctx_);
  if (model_) llama_model_free(model_);
  llama_backend_free();
}

Llama* LlamaComponent::Create(Config* config) {
  LOG(INFO) << "llama: creating new Llama object";
  return new Llama(config, this);
}

void LlamaComponent::Init() {
  if (inited_)
    return;
  else
    inited_ = true;

  LOG(INFO) << "llama: trying to initialize llama.cpp";
  llama_backend_init();

  llama_model_params model_params = llama_model_default_params();
  model_ = llama_model_load_from_file("/tmp/model.gguf", model_params);
  if (!model_) {
    LOG(ERROR) << "llama: Failed to load the model file\n";
    init_failed_ = true;
    return;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 1024;
  ctx_params.embeddings = true;
  ctx_params.n_threads = std::thread::hardware_concurrency();
  ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) {
    LOG(ERROR) << "llama: Failed to create context";
    init_failed_ = true;
    return;
  }

  dim_ = llama_model_n_embd(model_);
  LOG(INFO) << "llama: model loaded successfully, dim=" << dim_;
  init_failed_ = false;
}

std::vector<float> LlamaComponent::GetEmbedding(const string& s) {
  Init();
  if (init_failed_)
    return {};

  auto it = cache_.find(s);


  // Tokenization
  std::vector<llama_token> tokens(512);
  const llama_vocab *vocab = llama_model_get_vocab(model_);
  int n_tokens = llama_tokenize(vocab,
                                s.c_str(),
                                s.length(),
                                tokens.data(),
                                tokens.size(),
                                true,
                                false);
  if (n_tokens <= 0)
    return {};
  tokens.resize(n_tokens);

  // Batch
  llama_batch batch = llama_batch_init(n_tokens, 0, 1);
  for (int i = 0; i < n_tokens; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = false;
  }
  batch.n_tokens = n_tokens;
  batch.logits[batch.n_tokens - 1] = true;

  // Encode
  if (llama_encode(ctx_, batch) != 0) {
    llama_batch_free(batch);
    LOG(ERROR) << "llama: Failed to encode '" << s << "'";
    return {};
  }

  // Get embeddings
  const float *emb = llama_get_embeddings_seq(ctx_, 0);
  if (!emb) {
    LOG(ERROR) << "llama: Failed to get embeddings of '" << s << "'";
    return {};
  }
  std::vector<float> result(emb, emb + dim_);
  llama_batch_free(batch);
  if (cache_.size() < 10000) {
      cache_[s] = result;
  } else {
      cache_.clear();
  }
  return result;
}

static float dot(std::vector<float> a, std::vector<float> b) {
  if (a.size() != b.size())
    return 0.0f;

  size_t n = a.size();
  const float* __restrict__ pa = a.data();
  const float* __restrict__ pb = b.data();

  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    dot += pa[i] * pb[i];
    norm_a += pa[i] * pa[i];
    norm_b += pb[i] * pb[i];
  }
  if (norm_a == 0.0f || norm_b == 0.0f)
    return 0.0f;
  return sqrtf(dot * dot / (norm_a * norm_b));
}

}
