//
// Copyright RIME Developers
// Distributed under GPLv3
//

#pragma once

#include <rime/common.h>
#include <rime/component.h>
#include <rime/resource.h>
#include <rime/gear/grammar.h>

class llama_model;
class llama_context;

namespace rime {

class LlamaComponent;

class Llama : public Grammar {
 public:
  Llama(Config* config, LlamaComponent* component);
  virtual ~Llama();
  double Query(const string& context,
               const string& word,
               bool is_rear) override;
  
 private:
  LlamaComponent* component_;
  string last_context_;
  std::vector<float> last_context_emb_;
};

class LlamaComponent : public Llama::Component {
 public:
  LlamaComponent();
  virtual ~LlamaComponent();

  Llama* Create(Config* config) override;

  void Init();
  std::vector<float> GetEmbedding(const string& s);

 private:
  bool inited_ = false;
  bool init_failed_ = false;
  llama_model *model_ = nullptr;
  llama_context *ctx_ = nullptr;
  int dim_ = 0;
  unordered_map<string, vector<float>> cache_;
};

}
