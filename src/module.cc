//
// Copyright RIME Developers
// Distributed under GPLv3
//

#include <rime/component.h>
#include <rime/registry.h>
#include <rime/setup.h>  // for rime::LoadModules in RIME_REGISTER_MODULE_GROUP
#include <rime_api.h>

#include "plugin.h"

using namespace rime;

static void rime_grammar_initialize() {
  LOG(INFO) << "registering components from module 'llama'.";
  Registry& r = Registry::instance();
  r.Register("grammar", new LlamaComponent);
}

static void rime_grammar_finalize() {}

RIME_REGISTER_MODULE(grammar)

RIME_REGISTER_MODULE_GROUP(llama, "grammar")
