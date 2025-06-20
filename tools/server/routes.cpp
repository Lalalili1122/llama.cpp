#include <stdio.h>

#include "server.hpp"

using Handler = std::function<void(const httplib::Request &, httplib::Response &)>;

// Define handlers
static void handle_health(const httplib::Request &, httplib::Response & res) {
    // error and loading states are handled by middleware
    json health = {
        { "status", "ok" }
    };
    res_ok(res, health);
}

static void handle_metrics(const httplib::Request &, httplib::Response & res, server_context & ctx_server,
                           common_params & params) {
    if (!params.endpoint_metrics) {
        res_error(res, format_error_response("This server does not support metrics "
                                             "endpoint. Start it with `--metrics`",
                                             ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    // request slots data using task queue
    int task_id = ctx_server.queue_tasks.get_new_id();
    {
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.id = task_id;
        ctx_server.queue_results.add_waiting_task_id(task_id);
        ctx_server.queue_tasks.post(std::move(task), true);  // high-priority task
    }

    // get the result
    server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
    ctx_server.queue_results.remove_waiting_task_id(task_id);

    if (result->is_error()) {
        res_error(res, result->to_json());
        return;
    }

    // TODO: get rid of this dynamic_cast
    auto res_metrics = dynamic_cast<server_task_result_metrics *>(result.get());
    GGML_ASSERT(res_metrics != nullptr);

    // metrics definition:
    // https://prometheus.io/docs/practices/naming/#metric-names
    json all_metrics_def = json{
        { "counter",
         { { { "name", "prompt_tokens_total" },
              { "help", "Number of prompt tokens processed." },
              { "value", (uint64_t) res_metrics->n_prompt_tokens_processed_total } },
            { { "name", "prompt_seconds_total" },
              { "help", "Prompt process time" },
              { "value", (uint64_t) res_metrics->t_prompt_processing_total / 1.e3 } },
            { { "name", "tokens_predicted_total" },
              { "help", "Number of generation tokens processed." },
              { "value", (uint64_t) res_metrics->n_tokens_predicted_total } },
            { { "name", "tokens_predicted_seconds_total" },
              { "help", "Predict process time" },
              { "value", (uint64_t) res_metrics->t_tokens_generation_total / 1.e3 } },
            { { "name", "n_decode_total" },
              { "help", "Total number of llama_decode() calls" },
              { "value", res_metrics->n_decode_total } },
            { { "name", "n_busy_slots_per_decode" },
              { "help", "Average number of busy slots per llama_decode() call" },
              { "value",
                (float) res_metrics->n_busy_slots_total / std::max((float) res_metrics->n_decode_total, 1.f) } } } },
        { "gauge",
         { { { "name", "prompt_tokens_seconds" },
              { "help", "Average prompt throughput in tokens/s." },
              { "value", res_metrics->n_prompt_tokens_processed ?
                             1.e3 / res_metrics->t_prompt_processing * res_metrics->n_prompt_tokens_processed :
                             0. } },
            { { "name", "predicted_tokens_seconds" },
              { "help", "Average generation throughput in tokens/s." },
              { "value", res_metrics->n_tokens_predicted ?
                             1.e3 / res_metrics->t_tokens_generation * res_metrics->n_tokens_predicted :
                             0. } },
            { { "name", "requests_processing" },
              { "help", "Number of requests processing." },
              { "value", (uint64_t) res_metrics->n_processing_slots } },
            { { "name", "requests_deferred" },
              { "help", "Number of requests deferred." },
              { "value", (uint64_t) res_metrics->n_tasks_deferred } } }                                            }
    };

    std::stringstream prometheus;

    for (const auto & el : all_metrics_def.items()) {
        const auto & type        = el.key();
        const auto & metrics_def = el.value();

        for (const auto & metric_def : metrics_def) {
            const std::string name = metric_def.at("name");
            const std::string help = metric_def.at("help");

            auto value = json_value(metric_def, "value", 0.);
            prometheus << "# HELP llamacpp:" << name << " " << help << "\n"
                       << "# TYPE llamacpp:" << name << " " << type << "\n"
                       << "llamacpp:" << name << " " << value << "\n";
        }
    }

    res.set_header("Process-Start-Time-Unix", std::to_string(res_metrics->t_start));

    res.set_content(prometheus.str(), "text/plain; version=0.0.4");
    res.status = 200;  // HTTP OK
}

static void handle_props(const httplib::Request &, httplib::Response & res, server_context & ctx_server) {
    // this endpoint is publicly available, please only return what is safe to
    // be exposed
    json data = {
        { "default_generation_settings", ctx_server.default_generation_settings_for_props },
        { "total_slots", ctx_server.params_base.n_parallel },
        { "model_path", ctx_server.params_base.model.path },
        { "modalities",
         json{
              { "vision", ctx_server.oai_parser_opt.allow_image },
              { "audio", ctx_server.oai_parser_opt.allow_audio },
          } },
        { "chat_template", common_chat_templates_source(ctx_server.chat_templates.get()) },
        { "bos_token", common_token_to_piece(ctx_server.ctx, llama_vocab_bos(ctx_server.vocab),
         /* special= */ true) },
        { "eos_token", common_token_to_piece(ctx_server.ctx, llama_vocab_eos(ctx_server.vocab),
         /* special= */ true) },
        { "build_info", build_info },
    };
    if (ctx_server.params_base.use_jinja) {
        if (auto tool_use_src = common_chat_templates_source(ctx_server.chat_templates.get(), "tool_use")) {
            data["chat_template_tool_use"] = tool_use_src;
        }
    }

    res_ok(res, data);
}

static void handle_props_change(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    if (!ctx_server.params_base.endpoint_props) {
        res_error(res, format_error_response("This server does not support changing global "
                                             "properties. Start it with `--props`",
                                             ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    json data = json::parse(req.body);

    // update any props here
    res_ok(res, {
                    { "success", true }
    });
}

static void handle_api_show(const httplib::Request &, httplib::Response & res, server_context & ctx_server) {
    json data = {
        {
         "template", common_chat_templates_source(ctx_server.chat_templates.get()),
         },
        { "model_info",
         {
              {
                  "llama.context_length",
                  ctx_server.slots.back().n_ctx,
              },
          } },
        { "modelfile", "" },
        { "parameters", "" },
        { "template", common_chat_templates_source(ctx_server.chat_templates.get()) },
        { "details",
         { { "parent_model", "" },
            { "format", "gguf" },
            { "family", "" },
            { "families", { "" } },
            { "parameter_size", "" },
            { "quantization_level", "" } } },
        { "model_info", "" },
        { "capabilities", { "completion" } }
    };

    res_ok(res, data);
}

static void handle_models(const httplib::Request &, httplib::Response & res, common_params & params,
                          server_context & ctx_server, std::atomic<server_state> & state) {
    server_state current_state = state.load();
    json         model_meta    = nullptr;
    if (current_state == SERVER_STATE_READY) {
        model_meta = ctx_server.model_meta();
    }

    json models = {
        { "models",
         { { { "name", params.model_alias.empty() ? params.model.path : params.model_alias },
              { "model", params.model_alias.empty() ? params.model.path : params.model_alias },
              { "modified_at", "" },
              { "size", "" },
              { "digest", "" },  // dummy value, llama.cpp does not support managing
                                 // model file's hash
              { "type", "model" },
              { "description", "" },
              { "tags", { "" } },
              { "capabilities", { "completion" } },
              { "parameters", "" },
              { "details",
                { { "parent_model", "" },
                  { "format", "gguf" },
                  { "family", "" },
                  { "families", { "" } },
                  { "parameter_size", "" },
                  { "quantization_level", "" } } } } } },
        { "object", "list"                             },
        { "data",
         {
              {
                  { "id", params.model_alias.empty() ? params.model.path : params.model_alias },
                  { "object", "model" },
                  { "created", std::time(0) },
                  { "owned_by", "llamacpp" },
                  { "meta", model_meta },
              },
          }                                            }
    };

    res_ok(res, models);
}

static void handle_completions_impl(server_task_type type, json & data, const std::vector<raw_buffer> & files,
                                    const std::function<bool()> & is_connection_closed, httplib::Response & res,
                                    oaicompat_type oaicompat, server_context & ctx_server) {
    printf("impl: entry\n");
    printf("ctx_server address: %p\n", (void *) &ctx_server);
    printf("ctx_server.ctx: %p\n", (void *) ctx_server.ctx);
    printf("ctx_server.vocab: %p\n", (void *) ctx_server.vocab);
    fflush(stdout);

    if (!ctx_server.ctx) {
        res_error(res, format_error_response("Model not loaded", ERROR_TYPE_SERVER));
        return;
    }
    if (!ctx_server.vocab) {
        res_error(res, format_error_response("Model not loaded", ERROR_TYPE_SERVER));
        return;
    }

    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    if (ctx_server.params_base.embedding) {
        res_error(res, format_error_response("This server does not support completions. Start it "
                                             "without `--embeddings`",
                                             ERROR_TYPE_NOT_SUPPORTED));
        return;
    }
    printf("impl: got data\n");
    fflush(stdout);

    auto completion_id = gen_chatcmplid();
    SRV_DBG("handle_completions_impl: got completion_id %s", completion_id.c_str());
    std::unordered_set<int> task_ids;
    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        SRV_DBG("handle_completions_impl: got prompt");
        // TODO: this log can become very long, put it behind a flag or think
        // about a more compact format
        // SRV_DBG("Prompt: %s\n", prompt.is_string() ?
        // prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process files
        mtmd::bitmaps bitmaps;
        const bool    has_mtmd = ctx_server.mctx != nullptr;
        {
            if (!has_mtmd && !files.empty()) {
                throw std::runtime_error("This server does not support multimodal");
            }
            for (auto & file : files) {
                mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, file.data(), file.size()));
                if (!bmp.ptr) {
                    throw std::runtime_error("Failed to load image or audio file");
                }
                // calculate bitmap hash (for KV caching)
                std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
                bmp.set_id(hash.c_str());
                bitmaps.entries.push_back(std::move(bmp));
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;
        if (oaicompat && !prompt.is_string()) {
            throw std::runtime_error("prompt must be a string");
        }

        if (oaicompat && has_mtmd) {
            // multimodal
            std::string     prompt_str = prompt.get<std::string>();
            mtmd_input_text inp_txt    = {
                prompt_str.c_str(),
                /* add_special */ true,
                /* parse_special */ true,
            };
            mtmd::input_chunks chunks(mtmd_input_chunks_init());
            auto               bitmaps_c_ptr = bitmaps.c_ptr();
            int32_t            tokenized =
                mtmd_tokenize(ctx_server.mctx, chunks.ptr.get(), &inp_txt, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
            if (tokenized != 0) {
                throw std::runtime_error("Failed to tokenize prompt");
            }

            server_tokens tmp(chunks, true);
            inputs.push_back(std::move(tmp));
        } else {
            // non-multimodal version
            auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
            for (auto & p : tokenized_prompts) {
                auto tmp = server_tokens(p, ctx_server.mctx != nullptr);
                inputs.push_back(std::move(tmp));
            }
        }

        tasks.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id    = ctx_server.queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens    = std::move(inputs[i]);
            task.params           = server_task::params_from_json_cmpl(ctx_server.ctx, ctx_server.params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat         = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(std::move(task));
        }

        task_ids = server_task::get_list_id(tasks);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(std::move(tasks));
    } catch (const std::exception & e) {
        res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        ctx_server.receive_multi_results(
            task_ids,
            [&](std::vector<server_task_result_ptr> & results) {
                if (results.size() == 1) {
                    // single result
                    res_ok(res, results[0]->to_json());
                } else {
                    // multiple results (multitask)
                    json arr = json::array();
                    for (auto & res : results) {
                        arr.push_back(res->to_json());
                    }
                    res_ok(res, arr);
                }
            },
            [&](const json & error_data) { res_error(res, error_data); }, is_connection_closed);

        ctx_server.queue_results.remove_waiting_task_ids(task_ids);
    } else {
        const auto chunked_content_provider = [task_ids, &ctx_server, oaicompat](size_t, httplib::DataSink & sink) {
            ctx_server.receive_cmpl_results_stream(
                task_ids,
                [&](server_task_result_ptr & result) -> bool {
                    json res_json = result->to_json();
                    if (res_json.is_array()) {
                        for (const auto & res : res_json) {
                            if (!server_sent_event(sink, "data", res)) {
                                // sending failed (HTTP connection closed), cancel the
                                // generation
                                return false;
                            }
                        }
                        return true;
                    } else {
                        return server_sent_event(sink, "data", res_json);
                    }
                },
                [&](const json & error_data) { server_sent_event(sink, "error", error_data); },
                [&sink]() {
                    // note: do not use req.is_connection_closed here because req
                    // is already destroyed
                    return !sink.is_writable();
                });
            if (oaicompat != OAICOMPAT_TYPE_NONE) {
                static const std::string ev_done = "data: [DONE]\n\n";
                sink.write(ev_done.data(), ev_done.size());
            }
            sink.done();
            return false;
        };

        auto on_complete = [task_ids, &ctx_server](bool) {
            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        };

        res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
    }
}

static void handle_completions(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    json                    data = json::parse(req.body);
    std::vector<raw_buffer> files;
    std::string             extra_text;

    // Handle "files" field for uploaded file references
    if (data.contains("files") && data["files"].is_array()) {
        for (const auto & filename_json : data["files"]) {
            std::string filename  = filename_json.get<std::string>();
            std::string file_path = "./uploads/" + filename;

            if (has_extension(filename, ".txt") || has_extension(filename, ".md") || has_extension(filename, ".py") ||
                has_extension(filename, ".cpp") || has_extension(filename, ".json") ||
                has_extension(filename, ".csv")) {
                std::string content = read_file(file_path);
                if (!content.empty()) {
                    extra_text += "\n--- File: " + filename + " ---\n";
                    extra_text += content + "\n";
                }
            } else if (has_extension(filename, ".pdf")) {
                std::string content = extract_pdf_text(file_path);
                if (!content.empty()) {
                    extra_text += "\n--- File: " + filename + " ---\n";
                    extra_text += content + "\n";
                }
            } else if (has_extension(filename, ".png") || has_extension(filename, ".jpg") ||
                       has_extension(filename, ".jpeg") || has_extension(filename, ".webp") ||
                       has_extension(filename, ".bmp") || has_extension(filename, ".mp3") ||
                       has_extension(filename, ".wav")) {
                if (ctx_server.mctx == nullptr) {
                    res_error(res,
                              format_error_response("Model does not support images/audio.", ERROR_TYPE_NOT_SUPPORTED));
                    return;
                }
                std::string content = read_file(file_path);
                if (!content.empty()) {
                    files.emplace_back(reinterpret_cast<const unsigned char *>(content.data()),
                                       reinterpret_cast<const unsigned char *>(content.data()) + content.size());
                }
            } else {
                // Unknown file type: skip or error
                continue;
            }
        }
    }

    if (!extra_text.empty()) {
        if (data.contains("prompt") && data["prompt"].is_string()) {
            data["prompt"] = data["prompt"].get<std::string>() + "\n" + extra_text;
        } else {
            data["prompt"] = extra_text;
        }
    }

    handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, files, req.is_connection_closed, res,
                            OAICOMPAT_TYPE_NONE, ctx_server);
}

static void handle_infill(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    // check model compatibility
    std::string err;
    if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
        err += "prefix token is missing. ";
    }
    if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
        err += "suffix token is missing. ";
    }
    if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
        err += "middle token is missing. ";
    }
    if (!err.empty()) {
        res_error(res, format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()),
                                             ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    json data = json::parse(req.body);

    // validate input
    if (data.contains("prompt") && !data.at("prompt").is_string()) {
        // prompt is optional
        res_error(res, format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
    }

    if (!data.contains("input_prefix")) {
        res_error(res, format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
    }

    if (!data.contains("input_suffix")) {
        res_error(res, format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
    }

    if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
        // input_extra is optional
        res_error(res, format_error_response("\"input_extra\" must be an array of "
                                             "{\"filename\": string, \"text\": string}",
                                             ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    json input_extra = json_value(data, "input_extra", json::array());
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        if (!chunk.contains("text") || !chunk.at("text").is_string()) {
            res_error(res, format_error_response("extra_context chunk must contain a "
                                                 "\"text\" field with a string value",
                                                 ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        // filename is optional
        if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
            res_error(res, format_error_response("extra_context chunk's \"filename\" field must be a string",
                                                 ERROR_TYPE_INVALID_REQUEST));
            return;
        }
    }
    data["input_extra"] = input_extra;  // default to empty array if it's not exist

    std::string               prompt            = json_value(data, "prompt", std::string());
    std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, false, true);
    SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
    data["prompt"] =
        format_infill(ctx_server.vocab, data.at("input_prefix"), data.at("input_suffix"), data.at("input_extra"),
                      ctx_server.params_base.n_batch, ctx_server.params_base.n_predict,
                      ctx_server.slots[0].n_ctx,  // TODO: there should be a better way
                      ctx_server.params_base.spm_infill, tokenized_prompts[0]);

    std::vector<raw_buffer> files;  // dummy
    handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, files, req.is_connection_closed, res,
                            OAICOMPAT_TYPE_CHAT, ctx_server);
}

static void handle_chat_completions(const httplib::Request & req, httplib::Response & res,
                                    server_context & ctx_server) {
    // LOG_DBG("request: %s\n", req.body.c_str());

    if (ctx_server.params_base.embedding) {
        res_error(res, format_error_response("This server does not support completions. Start it "
                                             "without `--embeddings`",
                                             ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    auto                    body = json::parse(req.body);
    std::vector<raw_buffer> files;
    std::string             extra_text;

    if (body.contains("files") && body["files"].is_array()) {
        for (const auto & filename_json : body["files"]) {
            std::string filename  = filename_json.get<std::string>();
            std::string file_path = "./uploads/" + filename;

            if (has_extension(filename, ".txt") || has_extension(filename, ".md") || has_extension(filename, ".py") ||
                has_extension(filename, ".cpp") || has_extension(filename, ".json") ||
                has_extension(filename, ".csv")) {
                std::string content = read_file(file_path);
                if (!content.empty()) {
                    extra_text += "\n--- File: " + filename + " ---\n";
                    extra_text += content + "\n";
                }
            } else if (has_extension(filename, ".pdf")) {
                std::string content = extract_pdf_text(file_path);
                if (!content.empty()) {
                    extra_text += "\n--- File: " + filename + " ---\n";
                    extra_text += content + "\n";
                }
            } else if (has_extension(filename, ".png") || has_extension(filename, ".jpg") ||
                       has_extension(filename, ".jpeg") || has_extension(filename, ".webp") ||
                       has_extension(filename, ".bmp") || has_extension(filename, ".mp3") ||
                       has_extension(filename, ".wav")) {
                if (ctx_server.mctx == nullptr) {
                    res_error(res,
                              format_error_response("Model does not support images/audio.", ERROR_TYPE_NOT_SUPPORTED));
                    return;
                }
                std::string content = read_file(file_path);
                if (!content.empty()) {
                    files.emplace_back(reinterpret_cast<const unsigned char *>(content.data()),
                                       reinterpret_cast<const unsigned char *>(content.data()) + content.size());
                }
            } else {
                continue;
            }
        }
    }

    // Append extra_text to the last user message in the chat
    if (!extra_text.empty()) {
        if (body.contains("messages") && body["messages"].is_array() && !body["messages"].empty()) {
            // Find the last user message
            for (int i = (int) body["messages"].size() - 1; i >= 0; --i) {
                if (body["messages"][i].contains("role") && body["messages"][i]["role"] == "user" &&
                    body["messages"][i].contains("content")) {
                    if (body["messages"][i]["content"].is_string()) {
                        body["messages"][i]["content"] =
                            body["messages"][i]["content"].get<std::string>() + "\n" + extra_text;
                    } else {
                        body["messages"][i]["content"] = extra_text;
                    }
                    break;
                }
            }
        }
    }

    json data = oaicompat_chat_params_parse(body, ctx_server.oai_parser_opt, files);
    handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, files, req.is_connection_closed, res,
                            OAICOMPAT_TYPE_CHAT, ctx_server);
}

// same with handle_chat_completions, but without inference part
static void handle_apply_template(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    auto                    body = json::parse(req.body);
    std::vector<raw_buffer> files;  // dummy, unused
    json                    data = oaicompat_chat_params_parse(body, ctx_server.oai_parser_opt, files);
    res_ok(res, {
                    { "prompt", std::move(data.at("prompt")) }
    });
}

static void handle_upload_file(const httplib::Request & req, httplib::Response & res) {
    // Only accept multipart/form-data
    if (!req.is_multipart_form_data()) {
        res_error(res, format_error_response("Content-Type must be multipart/form-data", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    // Assume the form field is named "file"
    if (!req.has_file("file")) {
        res_error(res,
                  format_error_response("No file uploaded (field name: 'file' missing)", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    const auto &        file     = req.get_file_value("file");
    const std::string & filename = file.filename;
    const std::string & filedata = file.content;

    // (Optional) Validate filename, e.g., prevent path traversal
    if (!fs_validate_filename(filename)) {
        res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    // (Optional) Choose where to save
    std::string save_path = "./uploads/" + filename;

    // Save the file
    std::ofstream ofs(save_path, std::ios::binary);
    if (!ofs) {
        res_error(res, format_error_response("Failed to open file for writing", ERROR_TYPE_SERVER));
        return;
    }
    ofs.write(filedata.data(), filedata.size());
    ofs.close();

    // Respond with success
    res_ok(res,
           {
               { "filename", filename                     },
               { "size",     filedata.size()              },
               { "message",  "File uploaded successfully" }
    });
}

static void handle_tokenize(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    const json body = json::parse(req.body);

    json tokens_response = json::array();
    if (body.count("content") != 0) {
        const bool add_special = json_value(body, "add_special", false);
        const bool with_pieces = json_value(body, "with_pieces", false);

        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, true);

        if (with_pieces) {
            for (const auto & token : tokens) {
                std::string piece = common_token_to_piece(ctx_server.ctx, token);
                json        piece_json;

                // Check if the piece is valid UTF-8
                if (is_valid_utf8(piece)) {
                    piece_json = piece;
                } else {
                    // If not valid UTF-8, store as array of byte values
                    piece_json = json::array();
                    for (unsigned char c : piece) {
                        piece_json.push_back(static_cast<int>(c));
                    }
                }

                tokens_response.push_back({
                    { "id",    token      },
                    { "piece", piece_json }
                });
            }
        } else {
            tokens_response = tokens;
        }
    }

    const json data = format_tokenizer_response(tokens_response);
    res_ok(res, data);
}

static void handle_detokenize(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    const json body = json::parse(req.body);

    std::string content;
    if (body.count("tokens") != 0) {
        const llama_tokens tokens = body.at("tokens");
        content                   = tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend());
    }

    const json data = format_detokenized_response(content);
    res_ok(res, data);
}

static void handle_embeddings_impl(const httplib::Request & req, httplib::Response & res, oaicompat_type oaicompat,
                                   server_context & ctx_server) {
    const json body = json::parse(req.body);

    if (oaicompat != OAICOMPAT_TYPE_NONE && llama_pooling_type(ctx_server.ctx) == LLAMA_POOLING_TYPE_NONE) {
        res_error(res, format_error_response("Pooling type 'none' is not OAI compatible. Please "
                                             "use a different pooling type",
                                             ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        oaicompat = OAICOMPAT_TYPE_NONE;  // "content" field is not OAI compatible
        prompt    = body.at("content");
    } else {
        res_error(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res_error(res, format_error_response("The format to return the embeddings "
                                                 "in. Can be either float or base64",
                                                 ERROR_TYPE_INVALID_REQUEST));
            return;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the
        // input
        if (tokens.empty()) {
            res_error(res, format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
    }

    // create and queue the task
    json                    responses = json::array();
    bool                    error     = false;
    std::unordered_set<int> task_ids;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id            = ctx_server.queue_tasks.get_new_id();
            task.index         = i;
            task.prompt_tokens = server_tokens(tokenized_prompts[i], ctx_server.mctx != nullptr);

            // OAI-compat
            task.params.oaicompat = oaicompat;

            tasks.push_back(std::move(task));
        }

        task_ids = server_task::get_list_id(tasks);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(std::move(tasks));
    }

    // get the result
    ctx_server.receive_multi_results(
        task_ids,
        [&](std::vector<server_task_result_ptr> & results) {
            for (auto & res : results) {
                GGML_ASSERT(dynamic_cast<server_task_result_embd *>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        },
        [&](const json & error_data) {
            res_error(res, error_data);
            error = true;
        },
        req.is_connection_closed);

    ctx_server.queue_results.remove_waiting_task_ids(task_ids);

    if (error) {
        return;
    }

    // write JSON response
    json root = oaicompat == OAICOMPAT_TYPE_EMBEDDING ?
                    format_embeddings_response_oaicompat(body, responses, use_base64) :
                    json(responses);
    res_ok(res, root);
}

static void handle_embeddings(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    handle_embeddings_impl(req, res, OAICOMPAT_TYPE_NONE, ctx_server);
}

static void handle_embeddings_oai(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    handle_embeddings_impl(req, res, OAICOMPAT_TYPE_EMBEDDING, ctx_server);
}

static void handle_completions_oai(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    json                    data = oaicompat_completion_params_parse(json::parse(req.body));
    std::vector<raw_buffer> files;  // dummy
    handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, files, req.is_connection_closed, res,
                            OAICOMPAT_TYPE_COMPLETION, ctx_server);
}

static void handle_rerank(const httplib::Request & req, httplib::Response & res, server_context & ctx_server) {
    if (!ctx_server.params_base.reranking || ctx_server.params_base.embedding) {
        res_error(res, format_error_response("This server does not support reranking. Start it "
                                             "with `--reranking` and without `--embedding`",
                                             ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    const json body          = json::parse(req.body);
    bool       is_tei_format = body.contains("texts");

    json query;
    if (body.count("query") == 1) {
        query = body.at("query");
        if (!query.is_string()) {
            res_error(res, format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
    } else {
        res_error(res, format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    std::vector<std::string> documents =
        json_value(body, "documents", json_value(body, "texts", std::vector<std::string>()));
    if (documents.empty()) {
        res_error(res,
                  format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    llama_tokens tokenized_query = tokenize_input_prompts(ctx_server.vocab, query, /* add_special */ false, true)[0];

    // create and queue the task
    json                    responses = json::array();
    bool                    error     = false;
    std::unordered_set<int> task_ids;
    {
        std::vector<server_task> tasks;
        auto tokenized_docs = tokenize_input_prompts(ctx_server.vocab, documents, /* add_special */ false, true);
        tasks.reserve(tokenized_docs.size());
        for (size_t i = 0; i < tokenized_docs.size(); i++) {
            auto        tmp    = format_rerank(ctx_server.vocab, tokenized_query, tokenized_docs[i]);
            server_task task   = server_task(SERVER_TASK_TYPE_RERANK);
            task.id            = ctx_server.queue_tasks.get_new_id();
            task.index         = i;
            task.prompt_tokens = server_tokens(tmp, ctx_server.mctx != nullptr);
            tasks.push_back(std::move(task));
        }

        task_ids = server_task::get_list_id(tasks);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(std::move(tasks));
    }

    ctx_server.receive_multi_results(
        task_ids,
        [&](std::vector<server_task_result_ptr> & results) {
            for (auto & res : results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank *>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        },
        [&](const json & error_data) {
            res_error(res, error_data);
            error = true;
        },
        req.is_connection_closed);

    if (error) {
        return;
    }

    // write JSON response
    json root = format_response_rerank(body, responses, is_tei_format, documents);

    res_ok(res, root);
}

static void handle_lora_adapters_list(const httplib::Request &, httplib::Response & res, server_context & ctx_server) {
    json         result = json::array();
    const auto & loras  = ctx_server.params_base.lora_adapters;
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        result.push_back({
            { "id",    i          },
            { "path",  lora.path  },
            { "scale", lora.scale },
        });
    }
    res_ok(res, result);
    res.status = 200;  // HTTP OK
}

static void handle_lora_adapters_apply(const httplib::Request & req, httplib::Response & res,
                                       server_context & ctx_server) {
    const json body = json::parse(req.body);
    if (!body.is_array()) {
        res_error(res, format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    int task_id = ctx_server.queue_tasks.get_new_id();
    {
        server_task task(SERVER_TASK_TYPE_SET_LORA);
        task.id       = task_id;
        task.set_lora = parse_lora_request(ctx_server.params_base.lora_adapters, body);
        ctx_server.queue_results.add_waiting_task_id(task_id);
        ctx_server.queue_tasks.post(std::move(task));
    }

    // get the result
    server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
    ctx_server.queue_results.remove_waiting_task_id(task_id);

    if (result->is_error()) {
        res_error(res, result->to_json());
        return;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_apply_lora *>(result.get()) != nullptr);
    res_ok(res, result->to_json());
};

static void handle_slots(const httplib::Request & req, httplib::Response & res, server_context & ctx_server,
                         common_params & params) {
    if (!params.endpoint_slots) {
        res_error(res, format_error_response("This server does not support slots "
                                             "endpoint. Start it with `--slots`",
                                             ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    // request slots data using task queue
    int task_id = ctx_server.queue_tasks.get_new_id();
    {
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.id = task_id;
        ctx_server.queue_results.add_waiting_task_id(task_id);
        ctx_server.queue_tasks.post(std::move(task), true);  // high-priority task
    }

    // get the result
    server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
    ctx_server.queue_results.remove_waiting_task_id(task_id);

    if (result->is_error()) {
        res_error(res, result->to_json());
        return;
    }

    // TODO: get rid of this dynamic_cast
    auto res_metrics = dynamic_cast<server_task_result_metrics *>(result.get());
    GGML_ASSERT(res_metrics != nullptr);

    // optionally return "fail_on_no_slot" error
    if (req.has_param("fail_on_no_slot")) {
        if (res_metrics->n_idle_slots == 0) {
            res_error(res, format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
            return;
        }
    }

    res_ok(res, res_metrics->slots_data);
};

static void handle_slots_save(const httplib::Request & req, httplib::Response & res, server_context & ctx_server,
                              common_params & params, int id_slot) {
    json        request_data = json::parse(req.body);
    std::string filename     = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return;
    }
    std::string filepath = params.slot_save_path + filename;

    int task_id = ctx_server.queue_tasks.get_new_id();
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id                   = task_id;
        task.slot_action.slot_id  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;

        ctx_server.queue_results.add_waiting_task_id(task_id);
        ctx_server.queue_tasks.post(std::move(task));
    }

    server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
    ctx_server.queue_results.remove_waiting_task_id(task_id);

    if (result->is_error()) {
        res_error(res, result->to_json());
        return;
    }

    res_ok(res, result->to_json());
};

static void handle_slots_restore(const httplib::Request & req, httplib::Response & res, server_context & ctx_server,
                                 common_params & params, int id_slot) {
    json        request_data = json::parse(req.body);
    std::string filename     = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return;
    }
    std::string filepath = params.slot_save_path + filename;

    int task_id = ctx_server.queue_tasks.get_new_id();
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id                   = task_id;
        task.slot_action.slot_id  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;

        ctx_server.queue_results.add_waiting_task_id(task_id);
        ctx_server.queue_tasks.post(std::move(task));
    }

    server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
    ctx_server.queue_results.remove_waiting_task_id(task_id);

    if (result->is_error()) {
        res_error(res, result->to_json());
        return;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load *>(result.get()) != nullptr);
    res_ok(res, result->to_json());
};

static void handle_slots_erase(const httplib::Request &, httplib::Response & res, server_context & ctx_server,
                               int id_slot) {
    int task_id = ctx_server.queue_tasks.get_new_id();
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id                  = task_id;
        task.slot_action.slot_id = id_slot;

        ctx_server.queue_results.add_waiting_task_id(task_id);
        ctx_server.queue_tasks.post(std::move(task));
    }

    server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
    ctx_server.queue_results.remove_waiting_task_id(task_id);

    if (result->is_error()) {
        res_error(res, result->to_json());
        return;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase *>(result.get()) != nullptr);
    res_ok(res, result->to_json());
};

static void handle_slots_action(const httplib::Request & req, httplib::Response & res, server_context & ctx_server,
                                common_params & params) {
    if (params.slot_save_path.empty()) {
        res_error(res, format_error_response("This server does not support slots action. Start "
                                             "it with `--slot-save-path`",
                                             ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    std::string id_slot_str = req.path_params.at("id_slot");
    int         id_slot;

    try {
        id_slot = std::stoi(id_slot_str);
    } catch (const std::exception &) {
        res_error(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    std::string action = req.get_param_value("action");

    if (action == "save") {
        handle_slots_save(req, res, ctx_server, params, id_slot);
    } else if (action == "restore") {
        handle_slots_restore(req, res, ctx_server, params, id_slot);
    } else if (action == "erase") {
        handle_slots_erase(req, res, ctx_server, id_slot);
    } else {
        res_error(res, format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
    }
}

void register_routes(httplib::Server & svr, server_context & ctx_server, common_params & params,
                     std::atomic<server_state> & state) {
    using namespace std::placeholders;

    // GET routes (no ambiguity)
    svr.Get("/health", handle_health);
    svr.Get("/metrics", std::bind(handle_metrics, _1, _2, std::ref(ctx_server), std::ref(params)));
    svr.Get("/props", std::bind(handle_props, _1, _2, std::ref(ctx_server)));
    svr.Get("/models", std::bind(handle_models, _1, _2, std::ref(params), std::ref(ctx_server), std::ref(state)));
    svr.Get("/v1/models", std::bind(handle_models, _1, _2, std::ref(params), std::ref(ctx_server), std::ref(state)));
    svr.Get("/api/tags", std::bind(handle_models, _1, _2, std::ref(params), std::ref(ctx_server), std::ref(state)));
    svr.Get("/slots", std::bind(handle_slots, _1, _2, std::ref(ctx_server), std::ref(params)));
    svr.Get("/lora-adapters", std::bind(handle_lora_adapters_list, _1, _2, std::ref(ctx_server)));

    // POST routes (ambiguity, so wrap in std::function)
    svr.Post("/props", std::function<void(const httplib::Request &, httplib::Response &)>(
                           std::bind(handle_props_change, _1, _2, std::ref(ctx_server))));
    svr.Post("/api/show", std::function<void(const httplib::Request &, httplib::Response &)>(
                              std::bind(handle_api_show, _1, _2, std::ref(ctx_server))));
    svr.Post("/completion", std::function<void(const httplib::Request &, httplib::Response &)>(
                                std::bind(handle_completions, _1, _2, std::ref(ctx_server))));
    svr.Post("/completions", std::function<void(const httplib::Request &, httplib::Response &)>(
                                 std::bind(handle_completions, _1, _2, std::ref(ctx_server))));
    svr.Post("/v1/completions", std::function<void(const httplib::Request &, httplib::Response &)>(
                                    std::bind(handle_completions_oai, _1, _2, std::ref(ctx_server))));
    svr.Post("/chat/completions", std::function<void(const httplib::Request &, httplib::Response &)>(
                                      std::bind(handle_chat_completions, _1, _2, std::ref(ctx_server))));
    svr.Post("/v1/chat/completions", std::function<void(const httplib::Request &, httplib::Response &)>(
                                         std::bind(handle_chat_completions, _1, _2, std::ref(ctx_server))));
    svr.Post("/api/chat", std::function<void(const httplib::Request &, httplib::Response &)>(
                              std::bind(handle_chat_completions, _1, _2, std::ref(ctx_server))));
    svr.Post("/infill", std::function<void(const httplib::Request &, httplib::Response &)>(
                            std::bind(handle_infill, _1, _2, std::ref(ctx_server))));
    svr.Post("/embedding", std::function<void(const httplib::Request &, httplib::Response &)>(
                               std::bind(handle_embeddings, _1, _2, std::ref(ctx_server))));
    svr.Post("/embeddings", std::function<void(const httplib::Request &, httplib::Response &)>(
                                std::bind(handle_embeddings, _1, _2, std::ref(ctx_server))));
    svr.Post("/v1/embeddings", std::function<void(const httplib::Request &, httplib::Response &)>(
                                   std::bind(handle_embeddings_oai, _1, _2, std::ref(ctx_server))));
    svr.Post("/rerank", std::function<void(const httplib::Request &, httplib::Response &)>(
                            std::bind(handle_rerank, _1, _2, std::ref(ctx_server))));
    svr.Post("/reranking", std::function<void(const httplib::Request &, httplib::Response &)>(
                               std::bind(handle_rerank, _1, _2, std::ref(ctx_server))));
    svr.Post("/v1/rerank", std::function<void(const httplib::Request &, httplib::Response &)>(
                               std::bind(handle_rerank, _1, _2, std::ref(ctx_server))));
    svr.Post("/v1/reranking", std::function<void(const httplib::Request &, httplib::Response &)>(
                                  std::bind(handle_rerank, _1, _2, std::ref(ctx_server))));
    svr.Post("/tokenize", std::function<void(const httplib::Request &, httplib::Response &)>(
                              std::bind(handle_tokenize, _1, _2, std::ref(ctx_server))));
    svr.Post("/detokenize", std::function<void(const httplib::Request &, httplib::Response &)>(
                                std::bind(handle_detokenize, _1, _2, std::ref(ctx_server))));
    svr.Post("/apply-template", std::function<void(const httplib::Request &, httplib::Response &)>(
                                    std::bind(handle_apply_template, _1, _2, std::ref(ctx_server))));
    svr.Post("/upload", handle_upload_file);
    svr.Post("/lora-adapters", std::function<void(const httplib::Request &, httplib::Response &)>(
                                   std::bind(handle_lora_adapters_apply, _1, _2, std::ref(ctx_server))));
    svr.Post("/slots/:id_slot", std::function<void(const httplib::Request &, httplib::Response &)>(
                                    std::bind(handle_slots_action, _1, _2, std::ref(ctx_server), std::ref(params))));
}
