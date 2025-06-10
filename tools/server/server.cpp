#include "server.hpp"

#include "index.html.gz.hpp"
#include "loading.html.hpp"

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health" || req.path == "/v1/completions") {
        return;
    }

    // reminder: this function is not covered by httplib's exception handler; if someone does more complicated stuff, think about wrapping it in try-catch
    SRV_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);

    SRV_DBG("request:  %s\n", req.body.c_str());
    SRV_DBG("response: %s\n", res.body.c_str());
}

std::function<void(int)> shutdown_handler;
std::atomic_flag         is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

int main(int argc, char ** argv) {
    std::filesystem::create_directories("./uploads");
    // own arguments required by this example
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    common_init();

    // struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads,
            params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    std::unique_ptr<httplib::Server> svr;
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_ERR("Server is built without SSL support\n");
        return 1;
    }
    svr.reset(new httplib::Server());

    std::atomic<server_state> state{ SERVER_STATE_LOADING_MODEL };

    svr->set_default_headers({
        { "Server", "llama.cpp" }
    });
    svr->set_logger(log_server_request);

    svr->set_exception_handler([](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        try {
            json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
            LOG_WRN("got exception: %s\n", formatted_error.dump().c_str());
            res_error(res, formatted_error);
        } catch (const std::exception & e) {
            LOG_ERR("got another exception: %s | while hanlding exception: %s\n", e.what(), message.c_str());
        }
    });

    svr->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout(params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params.hostname;
    log_data["port"]     = std::to_string(params.port);

    if (params.api_keys.size() == 1) {
        auto key            = params.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int) (key.length() - 4), 0));
    } else if (params.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(params.api_keys.size()) + " keys loaded";
    }

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

    //
    // Middlewares
    //

    auto middleware_validate_api_key = [&params](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints = { "/health", "/models", "/v1/models",
                                                                          "/api/tags" };

        // If API key is not set, skip validation
        if (params.api_keys.empty()) {
            return true;
        }

        // If path is public or is static file, skip validation
        if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");

        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
                return true;  // API key is valid
            }
        }

        // API key is invalid or not provided
        res_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

        LOG_WRN("Unauthorized: Invalid API Key\n");

        return false;
    };

    auto middleware_server_state = [&state](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char *>(loading_html), loading_html_len,
                                "text/html; charset=utf-8");
                res.status = 503;
            } else if (req.path == "/models" || req.path == "/v1/models" || req.path == "/api/tags") {
                // allow the models endpoint to be accessed during loading
                return true;
            } else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key, &middleware_server_state](const httplib::Request & req,
                                                                                          httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "GET, POST");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/html");                  // blank response, no data
            return httplib::Server::HandlerResponse::Handled;  // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    register_routes(*svr, ctx_server, params, state);

    //
    // Router
    //

    if (!params.webui) {
        LOG_INF("Web UI is disabled\n");
    } else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            bool is_found = svr->set_mount_point("/", params.public_path);
            if (!is_found) {
                LOG_ERR("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                return 1;
            }
        } else {
            // using embedded static index.html
            svr->Get("/", [](const httplib::Request & req, httplib::Response & res) {
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                } else {
                    res.set_header("Content-Encoding", "gzip");
                    // COEP and COOP headers, required by pyodide (python interpreter)
                    res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                    res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                    res.set_content(reinterpret_cast<const char *>(index_html_gz), index_html_gz_len,
                                    "text/html; charset=utf-8");
                }
                return false;
            });
        }
    }

    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] = std::to_string(params.n_threads_http);
    svr->new_task_queue        = [&params] {
        return new httplib::ThreadPool(params.n_threads_http);
    };

    // clean up function, to be called before exit
    auto clean_up = [&svr, &ctx_server]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        svr->stop();
        ctx_server.queue_results.terminate();
        llama_backend_free();
    };

    bool was_bound = false;
    if (string_ends_with(std::string(params.hostname), ".sock")) {
        LOG_INF("%s: setting address family to AF_UNIX\n", __func__);
        svr->set_address_family(AF_UNIX);
        // bind_to_port requires a second arg, any value other than 0 should
        // simply get ignored
        was_bound = svr->bind_to_port(params.hostname, 8080);
    } else {
        LOG_INF("%s: binding port with default address family\n", __func__);
        // bind HTTP listen port
        if (params.port == 0) {
            int bound_port = svr->bind_to_any_port(params.hostname);
            if ((was_bound = (bound_port >= 0))) {
                params.port = bound_port;
            }
        } else {
            was_bound = svr->bind_to_port(params.hostname, params.port);
        }
    }

    if (!was_bound) {
        LOG_ERR("%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n", __func__, params.hostname.c_str(),
                params.port);
        clean_up();
        return 1;
    }

    // run the HTTP server in a thread
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    LOG_INF("%s: HTTP server is listening, hostname: %s, port: %d, http threads: %d\n", __func__,
            params.hostname.c_str(), params.port, params.n_threads_http);

    // load the model
    LOG_INF("%s: loading model\n", __func__);

    if (!ctx_server.load_model(params)) {
        clean_up();
        t.join();
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return 1;
    }
    ctx_server.init();
    state.store(SERVER_STATE_READY);

    LOG_INF("%s: model loaded\n", __func__);

    // print sample chat example to make it clear which template is used
    LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
            common_chat_templates_source(ctx_server.chat_templates.get()),
            common_chat_format_example(ctx_server.chat_templates.get(), ctx_server.params_base.use_jinja).c_str());

    ctx_server.queue_tasks.on_new_task(
        [&ctx_server](server_task && task) { ctx_server.process_single_task(std::move(task)); });

    ctx_server.queue_tasks.on_update_slots([&ctx_server]() { ctx_server.update_slots(); });

    shutdown_handler = [&](int) {
        // this will unblock start_loop()
        ctx_server.queue_tasks.terminate();
    };

    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);

    LOG_INF("%s: server is listening on http://%s:%d - starting the main loop\n", __func__, params.hostname.c_str(),
            params.port);

    // this call blocks the main thread until queue_tasks.terminate() is called
    ctx_server.queue_tasks.start_loop();

    clean_up();
    t.join();

    return 0;
}
