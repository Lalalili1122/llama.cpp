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
    // Ensure upload directory exists
    std::filesystem::create_directories("./uploads");

    // Parse command-line parameters
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // Initialize any global/common resources
    common_init();

    // Initialize server context and load model ONCE
    server_context ctx_server;
    if (!ctx_server.load_model(params)) {
        if (!ctx_server.ctx) {
            fprintf(stderr, "[MAIN]: ctx_server.ctx is null\n");
        }
        if (!ctx_server.vocab) {
            fprintf(stderr, "[MAIN]: ctx_server.vocab is null\n");
        }
        fprintf(stderr, "[MAIN]: Model failed to load. Exiting.\n");
        return 1;
    }
    ctx_server.init();

    // Start backend and NUMA (after model load, before serving)
    llama_backend_init();
    llama_numa_init(params.numa);

    // Print system info
    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads,
            params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n%s\n\n", common_params_get_system_info(params).c_str());

    // Prepare HTTP server
    std::unique_ptr<httplib::Server> svr(new httplib::Server());

    if (!params.ssl_file_key.empty() || !params.ssl_file_cert.empty()) {
        LOG_ERR("Server is built without SSL support\n");
        return 1;
    }

    // Server state (atomic for thread safety)
    std::atomic<server_state> state{ SERVER_STATE_LOADING_MODEL };

    // Set default headers and logging
    svr->set_default_headers({
        { "Server", "llama.cpp" }
    });
    svr->set_logger(log_server_request);

    // Exception and error handlers
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
            LOG_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
        }
    });

    svr->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // For other error codes, skip processing here because it's already done by res_error()
    });

    // Set timeouts
    svr->set_read_timeout(params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    // Set up API key and server state middleware
    auto middleware_validate_api_key = [&params](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints{ "/health", "/models", "/v1/models",
                                                                       "/api/tags" };
        if (params.api_keys.empty()) {
            return true;
        }
        if (public_endpoints.count(req.path) || req.path == "/") {
            return true;
        }
        auto        auth_header = req.get_header_value("Authorization");
        std::string prefix      = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
                return true;
            }
        }
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
                return true;
            } else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };

    svr->set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "GET, POST");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/html");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Optionally serve web UI/static assets
    if (params.webui) {
        if (!params.public_path.empty()) {
            bool is_found = svr->set_mount_point("/", params.public_path);
            if (!is_found) {
                LOG_ERR("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                return 1;
            }
        } else {
            svr->Get("/", [](const httplib::Request & req, httplib::Response & res) {
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                } else {
                    res.set_header("Content-Encoding", "gzip");
                    res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                    res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                    res.set_content(reinterpret_cast<const char *>(index_html_gz), index_html_gz_len,
                                    "text/html; charset=utf-8");
                }
                return false;
            });
        }
    } else {
        LOG_INF("Web UI is disabled\n");
    }

    // Set HTTP server thread pool size
    if (params.n_threads_http < 1) {
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    svr->new_task_queue = [&params] {
        return new httplib::ThreadPool(params.n_threads_http);
    };

    // Clean up function
    auto clean_up = [&svr, &ctx_server]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        svr->stop();
        ctx_server.queue_results.terminate();
        llama_backend_free();
    };

    // Bind HTTP server
    bool was_bound = false;
    if (string_ends_with(params.hostname, ".sock")) {
        LOG_INF("%s: setting address family to AF_UNIX\n", __func__);
        svr->set_address_family(AF_UNIX);
        was_bound = svr->bind_to_port(params.hostname, 8080);
    } else {
        LOG_INF("%s: binding port with default address family\n", __func__);
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

    // Register all API routes (after model is loaded and ctx_server is initialized)
    register_routes(*svr, ctx_server, params, state);

    // Set server state to READY
    state.store(SERVER_STATE_READY);

    // Print chat template info
    LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
            common_chat_templates_source(ctx_server.chat_templates.get()),
            common_chat_format_example(ctx_server.chat_templates.get(), ctx_server.params_base.use_jinja).c_str());

    // Set up task processing
    ctx_server.queue_tasks.on_new_task(
        [&ctx_server](server_task && task) { ctx_server.process_single_task(std::move(task)); });
    ctx_server.queue_tasks.on_update_slots([&ctx_server]() { ctx_server.update_slots(); });

    // Set up shutdown handler and signals
    shutdown_handler = [&](int) {
        ctx_server.queue_tasks.terminate();
    };
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);

    // Start HTTP server in a thread
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    LOG_INF("%s: HTTP server is listening, hostname: %s, port: %d, http threads: %d\n", __func__,
            params.hostname.c_str(), params.port, params.n_threads_http);
    LOG_INF("%s: server is listening on http://%s:%d - starting the main loop\n", __func__, params.hostname.c_str(),
            params.port);

    // Block main thread until shutdown
    ctx_server.queue_tasks.start_loop();

    // Cleanup and exit
    clean_up();
    t.join();

    return 0;
}
