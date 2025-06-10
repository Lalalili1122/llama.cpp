#include "server.hpp"

bool has_extension(const std::string & filename, const std::string & ext) {
    if (filename.length() >= ext.length()) {
        return (0 == filename.compare(filename.length() - ext.length(), ext.length(), ext));
    }
    return false;
}

std::string read_file(const std::string & filepath) {
    std::ifstream     ifs(filepath, std::ios::binary);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

std::string read_file_to_string(const std::string & filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        return "";
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

std::string extract_pdf_text(const std::string & file_path) {
    std::ostringstream                 all_text;
    std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(file_path));
    if (!doc) {
        return "";
    }
    for (int i = 0; i < doc->pages(); ++i) {
        std::unique_ptr<poppler::page> page(doc->create_page(i));
        if (page) {
            auto utf8 = page->text().to_utf8();
            all_text << std::string(utf8.data(), utf8.size()) << "\n";
        }
    }
    return all_text.str();
}

void res_ok(httplib::Response & res, const json & data) {
    res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
    res.status = 200;
}

void res_error(httplib::Response & res, const json & error_data) {
    json final_response{
        { "error", error_data }
    };
    res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
    res.status = json_value(error_data, "code", 500);
}
