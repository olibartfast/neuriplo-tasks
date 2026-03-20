#include "vision-core/open_vocab_detection/clip_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace vision_core {

namespace {

std::string trim(const std::string& input) {
    const auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string readFileContents(const std::string& path) {
    std::ifstream stream(path);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::stringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void parseVocabContent(const std::string& vocab_content, std::unordered_map<std::string, int32_t>& vocab) {
    std::regex entry_regex(R"json("([^"]+)"\s*:\s*([0-9]+))json");
    for (std::sregex_iterator it(vocab_content.begin(), vocab_content.end(), entry_regex), end; it != end; ++it) {
        vocab.emplace((*it)[1].str(), static_cast<int32_t>(std::stoi((*it)[2].str())));
    }

    if (vocab.empty()) {
        throw std::runtime_error("Failed to parse tokenizer vocabulary content");
    }
}

void parseMergesContent(const std::string& merges_content, std::vector<std::pair<std::string, std::string>>& merges,
                        std::unordered_map<std::string, int>& merge_rank) {
    std::istringstream merges_stream(merges_content);
    std::string line;
    int rank = 0;
    while (std::getline(merges_stream, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::string left;
        std::string right;
        if (!(iss >> left >> right)) {
            continue;
        }
        merges.emplace_back(left, right);
        merge_rank.emplace(left + " " + right, rank++);
    }
}

} // namespace

ClipTokenizer::ClipTokenizer(const std::string& vocab_file, const std::string& merges_file)
    : ClipTokenizer(readFileContents(vocab_file), readFileContents(merges_file), true) {}

ClipTokenizer::ClipTokenizer(const std::string& vocab_json, const std::string& merges_text, bool preloaded_assets)
    : byte_encoder_(buildByteEncoder()) {
    if (!preloaded_assets) {
        throw std::invalid_argument("Use the two-argument constructor for file paths");
    }
    parseVocabContent(vocab_json, vocab_);
    parseMergesContent(merges_text, merges_, merge_rank_);
}

std::vector<int32_t> ClipTokenizer::encode(const std::string& text, int context_length) const {
    if (context_length <= 0) {
        throw std::invalid_argument("context_length must be positive");
    }

    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(context_length));
    ids.push_back(SOT_TOKEN);

    const std::vector<int32_t> tokens = tokenize(text);
    ids.insert(ids.end(), tokens.begin(), tokens.end());
    ids.push_back(EOT_TOKEN);

    if (ids.size() > static_cast<size_t>(context_length)) {
        ids.resize(static_cast<size_t>(context_length));
        ids.back() = EOT_TOKEN;
    }

    ids.resize(static_cast<size_t>(context_length), PAD_TOKEN);
    return ids;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> ClipTokenizer::batchEncode(const std::vector<std::string>& texts,
                                                                                 int context_length) const {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    input_ids.reserve(texts.size() * static_cast<size_t>(context_length));
    attention_mask.reserve(texts.size() * static_cast<size_t>(context_length));

    for (const auto& text : texts) {
        const std::vector<int32_t> encoded = encode(text, context_length);
        input_ids.insert(input_ids.end(), encoded.begin(), encoded.end());
        for (int32_t token : encoded) {
            attention_mask.push_back(token == PAD_TOKEN ? 0 : 1);
        }
    }

    return {input_ids, attention_mask};
}

std::unordered_map<uint8_t, std::string> ClipTokenizer::buildByteEncoder() {
    std::unordered_map<uint8_t, std::string> encoder;
    for (int value = 0; value < 256; ++value) {
        encoder.emplace(static_cast<uint8_t>(value), std::string(1, static_cast<char>(value)));
    }
    return encoder;
}

std::string ClipTokenizer::bpe(const std::string& token) const {
    if (token.empty()) {
        return token;
    }

    if (vocab_.count(token) > 0U) {
        return token;
    }
    if (vocab_.count(token + "</w>") > 0U) {
        return token + "</w>";
    }
    return token;
}

std::vector<int32_t> ClipTokenizer::tokenize(const std::string& text) const {
    std::vector<int32_t> token_ids;
    std::regex token_regex(R"([A-Za-z0-9]+|[^\sA-Za-z0-9])");

    for (std::sregex_iterator it(text.begin(), text.end(), token_regex), end; it != end; ++it) {
        std::string piece = toLower((*it).str());
        piece = bpe(piece);

        auto vocab_it = vocab_.find(piece);
        if (vocab_it == vocab_.end()) {
            vocab_it = vocab_.find(piece + "</w>");
        }
        if (vocab_it == vocab_.end()) {
            vocab_it = vocab_.find("<|endoftext|>");
        }
        if (vocab_it == vocab_.end()) {
            throw std::runtime_error("Tokenizer vocabulary does not contain token: " + piece);
        }
        token_ids.push_back(vocab_it->second);
    }

    return token_ids;
}

} // namespace vision_core
