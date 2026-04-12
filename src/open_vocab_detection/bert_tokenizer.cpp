#include "vision-core/open_vocab_detection/bert_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vision_core {

namespace {

std::string strToLower(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return result;
}

} // namespace

// ─── Construction ────────────────────────────────────────────────────────────

BertTokenizer::BertTokenizer(const std::string& vocab_file) {
    std::ifstream ifs(vocab_file);
    if (!ifs) {
        throw std::runtime_error("BertTokenizer: cannot open vocab file: " + vocab_file);
    }
    loadVocab(ifs, vocab_);
}

BertTokenizer::BertTokenizer(const std::string& vocab_text, bool /*preloaded_assets*/) {
    loadVocabFromText(vocab_text, vocab_);
}

// ─── Vocabulary loading ───────────────────────────────────────────────────────

void BertTokenizer::loadVocab(std::istream& stream, std::unordered_map<std::string, int32_t>& vocab) {
    std::string line;
    int32_t index = 0;
    while (std::getline(stream, line)) {
        // Strip trailing CR for Windows line endings
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            vocab[line] = index;
        }
        ++index;
    }
}

void BertTokenizer::loadVocabFromText(const std::string& text,
                                      std::unordered_map<std::string, int32_t>& vocab) {
    std::istringstream iss(text);
    loadVocab(iss, vocab);
}

// ─── Character classification ─────────────────────────────────────────────────

bool BertTokenizer::isWhitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool BertTokenizer::isPunctuation(char c) {
    const auto uc = static_cast<unsigned char>(c);
    if (uc >= 33U && uc <= 47U) {
        return true; // !"#$%&'()*+,-./
    }
    if (uc >= 58U && uc <= 64U) {
        return true; // :;<=>?@
    }
    if (uc >= 91U && uc <= 96U) {
        return true; // [\]^_`
    }
    if (uc >= 123U && uc <= 126U) {
        return true; // {|}~
    }
    return false;
}

// ─── Tokenization ─────────────────────────────────────────────────────────────

int32_t BertTokenizer::tokenToId(const std::string& token) const {
    const auto it = vocab_.find(token);
    return it != vocab_.end() ? it->second : UNK_TOKEN;
}

std::vector<std::string> BertTokenizer::basicTokenize(const std::string& text) const {
    const std::string lower = strToLower(text);

    // Add spaces around punctuation characters
    std::string spaced;
    spaced.reserve(lower.size() * 2);
    for (char c : lower) {
        if (isPunctuation(c)) {
            spaced.push_back(' ');
            spaced.push_back(c);
            spaced.push_back(' ');
        } else {
            spaced.push_back(c);
        }
    }

    // Split on whitespace
    std::vector<std::string> tokens;
    std::string current;
    for (char c : spaced) {
        if (isWhitespace(c)) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

std::vector<int32_t> BertTokenizer::wordpieceEncode(const std::string& word) const {
    // Fast path: whole word in vocabulary
    const auto it = vocab_.find(word);
    if (it != vocab_.end()) {
        return {it->second};
    }

    std::vector<int32_t> token_ids;
    size_t start = 0;
    const size_t word_len = word.size();
    bool is_bad = false;

    while (start < word_len) {
        size_t end = word_len;
        bool found = false;

        while (start < end) {
            std::string substr = word.substr(start, end - start);
            if (start > 0) {
                substr = "##" + substr;
            }
            const auto sit = vocab_.find(substr);
            if (sit != vocab_.end()) {
                token_ids.push_back(sit->second);
                found = true;
                break;
            }
            --end;
        }

        if (!found) {
            is_bad = true;
            break;
        }
        start = end;
    }

    if (is_bad) {
        return {UNK_TOKEN};
    }
    return token_ids;
}

std::vector<int32_t> BertTokenizer::tokenize(const std::string& text) const {
    const std::vector<std::string> words = basicTokenize(text);
    std::vector<int32_t> token_ids;
    for (const auto& word : words) {
        const auto pieces = wordpieceEncode(word);
        token_ids.insert(token_ids.end(), pieces.begin(), pieces.end());
    }
    return token_ids;
}

// ─── Public encode interface ──────────────────────────────────────────────────

std::pair<std::vector<int32_t>, std::vector<int32_t>>
BertTokenizer::encode(const std::string& text, int max_length) const {
    std::vector<int32_t> content = tokenize(text);

    // Truncate to max_length - 2 (reserve space for [CLS] and [SEP])
    const int content_max = max_length > 2 ? max_length - 2 : 0;
    if (static_cast<int>(content.size()) > content_max) {
        content.resize(static_cast<size_t>(content_max));
    }

    // Build: [CLS] tokens [SEP] [PAD...]
    std::vector<int32_t> input_ids;
    input_ids.reserve(static_cast<size_t>(max_length));
    input_ids.push_back(CLS_TOKEN);
    input_ids.insert(input_ids.end(), content.begin(), content.end());
    input_ids.push_back(SEP_TOKEN);

    std::vector<int32_t> attention_mask(input_ids.size(), 1);

    while (static_cast<int>(input_ids.size()) < max_length) {
        input_ids.push_back(PAD_TOKEN);
        attention_mask.push_back(0);
    }

    return {input_ids, attention_mask};
}

BertTokenizer::PhraseEncoding
BertTokenizer::encodePhrases(const std::vector<std::string>& phrases, int max_length) const {
    // Grounding DINO format: "phrase1 . phrase2 . phrase3 ."
    // Tokenise each phrase independently to obtain per-phrase token counts,
    // then join and compute token ranges relative to the final sequence.

    const int32_t dot_id = tokenToId(".");

    // Tokenise each phrase
    std::vector<std::vector<int32_t>> phrase_tokens;
    phrase_tokens.reserve(phrases.size());
    for (const auto& phrase : phrases) {
        phrase_tokens.push_back(tokenize(phrase));
    }

    // Assemble content: phrase_tokens[0] + [.] + phrase_tokens[1] + [.] + ...
    // Track [start, end) of each phrase in content_ids
    const int content_max = max_length > 2 ? max_length - 2 : 0;
    std::vector<int32_t> content_ids;
    std::vector<std::pair<int, int>> phrase_ranges_in_content;

    for (const auto& tok_ids : phrase_tokens) {
        const int start = static_cast<int>(content_ids.size());
        for (int32_t id : tok_ids) {
            if (static_cast<int>(content_ids.size()) >= content_max) {
                break;
            }
            content_ids.push_back(id);
        }
        const int end = static_cast<int>(content_ids.size());
        phrase_ranges_in_content.emplace_back(start, end);

        // Separator dot after every phrase
        if (static_cast<int>(content_ids.size()) < content_max) {
            content_ids.push_back(dot_id);
        }
    }

    // Build final sequence: [CLS] + content + [SEP] + [PAD...]
    PhraseEncoding result;
    result.input_ids.reserve(static_cast<size_t>(max_length));
    result.input_ids.push_back(CLS_TOKEN);
    result.input_ids.insert(result.input_ids.end(), content_ids.begin(), content_ids.end());
    result.input_ids.push_back(SEP_TOKEN);

    result.attention_mask.resize(result.input_ids.size(), 1);

    while (static_cast<int>(result.input_ids.size()) < max_length) {
        result.input_ids.push_back(PAD_TOKEN);
        result.attention_mask.push_back(0);
    }

    // Shift phrase ranges by 1 to account for the leading [CLS] token
    result.phrase_token_ranges.reserve(phrase_ranges_in_content.size());
    for (const auto& r : phrase_ranges_in_content) {
        result.phrase_token_ranges.emplace_back(r.first + 1, r.second + 1);
    }

    return result;
}

} // namespace vision_core
