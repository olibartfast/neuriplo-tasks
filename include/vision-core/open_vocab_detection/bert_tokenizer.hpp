#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vision_core {

/**
 * @brief BERT-style WordPiece tokenizer for Grounding DINO
 *
 * Implements the WordPiece tokenization algorithm used by BERT-based models.
 * Loads vocabulary from a standard vocab.txt file (one token per line, line index = ID).
 *
 * Grounding DINO encodes all query phrases as a single concatenated text:
 *   "phrase1 . phrase2 . phrase3 ."
 * and uses token-level output logits to match detected boxes to input phrases.
 *
 * Usage:
 * @code
 *   BertTokenizer tok("/path/to/vocab.txt");
 *   auto enc = tok.encodePhrases({"cat", "dog"}, 256);
 *   // enc.input_ids         – full token-ID sequence (length = max_length)
 *   // enc.attention_mask    – 1 for real tokens, 0 for padding
 *   // enc.phrase_token_ranges[i] – [start, end) of phrase i in the sequence
 * @endcode
 */
class BertTokenizer {
  public:
    static constexpr int32_t PAD_TOKEN = 0;   ///< [PAD] token id
    static constexpr int32_t UNK_TOKEN = 100; ///< [UNK] token id
    static constexpr int32_t CLS_TOKEN = 101; ///< [CLS] token id
    static constexpr int32_t SEP_TOKEN = 102; ///< [SEP] token id

    /**
     * @brief Construct from a vocab.txt file
     * @param vocab_file Path to vocab.txt (one token per line)
     * @throws std::runtime_error if the file cannot be opened
     */
    explicit BertTokenizer(const std::string& vocab_file);

    /**
     * @brief Construct from preloaded vocab.txt content
     * @param vocab_text  Raw contents of vocab.txt
     * @param preloaded_assets Must be true to select this overload
     * @throws std::runtime_error if content cannot be parsed
     */
    BertTokenizer(const std::string& vocab_text, bool preloaded_assets);

    /**
     * @brief Encode a single text string
     * @param text       Input text (UTF-8)
     * @param max_length Maximum sequence length (padded / truncated)
     * @return Pair (input_ids, attention_mask)
     */
    [[nodiscard]] std::pair<std::vector<int32_t>, std::vector<int32_t>> encode(const std::string& text,
                                                                               int max_length = 256) const;

    /**
     * @brief Phrase encoding result for Grounding DINO
     */
    struct PhraseEncoding {
        std::vector<int32_t> input_ids;
        std::vector<int32_t> attention_mask;
        /// [start, end) token indices in the sequence for each input phrase
        std::vector<std::pair<int, int>> phrase_token_ranges;
    };

    /**
     * @brief Encode phrases in Grounding DINO format: "phrase1 . phrase2 . "
     * @param phrases    Input phrase list
     * @param max_length Maximum total sequence length (padded / truncated)
     * @return PhraseEncoding with token IDs, mask, and per-phrase token ranges
     */
    [[nodiscard]] PhraseEncoding encodePhrases(const std::vector<std::string>& phrases, int max_length = 256) const;

  private:
    std::unordered_map<std::string, int32_t> vocab_;

    static void loadVocab(std::istream& stream, std::unordered_map<std::string, int32_t>& vocab);
    static void loadVocabFromText(const std::string& text, std::unordered_map<std::string, int32_t>& vocab);

    [[nodiscard]] int32_t tokenToId(const std::string& token) const;

    [[nodiscard]] std::vector<std::string> basicTokenize(const std::string& text) const;
    [[nodiscard]] std::vector<int32_t> wordpieceEncode(const std::string& word) const;
    [[nodiscard]] std::vector<int32_t> tokenize(const std::string& text) const;

    static bool isPunctuation(char c);
    static bool isWhitespace(char c);
};

} // namespace vision_core
