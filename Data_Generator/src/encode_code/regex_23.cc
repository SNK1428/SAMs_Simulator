#include "regex_23.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <regex>


// 使用正则表达式分隔字符串的方法
std::vector<std::string> split_reg(const std::string& str, const std::string& delimiterRegex) {
    std::regex re(delimiterRegex);
    std::sregex_token_iterator it(str.begin(), str.end(), re, -1);
    std::sregex_token_iterator end;

    std::vector<std::string> result;
    result.reserve(std::distance(it, end)); // 预先分配内存

    for (; it != end; ++it) {
        result.push_back(it->str());
    }
    return result;
}

/**
 * @brief Enumeration for different types of events that can occur while parsing.
 */
enum Event
{
    UPPER_LETTER,  ///< An uppercase letter event
    LOWER_LETTER,  ///< A lowercase letter event
    DIGIT,         ///< A digit event
    DOT,           ///< A dot (.) event
    OPEN_BRACKET,  ///< An opening bracket event
    CLOSE_BRACKET, ///< A closing bracket event
    OTHER,         ///< Any other character event
    EVENT_COUNT    ///< Used to represent the number of event types
};

/**
 * @brief Get the event type based on the character.
 *
 * @param ch The character to evaluate.
 * @return The corresponding event type.
 */
inline static Event getEvent(char ch)
{
    static std::vector<Event> eventTable(256, OTHER);
    static bool initialized = false;
    if (!initialized)
    {
        // Initialize the event table for different character ranges
        for (char c = 'A'; c <= 'Z'; ++c)
            eventTable[c] = UPPER_LETTER;
        for (char c = 'a'; c <= 'z'; ++c)
            eventTable[c] = LOWER_LETTER;
        for (char c = '0'; c <= '9'; ++c)
            eventTable[c] = DIGIT;
        eventTable['.'] = DOT;
        eventTable['('] = OPEN_BRACKET;
        eventTable[')'] = CLOSE_BRACKET;
        eventTable['['] = OPEN_BRACKET;
        eventTable[']'] = CLOSE_BRACKET;
        initialized = true;
    }
    return eventTable[static_cast<unsigned char>(ch)];
}

// Type alias for the event handler function
using EventHandler = std::function<void(char, std::string &, double &, std::map<std::string, double> &, std::stack<std::map<std::string, double>> &, std::stack<double> &, double &,
        const std::string &, size_t &, const std::unordered_set<std::string> &)>;

/**
 * @brief Handle uppercase letter event.
 */
inline static void handleUpperLetter(char ch, std::string &element, double &count, std::map<std::string, double> &elementCounts, std::stack<std::map<std::string, double>> &,
        std::stack<double> &, double multiplier, const std::string &, size_t &, const std::unordered_set<std::string> &)
{
    if (!element.empty())
    {
        if (count == 0)
            count = 1;
        elementCounts[element] += count * multiplier;
        element.clear();
        count = 0;
    }
    element = ch;
}

/**
 * @brief Handle lowercase letter event.
 */
inline static void handleLowerLetter(char ch, std::string &element, double &, std::map<std::string, double> &, std::stack<std::map<std::string, double>> &, std::stack<double> &,
        double &, const std::string &, size_t &, const std::unordered_set<std::string> &)
{
    element += ch;
}

/**
 * @brief Handle digit event.
 */
inline static void handleDigit(char ch, std::string &, double &count, std::map<std::string, double> &, std::stack<std::map<std::string, double>> &, std::stack<double> &, double &,
        const std::string &, size_t &, const std::unordered_set<std::string> &)
{
    count = count * 10 + (ch - '0');
}

/**
 * @brief Handle dot (.) event.
 */
inline static void handleDot(char, std::string &, double &count, std::map<std::string, double> &, std::stack<std::map<std::string, double>> &, std::stack<double> &, double &,
        const std::string &formula, size_t &index, const std::unordered_set<std::string> &)
{
    ++index;
    double fraction = 0.0;
    double divisor = 10.0;
    while (index < formula.size() && std::isdigit(formula[index]))
    {
        fraction += (formula[index] - '0') / divisor;
        divisor *= 10.0;
        ++index;
    }
    --index;
    count += fraction;
}

/**
 * @brief Handle opening bracket event.
 */
inline static void handleOpenBracket(char, std::string &element, double &count, std::map<std::string, double> &elementCounts,
        std::stack<std::map<std::string, double>> &stackCounts, std::stack<double> &stackMultipliers, double &multiplier, const std::string &,
        size_t &, const std::unordered_set<std::string> &)
{
    if (!element.empty())
    {
        if (count == 0)
            count = 1;
        elementCounts[element] += count * multiplier;
        element.clear();
        count = 0;
    }
    stackCounts.push(elementCounts);
    elementCounts.clear();
    stackMultipliers.push(multiplier);
    multiplier = 1;
}

/**
 * @brief Handle closing bracket event.
 */
inline static void handleCloseBracket(char, std::string &element, double &count, std::map<std::string, double> &elementCounts,
        std::stack<std::map<std::string, double>> &stackCounts, std::stack<double> &stackMultipliers, double &multiplier, const std::string &formula,
        size_t &i, const std::unordered_set<std::string> &predefinedSymbols)
{
    if (!element.empty())
    {
        if (count == 0)
            count = 1;
        elementCounts[element] += count * multiplier;
        element.clear();
        count = 0;
    }
    multiplier = 0;
    ++i;
    while (i < formula.size() && (std::isdigit(formula[i]) || formula[i] == '.'))
    {
        if (formula[i] == '.')
        {
            handleDot('.', element, multiplier, elementCounts, stackCounts, stackMultipliers, multiplier, formula, i, predefinedSymbols);
        }
        else
        {
            multiplier = multiplier * 10 + (formula[i] - '0');
        }
        ++i;
    }
    --i;
    if (multiplier == 0)
        multiplier = 1;
    for (auto &pair : elementCounts)
    {
        pair.second *= multiplier;
    }
    if (!stackCounts.empty())
    {
        std::map<std::string, double> previousCounts = stackCounts.top();
        stackCounts.pop();
        for (const auto &pair : elementCounts)
        {
            previousCounts[pair.first] += pair.second;
        }
        elementCounts = previousCounts;
    }
    multiplier = stackMultipliers.top();
    stackMultipliers.pop();
}

/**
 * @brief Handle other character events.
 */
inline static void handleOther(char, std::string &, double &, std::map<std::string, double> &, std::stack<std::map<std::string, double>> &, std::stack<double> &, double &,
        const std::string &, size_t &, const std::unordered_set<std::string> &)
{
}

/**
 * @brief Event handler function table.
 */
static const std::vector<EventHandler> eventHandlers = {handleUpperLetter, handleLowerLetter, handleDigit, handleDot, handleOpenBracket, handleCloseBracket, handleOther};

/**
 * @brief Backtracking function to parse chemical formulas.
 *
 * @param formula The chemical formula string.
 * @param index Current index in the formula string.
 * @param predefinedSymbols Set of predefined symbols.
 * @param elementCounts Map to store element counts.
 * @param stackCounts Stack to store element counts for nested structures.
 * @param stackMultipliers Stack to store multipliers for nested structures.
 * @param element Current element being processed.
 * @param count Current count of the element being processed.
 * @param multiplier Current multiplier for nested structures.
 * @param results Vector to store the final results of element counts.
 */
static void backtrack(const std::string &formula, size_t index, const std::unordered_set<std::string> &predefinedSymbols, std::map<std::string, double> &elementCounts,
        std::stack<std::map<std::string, double>> &stackCounts, std::stack<double> &stackMultipliers, std::string &element, double &count, double &multiplier,
        std::vector<std::map<std::string, double>> &results)
{
    if (index == formula.size())
    {
        if (!element.empty())
        {
            if (count == 0)
                count = 1;
            elementCounts[element] += count * multiplier;
        }
        results.push_back(elementCounts);
        return;
    }

    std::unordered_map<size_t, std::string> possibleSymbols;
    for (const auto &s : predefinedSymbols)
    {
        if (formula.substr(index, s.size()) == s)
        {
            possibleSymbols[s.size()] = s;
        }
    }

    if (possibleSymbols.size() > 1)
    {
        std::cout << "Multiple possible symbols at position " << index << " in " << formula << ":\n";
        int choiceIndex = 1;
        std::vector<std::string> choices;
        for (auto it = possibleSymbols.begin(); it != possibleSymbols.end(); ++it)
        {
            std::cout << choiceIndex << ". " << it->second << "\n";
            choices.push_back(it->second);
            ++choiceIndex;
        }
        std::cout << "Choose one (enter the number): ";
        int chosenIndex;
        std::cin >> chosenIndex;

        if (chosenIndex < 1 || chosenIndex > choices.size())
        {
            std::cerr << "Invalid choice. Exiting.\n";
            exit(1);
        }

        std::string chosenSymbol = choices[chosenIndex - 1];

        if (!element.empty())
        {
            if (count == 0)
                count = 1;
            elementCounts[element] += count * multiplier;
            element.clear();
            count = 0;
        }
        element = chosenSymbol;
        backtrack(formula, index + chosenSymbol.size(), predefinedSymbols, elementCounts, stackCounts, stackMultipliers, element, count, multiplier, results);
        element.clear();
        return;
    }
    else if (possibleSymbols.size() == 1)
    {
        std::string symbol = possibleSymbols.begin()->second;
        if (!element.empty())
        {
            if (count == 0)
                count = 1;
            elementCounts[element] += count * multiplier;
            element.clear();
            count = 0;
        }
        element = symbol;
        backtrack(formula, index + symbol.size(), predefinedSymbols, elementCounts, stackCounts, stackMultipliers, element, count, multiplier, results);
        element.clear();
        return;
    }

    // 事件处理
    char ch = formula[index];
    Event event = getEvent(ch);
    eventHandlers[event](ch, element, count, elementCounts, stackCounts, stackMultipliers, multiplier, formula, index, predefinedSymbols);
    backtrack(formula, index + 1, predefinedSymbols, elementCounts, stackCounts, stackMultipliers, element, count, multiplier, results);

    // 回溯处理
    if (!element.empty())
    {
        elementCounts[element] -= count * multiplier;
        if (elementCounts[element] == 0)
        {
            elementCounts.erase(element);
        }
        element.clear();
        count = 0;
    }
}

/**
 * @brief Parse a chemical formula into its constituent elements and their counts.
 *
 * @param formula The chemical formula string.
 * @param predefinedSymbols Set of predefined symbols.
 * @return A vector of maps containing element counts for each possible interpretation.
 */
inline static std::vector<std::map<std::string, double>> parse_chemical_formula(const std::string &formula, const std::unordered_set<std::string> &predefinedSymbols)
{
    std::map<std::string, double> elementCounts;
    std::stack<std::map<std::string, double>> stackCounts;
    std::stack<double> stackMultipliers;
    std::string element;
    double count = 0;
    double multiplier = 1;
    std::vector<std::map<std::string, double>> results;
    results.reserve(formula.size());
    backtrack(formula, 0, predefinedSymbols, elementCounts, stackCounts, stackMultipliers, element, count, multiplier, results);
    return results;
}

/**
 * @brief Read predefined symbols from a file.
 *
 * @param filename The name of the file containing the predefined symbols.
 * @return A set of predefined symbols.
 */
inline static std::unordered_set<std::string> readPredefinedSymbols(const std::string &filename)
{
    std::unordered_set<std::string> symbols;
    std::ifstream infile(filename);
    std::string line;
    while (std::getline(infile, line))
    {
        symbols.insert(line);
    }
    return symbols;
}

std::vector<std::vector<std::string>> build_23_raw_results(const std::string& defined_symbols_path, const std::vector<std::string>& formulas)
{
    auto predefinedSymbols = readPredefinedSymbols(defined_symbols_path);
    std::vector<std::vector<std::string>> results;
    results.reserve(formulas.size());
    for(const auto formula : formulas)
    {
        std::vector<std::map<std::string, double>> elementCountsList = parse_chemical_formula(formula, predefinedSymbols);
        std::vector<std::string> encoded_formulas;
        encoded_formulas.reserve(elementCountsList.size()*2);
        for (const auto &elementCounts : elementCountsList)
            for (const auto &pair : elementCounts)
            {
                encoded_formulas.emplace_back(pair.first);
                encoded_formulas.emplace_back(std::to_string(pair.second));
            }
        results.emplace_back(std::move(encoded_formulas));
    }
    return results;
}

