
// IntersectionStrategyRegistry.cpp
#include "./IntersectionStrategyRegistry.h"
#include "./IntersectionStrategy.h"

// Initialize the static strategies map
std::map<std::string, std::function<std::unique_ptr<IntersectionStrategy>()>> IntersectionStrategyRegistry::strategies;

void IntersectionStrategyRegistry::registerStrategy(const std::string& name, std::function<std::unique_ptr<IntersectionStrategy>()> creator) {
    strategies[name] = creator;
}

std::unique_ptr<IntersectionStrategy> IntersectionStrategyRegistry::createStrategy(const std::string& name) {
    if (strategies.find(name) != strategies.end()) {
        return strategies[name]();
    }
    return nullptr;
}
