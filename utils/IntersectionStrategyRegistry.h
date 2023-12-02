// IntersectionStrategyRegistry.h
#ifndef INTERSECTION_STRATEGY_REGISTRY_H
#define INTERSECTION_STRATEGY_REGISTRY_H

#include <map>
#include <string>
#include <functional>
#include <memory>

class IntersectionStrategy;

class IntersectionStrategyRegistry {
private:
    static std::map<std::string, std::function<std::unique_ptr<IntersectionStrategy>()>> strategies;

public:
    static void registerStrategy(const std::string& name, std::function<std::unique_ptr<IntersectionStrategy>()> creator);
    static std::unique_ptr<IntersectionStrategy> createStrategy(const std::string& name);
};

#endif
