#pragma once
#include "./base.h"

class IntersectionStrategy {
public:
    virtual ~IntersectionStrategy() {}
    virtual void init(const Model& model) = 0;
    virtual void intersect(const Ray& r, Intersection& isect) = 0;
};

