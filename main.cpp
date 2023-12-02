#include "./utils/IntersectionStrategyRegistry.h"
#include "./intersection001/BVHIntersection.h"
#include "./Scene.h"

void initializeStrategies() {
    IntersectionStrategyRegistry::registerStrategy("bvh", []() { return std::make_unique<BVHIntersectionStrategy>(); });

}


int main(int argc, char const *argv[])
{
    initializeStrategies();
    std::string path = "../models/Cute_triceratops.stl";

    // Model model;
    Triangle_model model;
    
    model.load_model(path);
    model.setIntersectionStrategy("bvh");
    model.init();
    return 0;
}
