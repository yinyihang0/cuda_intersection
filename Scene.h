#include <string>
#include "./utils/base.h"
#include <memory>
#include "./utils/IntersectionStrategy.h"
#include "./utils/IntersectionStrategyRegistry.h"
#include "./utils/stl_reader.h"
#include <iostream>

class Triangle_model{
private:
    Model model;
    std::unique_ptr<IntersectionStrategy> intersectionStrategy;

public:
    // Triangle_model();
    // ~Triangle_model();
    void load_model(std::string path){
        std::string type = path.substr(path.find_last_of(".") + 1);
        std::vector<unsigned int> soilds;
        if(type == "stl"){
            try{
                stl_reader::ReadStlFile(path.c_str(), model.vertices, model.normals, model.indices, soilds);
                std::cout << "coordinates number: " << model.vertices.size()/3 << std::endl;
                std::cout << "triangles number: " << model.indices.size()/3 << std::endl;
            }
            catch(const std::exception& e){
                std::cerr << e.what() << '\n';
            }
        }
        else{
            std::cout << "Error: Invalid model type" << std::endl;
        }
    }

    void setIntersectionStrategy(const std::string& name){
        intersectionStrategy = IntersectionStrategyRegistry::createStrategy(name);
    }
    void init(){
        intersectionStrategy->init(model);
    }
    void intersect(const Ray& r, Intersection& isect){
        intersectionStrategy->intersect(r, isect);
    }
    // void packet_intersect();
};