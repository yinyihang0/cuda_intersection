#include "../utils/base.h"
#include "../utils/IntersectionStrategy.h"
#include "../utils/IntersectionStrategyRegistry.h"
#include "../utils/BoundingBox.h"
#include "../utils/cudaUtils.h"
#include "../utils/performance.h"
#include "../utils/Visualization.h"

#include <cuda_runtime.h>

struct Node{
    Node* childA;
    Node* childB;
    Node* parent;

    int flag;
    bool isLeaf;
    // BoundingBox bbox;

    __device__ __host__
    Node() : childA(NULL), childB(NULL), parent(NULL), flag(0), isLeaf(false) {}
};
struct LeafNode : public Node{
    unsigned int object_id;
    __device__ __host__
    LeafNode() : Node() {
        isLeaf = true;
    }
};
struct InternalNode : public Node{
    __device__ __host__
    InternalNode() : Node() {
        isLeaf = false;
    }
};


class BVHIntersectionStrategy : public IntersectionStrategy, public Visualization{
private:
    // generated on device
    unsigned int* mortonCodes;
    unsigned int* object_ids;
    LeafNode* leafNodes; //numTriangles
    InternalNode* internalNodes; //numTriangles - 1

    BoundingBox* BBoxs;
    BoundingBox* internalBBoxs;
    float3 min;
    float3 max;

    // data import from model
    int numTriangles;
    float* vertices;
    unsigned int* indices;

public:
    BVHIntersectionStrategy();
    ~BVHIntersectionStrategy();


    void toDevice(const Model& model);
    void modelMinMax();
    void setTriBBoxs();

    void computeMortonCodes();
    void setupLeafNodes();
    void setupInternalNodes();
    void init(const Model& model) override {
        std::cout << "BVHIntersectionStrategy init" << std::endl;
        TIMER_FUNC( this->toDevice(model) );
        TIMER_FUNC( this->setTriBBoxs() );
        TIMER_FUNC( this->modelMinMax() );
        TIMER_FUNC( this->computeMortonCodes() );
        TIMER_FUNC( this->setupLeafNodes() );
        TIMER_FUNC( this->setupInternalNodes() );
        std::cout << "BVHIntersectionStrategy init done" << std::endl;
        this->draw(); 
    }
    void intersect(const Ray& r, Intersection& isect) override;

    void draw() override;



};