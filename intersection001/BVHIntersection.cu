#include <stdio.h>
#include "./BVHIntersection.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <algorithm>
#include "../utils/cudaUtils.h"
#include <thrust/sequence.h>
#include <vector_functions.h>
#include "../utils/performance.h"
#include <fstream>



__device__ BoundingBox getTriBBox(float3 a, float3 b, float3 c) {
    float3 min = make_float3(fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)), fminf(a.z, fminf(b.z, c.z)));
    float3 max = make_float3(fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)), fmaxf(a.z, fmaxf(b.z, c.z)));
    return BoundingBox(min, max);
}

BVHIntersectionStrategy::BVHIntersectionStrategy() {
    // TODO Auto-generated constructor stub

}

BVHIntersectionStrategy::~BVHIntersectionStrategy() {
    // TODO Auto-generated destructor stub
}

void BVHIntersectionStrategy::toDevice(const Model &model) {
    this->numTriangles = model.indices.size() / 3;
    gpuErrchk( cudaMalloc(&(this->vertices), model.vertices.size() * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(this->indices), model.indices.size() * sizeof(unsigned int)) );

    gpuErrchk( cudaMemcpy(this->vertices, model.vertices.data(), model.vertices.size() * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(this->indices, model.indices.data(), model.indices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice) );
}

void BVHIntersectionStrategy::setTriBBoxs(){
    gpuErrchk( cudaMalloc(&(this->BBoxs), this->numTriangles * sizeof(BoundingBox)) );

    thrust::device_ptr<float> vertices_ptr(this->vertices);
    thrust::device_ptr<unsigned int> indices_ptr(this->indices);
    thrust::device_ptr<BoundingBox> triBBoxs_ptr(this->BBoxs);
    thrust::transform(thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(this->numTriangles), triBBoxs_ptr, 
        [vertices_ptr, indices_ptr] __device__ (unsigned int i) {
            unsigned int i0 = indices_ptr[i * 3];
            unsigned int i1 = indices_ptr[i * 3 + 1];
            unsigned int i2 = indices_ptr[i * 3 + 2];
            float3 v0 = make_float3(vertices_ptr[i0 * 3], vertices_ptr[i0 * 3 + 1], vertices_ptr[i0 * 3 + 2]);
            float3 v1 = make_float3(vertices_ptr[i1 * 3], vertices_ptr[i1 * 3 + 1], vertices_ptr[i1 * 3 + 2]);
            float3 v2 = make_float3(vertices_ptr[i2 * 3], vertices_ptr[i2 * 3 + 1], vertices_ptr[i2 * 3 + 2]);
            return getTriBBox(v0, v1, v2);
        }
    );
}

struct minAccessor {
    __host__ __device__ float3 operator()(const BoundingBox& bbox) const {
        return bbox.getMin();
    }
};
struct maxAccessor {
    __host__ __device__ float3 operator()(const BoundingBox& bbox) const {
        return bbox.getMax();
    }
};

struct minFunctor {
    __host__ __device__ float3 operator()(const float3& a, const float3& b) const {
        return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    }
};
struct maxFunctor {
    __host__ __device__ float3 operator()(const float3& a, const float3& b) const {
        return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
    }
};

void BVHIntersectionStrategy::modelMinMax() {
    thrust::device_ptr<BoundingBox> triBBoxs_ptr(this->BBoxs);
    this->min = thrust::transform_reduce(triBBoxs_ptr, triBBoxs_ptr + this->numTriangles, 
                                        minAccessor(), make_float3(FLT_MAX, FLT_MAX, FLT_MAX), minFunctor());
    this->max = thrust::transform_reduce(triBBoxs_ptr, triBBoxs_ptr + this->numTriangles,
                                        maxAccessor(), make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), maxFunctor());
}


__device__
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}


__device__
unsigned int morton3D(float3 xyz){
    float x = xyz.x;
    float y = xyz.y;
    float z = xyz.z;
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

void BVHIntersectionStrategy::computeMortonCodes(){
    gpuErrchk( cudaMalloc(&(this->object_ids), this->numTriangles * sizeof(unsigned int)) );
    thrust::device_ptr<unsigned int> object_ids_ptr(this->object_ids);
    thrust::sequence(object_ids_ptr, object_ids_ptr + this->numTriangles);


    thrust::device_ptr<BoundingBox> triBBoxs_ptr(this->BBoxs);
    gpuErrchk( cudaMalloc(&(this->mortonCodes), this->numTriangles * sizeof(unsigned int)) );
    thrust::device_ptr<unsigned int> mortonCodes_ptr(this->mortonCodes);
    thrust::transform(triBBoxs_ptr, triBBoxs_ptr + this->numTriangles, mortonCodes_ptr, 
        [] __device__ (const BoundingBox& bbox) {
            float3 min = bbox.getMin();
            float3 max = bbox.getMax();
            float3 center = make_float3((min.x + max.x) / 2.0f, (min.y + max.y) / 2.0f, (min.z + max.z) / 2.0f);
            return morton3D(center);
        }
    );
    thrust::sort_by_key(mortonCodes_ptr, mortonCodes_ptr + this->numTriangles, object_ids_ptr);

}

void BVHIntersectionStrategy::setupLeafNodes(){
    gpuErrchk( cudaMalloc(&(this->leafNodes), this->numTriangles * sizeof(LeafNode)) );
    thrust::device_ptr<LeafNode> leafNodes_ptr(this->leafNodes);
    thrust::device_ptr<unsigned int> object_ids_ptr(this->object_ids);
    // thrust::device_ptr<BoundingBox> triBBoxs_ptr(this->BBoxs);

    thrust::transform(thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(this->numTriangles), leafNodes_ptr, 
        [object_ids_ptr] __device__ (unsigned int i) {
            LeafNode leafNode;
            leafNode.isLeaf = true;
            leafNode.object_id = object_ids_ptr[i];
            leafNode.flag = 0;
            return leafNode;
        }
    );
}

__device__
int2 determineRange(unsigned int* sortedMortonCodes, int numTriangles, int idx)
{
    //determine the range of keys covered by each internal node (as well as its children)
     //direction is found by looking at the neighboring keys ki-1 , ki , ki+1
     //the index is either the beginning of the range or the end of the range
    int direction = 0;
    int common_prefix_with_left = 0;
    int common_prefix_with_right = 0;

    common_prefix_with_right = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]);
    if (idx == 0) {
        common_prefix_with_left = -1;
    }
    else
    {
        common_prefix_with_left = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - 1]);

    }

    direction = ((common_prefix_with_right - common_prefix_with_left) > 0) ? 1 : -1;
    int min_prefix_range = 0;

    if (idx == 0)
    {
        min_prefix_range = -1;

    }
    else
    {
        min_prefix_range = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - direction]);
    }

    int lmax = 2;
    int next_key = idx + lmax * direction;

    while ((next_key >= 0) && (next_key < numTriangles) && (__clz(sortedMortonCodes[idx] ^ sortedMortonCodes[next_key]) > min_prefix_range))
    {
        lmax *= 2;
        next_key = idx + lmax * direction;
    }
    //find the other end using binary search
    unsigned int l = 0;

    do
    {
        lmax = (lmax + 1) >> 1; // exponential decrease
        int new_val = idx + (l + lmax) * direction;

        if (new_val >= 0 && new_val < numTriangles)
        {
            unsigned int Code = sortedMortonCodes[new_val];
            int Prefix = __clz(sortedMortonCodes[idx] ^ Code);
            if (Prefix > min_prefix_range)
                l = l + lmax;
        }
    } while (lmax > 1);

    int j = idx + l * direction;

    int left = 0;
    int right = 0;

    if (idx < j) {
        left = idx;
        right = j;
    }
    else
    {
        left = j;
        right = idx;
    }

    // printf("idx : (%d) returning range (%d, %d) \n", idx, left, right);
    return make_int2(left, right);
}

__device__
int findSplit(unsigned int* sortedMortonCodes,
    int first,
    int last)
{
    // Identical Morton codes => split the range in the middle.
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}


__global__
void setupInternalNodesBBox(LeafNode* leafNodes, InternalNode* internalNode, BoundingBox* leafBBoxs, BoundingBox* internalBBoxs, int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;
    Node* pa = leafNodes[idx].parent;
    while (pa != NULL)
    {
        if(atomicCAS(&(pa->flag), 0, 1))
        {
            unsigned int index = pa - internalNode;
            BoundingBox* bbox = internalBBoxs + index;
            // bbox->bEmpty = true;

            if(pa->childA->isLeaf)
            {
                assert(pa->childA - leafNodes >= 0 && pa->childA - leafNodes < numTriangles);
                bbox->merge(leafBBoxs[static_cast<LeafNode*>(pa->childA)->object_id]);
            }
            else
            {
                assert(pa->childA - internalNode >= 0 && pa->childA - internalNode < numTriangles - 1);
                bbox->merge(internalBBoxs[pa->childA - internalNode]);
            }

            if(pa->childB->isLeaf)
            {
                assert(pa->childB - leafNodes >= 0 && pa->childB - leafNodes < numTriangles);
                bbox->merge(leafBBoxs[static_cast<LeafNode*>(pa->childB)->object_id]);
            }
            else
            {
                assert(pa->childB - internalNode >= 0 && pa->childB - internalNode < numTriangles - 1);
                bbox->merge(internalBBoxs[pa->childB - internalNode]);
            }
            pa = pa->parent;
        }
        else{
            return;
        }
    }
}


void BVHIntersectionStrategy::setupInternalNodes(){
    gpuErrchk( cudaMalloc(&(this->internalNodes), (this->numTriangles - 1) * sizeof(InternalNode)) );
    thrust::device_ptr<InternalNode> internalNodes_ptr(this->internalNodes);
    thrust::device_ptr<LeafNode> leafNodes_ptr(this->leafNodes);
   
    thrust::device_ptr<unsigned int> mortonCodes_ptr(this->mortonCodes);
    

    unsigned int tri_num = this->numTriangles;
    thrust::for_each(internalNodes_ptr, internalNodes_ptr + this->numTriangles - 1, 
        [ leafNodes_ptr, internalNodes_ptr, mortonCodes_ptr, tri_num ] __device__ (InternalNode& internalNode) {
            int idx = &internalNode - thrust::raw_pointer_cast(internalNodes_ptr);
            int2 range = determineRange(thrust::raw_pointer_cast(mortonCodes_ptr), tri_num, idx);
            int split = findSplit(thrust::raw_pointer_cast(mortonCodes_ptr), range.x, range.y);

            internalNode.childA = (split == range.x) ? static_cast<Node*>(& thrust::raw_pointer_cast(leafNodes_ptr)[split]) : static_cast<Node*>(& thrust::raw_pointer_cast(internalNodes_ptr)[split]);
            internalNode.childB = (split + 1 == range.y) ? static_cast<Node*>(& thrust::raw_pointer_cast(leafNodes_ptr)[split + 1]) : static_cast<Node*>(& thrust::raw_pointer_cast(internalNodes_ptr)[split + 1]);
            internalNode.childA->parent = &internalNode;
            internalNode.childB->parent = &internalNode;
            internalNode.isLeaf = false;
            internalNode.flag = 0;
        }
    );

    // 不能再下面的device函数中传入this，因为this是在host上面的指针，而device函数是在device上面执行的
    // thrust::transform(thrust::make_counting_iterator<unsigned int>(0), 
    //     thrust::make_counting_iterator<unsigned int>(this->numTriangles - 1), 
    //     internalNodes_ptr, 
    //     [ leafNodes_ptr, internalNodes_ptr, mortonCodes_ptr, tri_num ] __device__ (unsigned int i) {
    //         InternalNode internalNode;

    //         int2 range = determineRange(thrust::raw_pointer_cast(mortonCodes_ptr), tri_num, i);
    //         int split = findSplit(thrust::raw_pointer_cast(mortonCodes_ptr), range.x, range.y);

    //         internalNode.childA = (split == range.x) ? static_cast<Node*>(& thrust::raw_pointer_cast(leafNodes_ptr)[split]) : 
    //                                                     static_cast<Node*>(& thrust::raw_pointer_cast(internalNodes_ptr)[split]);
    //         internalNode.childB = (split + 1 == range.y) ? static_cast<Node*>(& thrust::raw_pointer_cast(leafNodes_ptr)[split + 1]) :
    //                                                     static_cast<Node*>(& thrust::raw_pointer_cast(internalNodes_ptr)[split + 1]);

    //         // 因为最终返回的internalNode会覆盖掉下面的设置
    //         internalNode.childA->parent = static_cast<Node*>(thrust::raw_pointer_cast(internalNodes_ptr)[i]);
    //         internalNode.childB->parent = static_cast<Node*>(thrust::raw_pointer_cast(internalNodes_ptr)[i]);
    //         return internalNode;
    //     }
    // );

    std::cout << "setupInternalNodes done" << std::endl;

    gpuErrchk( cudaMalloc(&(this->internalBBoxs), (this->numTriangles - 1) * sizeof(BoundingBox)) );
    setupInternalNodesBBox<<<(this->numTriangles - 1 + 255) / 256, 256>>>(this->leafNodes, this->internalNodes, this->BBoxs, this->internalBBoxs, this->numTriangles);
    gpuErrchk( cudaDeviceSynchronize() );
}

void BVHIntersectionStrategy::intersect(const Ray &ray, Intersection &intersection) {
    

}

void BVHIntersectionStrategy::draw() {
    BoundingBox* leaf_bboxs = new BoundingBox[this->numTriangles];
    BoundingBox* internal_bboxs = new BoundingBox[this->numTriangles-1];

    LeafNode* leafNodes = new LeafNode[this->numTriangles];
    InternalNode* internalNodes = new InternalNode[this->numTriangles-1];

    cudaMemcpy(leaf_bboxs, this->BBoxs, this->numTriangles * sizeof(BoundingBox), cudaMemcpyDeviceToHost);
    cudaMemcpy(internal_bboxs, this->internalBBoxs, (this->numTriangles - 1) * sizeof(BoundingBox), cudaMemcpyDeviceToHost);

    // 不能这样进行移动，因为在GPU上面类如果有指针的话，指针是指向GPU上面的内存的，而不是CPU上面的内存
    cudaMemcpy(leafNodes, this->leafNodes, this->numTriangles * sizeof(LeafNode), cudaMemcpyDeviceToHost);
    cudaMemcpy(internalNodes, this->internalNodes, (this->numTriangles - 1) * sizeof(InternalNode), cudaMemcpyDeviceToHost);
    

    std::ofstream file("/mnt/h/coding/cuda_inter/models/bvh.txt");
    for(int i = 0; i < this->numTriangles; i++)
    {
        file << leaf_bboxs[i].getMin().x << " " << leaf_bboxs[i].getMin().y << " " << leaf_bboxs[i].getMin().z << " " 
        << leaf_bboxs[i].getMax().x << " " << leaf_bboxs[i].getMax().y << " " << leaf_bboxs[i].getMax().z << std::endl;
    }
    file.close();

    std::ofstream file2("/mnt/h/coding/cuda_inter/models/bvh2.txt");
    for(int i = 0; i < this->numTriangles - 1; i++)
    {
        file2 << internal_bboxs[i].getMin().x << " " << internal_bboxs[i].getMin().y << " " << internal_bboxs[i].getMin().z << " " 
        << internal_bboxs[i].getMax().x << " " << internal_bboxs[i].getMax().y << " " << internal_bboxs[i].getMax().z << std::endl;
    }
    file2.close();


    std::ofstream file3("/mnt/h/coding/cuda_inter/models/tree.txt");
    for(int i = 0; i < this->numTriangles - 1; i++)
    {
        // internal node index
        file3 << "internal node " << i << std::endl;
    

        file3 << "childA: " << static_cast<Node*>(internalNodes[i].childA) - static_cast<Node*>(this->internalNodes) <<
        " childB: " << static_cast<Node*>(internalNodes[i].childB) - static_cast<Node*>(this->internalNodes) << std::endl;
        file3 << "parent: " << static_cast<Node*>(internalNodes[i].parent) - static_cast<Node*>(this->internalNodes) << std::endl;
    }
    file3.close();
    

}