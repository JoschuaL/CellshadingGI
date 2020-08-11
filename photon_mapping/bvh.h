#ifndef BVH_H
#define BVH_H

#include <memory>
#include <cstdint>

#include "float3.h"
#include "intersect.h"
#include "bbox.h"

/// Bounding Volume Hierarchy.
class Bvh {
public:
    /// Builds a BVH given a list of vertices and a list of indices.
    void build(const float3* verts, const uint32_t* indices, size_t num_tris);

    /// Traverses the BVH in order to find the closest intersection, or any intersection if 'any' is set.
    void traverse(const Ray& ray, Hit& hit, bool any = false) const;

    /// Returns the number of nodes in the BVH.
    size_t node_count() const { return num_nodes; }

private:
    void build(const BBox*, const float3*, size_t);

    friend struct BvhBuilder;

    struct Node {
        float3 min;             ///< Min. BB corners
        union {
            int32_t child;      ///< Index of the first child, children are located next to each other in memory
            int32_t first_prim; ///< Index of the first primitive in the leaf
        };
        float3 max;             ///< Max. BB corners
        union {
            int32_t num_prims;  ///< Number of primitives for a leaf
            int32_t axis;       ///< Axis on which the inner node was split (negative value to distinguish leaves from inner nodes)
        };
    };

    std::unique_ptr<Node[]>           nodes;
    std::unique_ptr<uint32_t[]>       prim_ids;
    std::unique_ptr<PrecomputedTri[]> tris;
    size_t                            num_nodes;
};

#endif // BVH_H
